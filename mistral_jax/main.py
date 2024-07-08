import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import random
from transformers import AutoTokenizer

"""
Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

config = {
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "max_seq_len": 4096,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
    "rope_theta": 1000000.0,
    "sliding_window": 32768
}

def compute_freqs_cis(dim: int, max_pos: int, theta: float = 10000.0) -> jnp.ndarray:
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim)
    )
    t = jnp.arange(0, max_pos)
    freqs = jnp.outer(t, inv_freq)
    freqs_cis = (jnp.cos(freqs), jnp.sin(freqs))
    return freqs_cis


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
#     print(dim)
#     freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
#     t = jnp.arange(end)
#     freqs_LD = jnp.outer(t, freqs).astype(jnp.float32)
#     cos_LD, sin_LD = jnp.cos(freqs_LD), jnp.sin(freqs_LD)
#     return jnp.stack([cos_LD, sin_LD], axis=-1)

def apply_rotary_emb(
    xq_BLHK: jnp.ndarray,
    xk_BLHK: jnp.ndarray,
    freqs_cis_LK2: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    freqs_cos_LK, freqs_sin_LK = freqs_cis_LK2

    def rotate(x_BLHK):
        B, L, H, K = x_BLHK.shape
        x1_BLHK, x2_BLHK = jnp.split(x_BLHK, 2, axis=-1)
        K2 = K // 2

        freqs_cos = freqs_cos_LK[:L, :K2].reshape(1, L, 1, K2)
        freqs_sin = freqs_sin_LK[:L, :K2].reshape(1, L, 1, K2)

        pos_embed_1 = x1_BLHK * freqs_cos - x2_BLHK * freqs_sin
        pos_embed_2 = x1_BLHK * freqs_sin + x2_BLHK * freqs_cos

        pos_embed = jnp.concatenate([pos_embed_1, pos_embed_2], axis=-1)
        return pos_embed

    return rotate(xq_BLHK), rotate(xk_BLHK)

def repeat_kv(
    keys_BLHK: jnp.ndarray, values_BLHK: jnp.ndarray, repeats: int, dim: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    keys_BLHK = jnp.repeat(keys_BLHK, repeats=repeats, axis=dim)
    values_BLHK = jnp.repeat(values_BLHK, repeats=repeats, axis=dim)
    return keys_BLHK, values_BLHK

def create_mask(batch_size: int, seqlen: int, sliding_window: int) -> jnp.ndarray:
    mask_LL = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))

    if sliding_window is not None and sliding_window < seqlen:
        window_mask_LL = jnp.triu(jnp.ones((seqlen, seqlen), dtype=bool), k=-sliding_window)
        mask_LL = jnp.logical_and(mask_LL, window_mask_LL)

    return jnp.repeat(mask_LL[jnp.newaxis, :, :], batch_size, axis=0)

def mha(queries_BLHK: jnp.ndarray, keys_BLHK: jnp.ndarray, values_BLHK: jnp.ndarray, mask_BLL: jnp.ndarray, config: Dict[str, Any]):
    head_dim = config['head_dim']
    scores_BHLM = jnp.einsum('blhk,bmhk->bhlm', queries_BLHK, keys_BLHK)
    scores_BHLM = scores_BHLM / (head_dim ** 0.5)

    scores_BHLM = jnp.where(mask_BLL[:, jnp.newaxis, :, :], scores_BHLM, jnp.finfo(scores_BHLM.dtype).min)

    scores_BHLM = jax.nn.softmax(scores_BHLM, axis=-1)
    output_BLHK = jnp.einsum('bhlm,bmhk->blhk', scores_BHLM, values_BLHK)
    return output_BLHK

def gqa(x_BLD: jnp.ndarray, wq_DHK, wk_DHK, wv_DHK, wo_HDK, freqs_cis_LK2: jnp.ndarray, config: Dict[str, Any]):
    batch_size, seqlen_sum, _ = x_BLD.shape
    n_heads, n_kv_heads, head_dim = config['n_heads'], config['n_kv_heads'], config['head_dim']

    assert n_heads % n_kv_heads == 0

    queries_BLHK, keys_BLHK, values_BLHK = x_BLD @ wq_DHK, x_BLD @ wk_DHK, x_BLD @ wv_DHK

    queries_BLHK = queries_BLHK.reshape(batch_size, seqlen_sum, n_heads, head_dim)
    keys_BLHK = keys_BLHK.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)
    values_BLHK = values_BLHK.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)

    queries_BLHK, keys_BLHK = apply_rotary_emb(queries_BLHK, keys_BLHK, freqs_cis_LK2)

    repeats = n_heads // n_kv_heads
    keys_BLHK, values_BLHK = repeat_kv(keys_BLHK, values_BLHK, repeats, dim=2)

    mask_BLL = create_mask(batch_size, seqlen_sum, config['sliding_window'])
    output_BLHK = mha(queries_BLHK, keys_BLHK, values_BLHK, mask_BLL, config)

    output_BLD = output_BLHK.reshape(batch_size, seqlen_sum, n_heads * head_dim)
    output_BLD = output_BLD @ wo_HDK

    return output_BLD

def ffn(x_BLD: jnp.ndarray, w1_DF: jnp.ndarray, w2_FD: jnp.ndarray, w3_DF: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum('bsd,dh->bsh', jax.nn.silu(jnp.einsum('bsd,dh->bsh', x_BLD, w1_DF)) * jnp.einsum('bsd,dh->bsh', x_BLD, w3_DF), w2_FD)

def rms_norm(x_BLD: jnp.ndarray, weight_D: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    ms_B1D = jnp.mean(jnp.square(x_BLD), axis=-1, keepdims=True)
    x_normed_BLD = x_BLD * jax.lax.rsqrt(ms_B1D + eps)
    return x_normed_BLD * weight_D

def transformer(params: Dict[str, Any], x_BL: jnp.ndarray, freqs_cis_LK2: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
    x_BLD = params['token_embedding_VD'][x_BL]

    for layer_params in params['layers']:
        h_BLD = rms_norm(x_BLD, layer_params['attention_norm_D'])
        h_BLD = gqa(h_BLD, layer_params['attention_wq_DHK'], layer_params['attention_wk_DHK'],
                layer_params['attention_wv_DHK'], layer_params['attention_wo_HDK'], freqs_cis_LK2, config)
        h_BLD = x_BLD + h_BLD
        x_BLD = rms_norm(h_BLD, layer_params['ffn_norm_D'])
        h_BLD = ffn(x_BLD, layer_params['ffn_w1_DF'], layer_params['ffn_w2_FD'], layer_params['ffn_w3_DF'])
        x_BLD = x_BLD + h_BLD

    x_BLD = rms_norm(x_BLD, params['norm_D'])
    return x_BLD @ params['output_DV']

def init_params(config: Dict[str, Any], key: jnp.ndarray) -> Dict[str, Any]:
    dim, n_heads, n_kv_heads = config['dim'], config['n_heads'], config['n_kv_heads']
    head_dim, hidden_dim = config['head_dim'], config['hidden_dim']
    vocab_size, n_layers = config['vocab_size'], config['n_layers']

    def init_weight(key, shape):
        return random.normal(key, shape) * 0.02

    params = {}
    key, *subkeys = random.split(key, n_layers * 13 + 4)

    params['token_embedding_VD'] = init_weight(subkeys[0], (vocab_size, dim))
    params['norm_D'] = init_weight(subkeys[1], (dim,))
    params['output_DV'] = init_weight(subkeys[2], (dim, vocab_size))

    params['layers'] = []

    for i in range(n_layers):
        layer_params = {}
        layer_params['attention_norm_D'] = init_weight(subkeys[i*13+3], (dim,))
        layer_params['attention_wq_DHK'] = init_weight(subkeys[i*13+4], (dim, n_heads * head_dim))
        layer_params['attention_wk_DHK'] = init_weight(subkeys[i*13+5], (dim, n_kv_heads * head_dim))
        layer_params['attention_wv_DHK'] = init_weight(subkeys[i*13+6], (dim, n_kv_heads * head_dim))
        layer_params['attention_wo_HDK'] = init_weight(subkeys[i*13+7], (n_heads * head_dim, dim))
        layer_params['ffn_norm_D'] = init_weight(subkeys[i*13+8], (dim,))
        layer_params['ffn_w1_DF'] = init_weight(subkeys[i*13+9], (hidden_dim, dim))
        layer_params['ffn_w2_FD'] = init_weight(subkeys[i*13+10], (dim, hidden_dim))
        layer_params['ffn_w3_DF'] = init_weight(subkeys[i*13+11], (hidden_dim, dim))
        params['layers'].append(layer_params)

    return params

def load_model(model_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(Path(model_path) / "params.json", "r") as f:
        config = json.load(f)

    params = {
        'layers': []
    }

    state_dict = torch.load(str(Path(model_path) / "consolidated.00.pth"), map_location='cpu')

    def torch_to_numpy(t):
        return np.array(t.cpu().to(torch.float32).detach().numpy())

    for k, v in state_dict.items():
        if 'layers' in k:
            parts = k.split('.')
            layer_num = int(parts[1])

            while len(params['layers']) <= layer_num:
                params['layers'].append({})

            layer_params = params['layers'][layer_num]

            if 'attention_norm.weight' in k:
                layer_params['attention_norm_D'] = jnp.array(torch_to_numpy(v))
            elif 'ffn_norm.weight' in k:
                layer_params['ffn_norm_D'] = jnp.array(torch_to_numpy(v))
            elif 'attention.wq.weight' in k:
                layer_params['attention_wq_DHK'] = jnp.array(torch_to_numpy(v)).T
            elif 'attention.wk.weight' in k:
                layer_params['attention_wk_DHK'] = jnp.array(torch_to_numpy(v)).T
            elif 'attention.wv.weight' in k:
                layer_params['attention_wv_DHK'] = jnp.array(torch_to_numpy(v)).T
            elif 'attention.wo.weight' in k:
                layer_params['attention_wo_HDK'] = jnp.array(torch_to_numpy(v)).T
            elif 'feed_forward.w1.weight' in k:
                layer_params['ffn_w1_DF'] = jnp.array(torch_to_numpy(v)).T
            elif 'feed_forward.w2.weight' in k:
                layer_params['ffn_w2_FD'] = jnp.array(torch_to_numpy(v)).T
            elif 'feed_forward.w3.weight' in k:
                layer_params['ffn_w3_DF'] = jnp.array(torch_to_numpy(v)).T
        elif 'tok_embeddings.weight' in k:
            params['token_embedding_VD'] = jnp.array(torch_to_numpy(v))
        elif 'norm.weight' in k:
            params['norm_D'] = jnp.array(torch_to_numpy(v))
        elif 'output.weight' in k:
            params['output_DV'] = jnp.array(torch_to_numpy(v)).T

    return params, config

def generate(params: Dict[str, Any], config: Dict[str, Any], tokenizer: Any, prompt: str, max_new_tokens: int = 20, temperature: float = 0.7) -> str:
    input_ids_L = tokenizer.encode(prompt, return_tensors="jax").squeeze()

    x_BL = input_ids_L[None, :]

    max_seq_len = config['sliding_window']
    freqs_cis_LK2 = compute_freqs_cis(
        config['head_dim'],
        max_seq_len,
        config['rope_theta']
    )

    for _ in range(max_new_tokens):
        x_BL = x_BL[:, -config['sliding_window']:]

        logits_BLV = transformer(params, x_BL, freqs_cis_LK2[:x_BL.shape[1]], config)

        next_token_logits_V = logits_BLV[0, -1]

        next_token_logits_V = next_token_logits_V / temperature

        next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_logits_V)

        x_BL = jnp.concatenate([x_BL, next_token[None, None]], axis=1)

        if next_token == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(x_BL[0, len(input_ids_L):], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    model_path = "../sagasu/mistral-7B-v0.2/"

    print("Loading model...")
    params, config = load_model(model_path)

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    print("Generating text...")
    prompts = [
        "Once upon a time in a land far, far away,",
        "The solution to climate change is",
        "In the year 2050, artificial intelligence has",
        "The most important scientific discovery of the 21st century is"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")

        output = generate(params, config, tokenizer, prompt, max_new_tokens=50, temperature=0.7)

        print(f"Generated text: {output}")
        print("-" * 50)

    print("\nTesting complete!")

    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == 'exit':
            break

        output = generate(params, config, tokenizer, user_prompt, max_new_tokens=100, temperature=0.7)
        print(f"\nGenerated text: {output}")

    print("Thank you for using the language model. Goodbye!")