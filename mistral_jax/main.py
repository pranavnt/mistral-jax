import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import random
from transformers import AutoTokenizer
from dataclasses import dataclass
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class LayerWeights(NamedTuple):
    attention_norm_D: jnp.ndarray
    ffn_norm_D: jnp.ndarray
    attention_wq_DHK: jnp.ndarray
    attention_wk_DHK: jnp.ndarray
    attention_wv_DHK: jnp.ndarray
    attention_wo_HDK: jnp.ndarray
    ffn_w1_DF: jnp.ndarray
    ffn_w2_FD: jnp.ndarray
    ffn_w3_DF: jnp.ndarray

class TransformerWeights(NamedTuple):
    token_embedding_VD: jnp.ndarray
    layer_weights: List[LayerWeights]
    norm_D: jnp.ndarray
    output_DV: jnp.ndarray

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

def mha(queries_BLHK: jnp.ndarray, keys_BLHK: jnp.ndarray, values_BLHK: jnp.ndarray, mask_BLL: jnp.ndarray, head_dim: int):
    scores_BHLM = jnp.einsum('blhk,bmhk->bhlm', queries_BLHK, keys_BLHK)
    scores_BHLM = scores_BHLM / (head_dim ** 0.5)

    scores_BHLM = jnp.where(mask_BLL[:, jnp.newaxis, :, :], scores_BHLM, jnp.finfo(scores_BHLM.dtype).min)

    scores_BHLM = jax.nn.softmax(scores_BHLM, axis=-1)
    output_BLHK = jnp.einsum('bhlm,bmhk->blhk', scores_BHLM, values_BLHK)
    return output_BLHK

@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def gqa(x_BLD: jnp.ndarray, wq_DHK, wk_DHK, wv_DHK, wo_HDK, freqs_cis_LK2: jnp.ndarray,
        head_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int):
    batch_size, seqlen_sum, _ = x_BLD.shape

    assert n_heads % n_kv_heads == 0

    queries_BLHK, keys_BLHK, values_BLHK = x_BLD @ wq_DHK, x_BLD @ wk_DHK, x_BLD @ wv_DHK

    queries_BLHK = queries_BLHK.reshape(batch_size, seqlen_sum, n_heads, head_dim)
    keys_BLHK = keys_BLHK.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)
    values_BLHK = values_BLHK.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)

    queries_BLHK, keys_BLHK = apply_rotary_emb(queries_BLHK, keys_BLHK, freqs_cis_LK2)

    repeats = n_heads // n_kv_heads
    keys_BLHK, values_BLHK = repeat_kv(keys_BLHK, values_BLHK, repeats, dim=2)

    mask_BLL = create_mask(batch_size, seqlen_sum, sliding_window)
    output_BLHK = mha(queries_BLHK, keys_BLHK, values_BLHK, mask_BLL, head_dim)

    output_BLD = output_BLHK.reshape(batch_size, seqlen_sum, n_heads * head_dim)
    output_BLD = output_BLD @ wo_HDK

    return output_BLD

@jax.jit
def ffn(x_BLD: jnp.ndarray, w1_DF: jnp.ndarray, w2_FD: jnp.ndarray, w3_DF: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum('bsd,dh->bsh', jax.nn.silu(jnp.einsum('bsd,dh->bsh', x_BLD, w1_DF)) * jnp.einsum('bsd,dh->bsh', x_BLD, w3_DF), w2_FD)

@jax.jit
def rms_norm(x_BLD: jnp.ndarray, weight_D: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    ms_B1D = jnp.mean(jnp.square(x_BLD), axis=-1, keepdims=True)
    x_normed_BLD = x_BLD * jax.lax.rsqrt(ms_B1D + eps)
    return x_normed_BLD * weight_D

def restructure_layer_weights(layer_weights):
    return jax.tree.map(lambda *arrays: jnp.stack(arrays), *layer_weights)

def apply_layer(h_BLD: jnp.ndarray, layer_weights, freqs_cis_LK2: jnp.ndarray,
                head_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int) -> jnp.ndarray:
    # Attention
    h_BLD_attn = rms_norm(h_BLD, layer_weights.attention_norm_D)
    h_BLD_attn = gqa(h_BLD_attn,
                     layer_weights.attention_wq_DHK,
                     layer_weights.attention_wk_DHK,
                     layer_weights.attention_wv_DHK,
                     layer_weights.attention_wo_HDK,
                     freqs_cis_LK2,
                     head_dim, n_heads, n_kv_heads, sliding_window)
    h_BLD = h_BLD + h_BLD_attn

    # Feed-forward
    h_BLD_ffn = rms_norm(h_BLD, layer_weights.ffn_norm_D)
    h_BLD_ffn = ffn(h_BLD_ffn,
                    layer_weights.ffn_w1_DF,
                    layer_weights.ffn_w2_FD,
                    layer_weights.ffn_w3_DF)
    h_BLD = h_BLD + h_BLD_ffn

    return h_BLD

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def transformer(params: TransformerWeights, x: jnp.ndarray, freqs_cis_LK2: Tuple[jnp.ndarray, jnp.ndarray],
                head_dim: int, n_heads: int, n_kv_heads: int, sliding_window: int,
                max_seq_len: int) -> jnp.ndarray:
    h = params.token_embedding_VD[x]

    def scan_fn(carry, layer_weights):
        h = apply_layer(carry, layer_weights, freqs_cis_LK2,
                        head_dim, n_heads, n_kv_heads, sliding_window)
        return h, None

    h, _ = jax.lax.scan(scan_fn, h, params.layer_weights)
    h = rms_norm(h, params.norm_D)
    return h @ params.output_DV

def generate(params: TransformerWeights, config: Dict[str, Any], tokenizer: Any, prompt: str, max_new_tokens: int = 20, temperature: float = 0.7) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="jax").squeeze()
    x = input_ids[None, :]

    max_seq_len = config['sliding_window']
    freqs_cis = compute_freqs_cis(config['head_dim'], max_seq_len, config['rope_theta'])

    head_dim = config['head_dim']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    sliding_window = config['sliding_window']

    for _ in range(max_new_tokens):
        x = x[:, -sliding_window:]
        logits = transformer(params, x, freqs_cis, head_dim, n_heads, n_kv_heads, sliding_window, max_seq_len)
        next_token_logits = logits[0, -1] / temperature
        next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_logits)
        x = jnp.concatenate([x, next_token[None, None]], axis=1)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(x[0, len(input_ids):], skip_special_tokens=True)

def load_model(model_path: str) -> Tuple[TransformerWeights, Dict[str, Any]]:
    with open(Path(model_path) / "params.json", "r") as f:
        config = json.load(f)

    layers = []
    token_embedding_VD = norm_D = output_DV = None

    state_dict = torch.load(str(Path(model_path) / "consolidated.00.pth"), map_location='cpu')

    def torch_to_numpy(t):
        return np.array(t.cpu().to(torch.float32).detach().numpy())

    for k, v in state_dict.items():
        if 'layers' in k:
            parts = k.split('.')
            layer_num = int(parts[1])

            while len(layers) <= layer_num:
                layers.append(LayerWeights(**{f: None for f in LayerWeights._fields}))

            layer_params = layers[layer_num]
            if 'attention.wq.weight' in k:
                layer_params = layer_params._replace(attention_wq_DHK=jnp.array(torch_to_numpy(v)).T)
            elif 'attention.wk.weight' in k:
                layer_params = layer_params._replace(attention_wk_DHK=jnp.array(torch_to_numpy(v)).T)
            elif 'attention.wv.weight' in k:
                layer_params = layer_params._replace(attention_wv_DHK=jnp.array(torch_to_numpy(v)).T)
            elif 'attention.wo.weight' in k:
                layer_params = layer_params._replace(attention_wo_HDK=jnp.array(torch_to_numpy(v)).T)
            elif 'feed_forward.w1.weight' in k:
                layer_params = layer_params._replace(ffn_w1_DF=jnp.array(torch_to_numpy(v)).T)
            elif 'feed_forward.w2.weight' in k:
                layer_params = layer_params._replace(ffn_w2_FD=jnp.array(torch_to_numpy(v)).T)
            elif 'feed_forward.w3.weight' in k:
                layer_params = layer_params._replace(ffn_w3_DF=jnp.array(torch_to_numpy(v)).T)
            elif 'attention_norm.weight' in k:
                layer_params = layer_params._replace(attention_norm_D=jnp.array(torch_to_numpy(v)))
            elif 'ffn_norm.weight' in k:
                layer_params = layer_params._replace(ffn_norm_D=jnp.array(torch_to_numpy(v)))

            layers[layer_num] = layer_params
        elif 'tok_embeddings.weight' in k:
            token_embedding_VD = jnp.array(torch_to_numpy(v))
        elif 'norm.weight' in k:
            norm_D = jnp.array(torch_to_numpy(v))
        elif 'output.weight' in k:
            output_DV = jnp.array(torch_to_numpy(v)).T

    params = TransformerWeights(
        token_embedding_VD=token_embedding_VD,
        layer_weights=restructure_layer_weights(layers),
        norm_D=norm_D,
        output_DV=output_DV
    )

    return params, config

if __name__ == "__main__":
    model_path = "./mistral-7B-v0.3/"

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

        output = generate(params, config, tokenizer, prompt, max_new_tokens=10, temperature=0.7)

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