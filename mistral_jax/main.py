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

config = {
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
    "rope_theta": 1000000.0,
    "sliding_window": 32768
}

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return jnp.stack([cos, sin], axis=-1)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def rotate(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        freqs_cos, freqs_sin = jnp.split(freqs_cis, 2, axis=-1)

        print(x1.shape)
        print(freqs_cos.shape)
        freqs_cos = freqs_cos[:x.shape[1]]  # trim to seq length
        freqs_sin = freqs_sin[:x.shape[1]]  # trim to seq length

        freqs_cos = jnp.expand_dims(freqs_cos, axis=(0, 2))
        freqs_sin = jnp.expand_dims(freqs_sin, axis=(0, 2))


        return jnp.concatenate([
            x1 * freqs_cos - x2 * freqs_sin,
            x1 * freqs_sin + x2 * freqs_cos
        ], axis=-1)

    return rotate(xq), rotate(xk)

def repeat_kv(
    keys: jnp.ndarray, values: jnp.ndarray, repeats: int, dim: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    keys = jnp.repeat(keys, repeats=repeats, axis=dim)
    values = jnp.repeat(values, repeats=repeats, axis=dim)
    return keys, values

def create_mask(batch_size: int, seqlen: int, sliding_window: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))

    if sliding_window is not None and sliding_window < seqlen:
        window_mask = jnp.triu(jnp.ones((seqlen, seqlen), dtype=bool), k=-sliding_window)
        mask = jnp.logical_and(mask, window_mask)

    return jnp.repeat(mask[jnp.newaxis, :, :], batch_size, axis=0)

def mha(queries: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, mask: jnp.ndarray, config: Dict[str, Any]):
    head_dim = config['head_dim']
    scores = jnp.einsum('bnhqd,bnhkd->bnhqk', queries, keys)
    scores = scores / (head_dim ** 0.5)

    scores = jnp.where(mask[:, jnp.newaxis, :, :], scores, jnp.finfo(scores.dtype).min)

    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('bnhqk,bnhkd->bnhqd', scores, values)
    return output

def gqa(x: jnp.ndarray, wq, wk, wv, wo, freqs_cis: jnp.ndarray, config: Dict[str, Any]):
    batch_size, seqlen_sum, _ = x.shape
    n_heads, n_kv_heads, head_dim = config['n_heads'], config['n_kv_heads'], config['head_dim']

    assert n_heads % n_kv_heads == 0

    queries, keys, values = x @ wq, x @ wk, x @ wv

    queries = queries.reshape(batch_size, seqlen_sum, n_heads, head_dim)
    keys = keys.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)
    values = values.reshape(batch_size, seqlen_sum, n_kv_heads, head_dim)

    queries, keys = apply_rotary_emb(queries, keys, freqs_cis)

    repeats = n_heads // n_kv_heads
    keys, values = repeat_kv(keys, values, repeats, dim=2)

    mask = create_mask(batch_size, seqlen_sum, config['sliding_window'])
    output = mha(queries, keys, values, mask, config)

    output = output.reshape(batch_size, seqlen_sum, n_heads * head_dim)
    output = output @ wo

    return output

def ffn(x: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray, w3: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum('bsd,dh->bsh', jax.nn.silu(jnp.einsum('bsd,dh->bsh', x, w1)) * jnp.einsum('bsd,dh->bsh', x, w3), w2)

def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(ms + eps)
    return x_normed * weight



def transformer(params: Dict[str, Any], x: jnp.ndarray, freqs_cis: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
    x = params['token_embedding'][x]

    for layer_params in params['layers']:
        h = rms_norm(x, layer_params['attention_norm'])
        h = gqa(h, layer_params['attention_wq'], layer_params['attention_wk'],
                layer_params['attention_wv'], layer_params['attention_wo'], freqs_cis, config)
        h = x + h
        x = rms_norm(h, layer_params['ffn_norm'])
        h = ffn(x, layer_params['ffn_w1'], layer_params['ffn_w2'], layer_params['ffn_w3'])
        x = x + h

    x = rms_norm(x, params['norm'])
    return x @ params['output']


def init_params(config: Dict[str, Any], key: jnp.ndarray) -> Dict[str, Any]:
    dim = config['dim']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = config['head_dim']
    hidden_dim = config['hidden_dim']
    vocab_size = config['vocab_size']
    n_layers = config['n_layers']

    def init_weight(key, shape):
        return random.normal(key, shape) * 0.02

    params = {}
    key, *subkeys = random.split(key, n_layers * 13 + 4)

    params['token_embedding'] = init_weight(subkeys[0], (vocab_size, dim))
    params['norm'] = init_weight(subkeys[1], (dim,))
    params['output'] = init_weight(subkeys[2], (dim, vocab_size))

    params['layers'] = []

    for i in range(n_layers):
        layer_params = {}
        layer_params['attention_norm'] = init_weight(subkeys[i*13+3], (dim,))
        layer_params['attention_wq'] = init_weight(subkeys[i*13+4], (dim, n_heads * head_dim))
        layer_params['attention_wk'] = init_weight(subkeys[i*13+5], (dim, n_kv_heads * head_dim))
        layer_params['attention_wv'] = init_weight(subkeys[i*13+6], (dim, n_kv_heads * head_dim))
        layer_params['attention_wo'] = init_weight(subkeys[i*13+7], (n_heads * head_dim, dim))
        layer_params['ffn_norm'] = init_weight(subkeys[i*13+8], (dim,))
        layer_params['ffn_w1'] = init_weight(subkeys[i*13+9], (hidden_dim, dim))
        layer_params['ffn_w2'] = init_weight(subkeys[i*13+10], (dim, hidden_dim))
        layer_params['ffn_w3'] = init_weight(subkeys[i*13+11], (hidden_dim, dim))
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

            # Ensure we have enough layers in our params
            while len(params['layers']) <= layer_num:
                params['layers'].append({})

            layer_params = params['layers'][layer_num]

            if 'attention_norm.weight' in k:
                layer_params['attention_norm'] = jnp.array(torch_to_numpy(v))
                assert layer_params['attention_norm'].shape == (config['dim'],)
            elif 'ffn_norm.weight' in k:
                layer_params['ffn_norm'] = jnp.array(torch_to_numpy(v))
                assert layer_params['ffn_norm'].shape == (config['dim'],)
            elif 'attention.wq.weight' in k:
                layer_params['attention_wq'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['attention_wq'].shape == (config['dim'], config['n_heads'] * config['head_dim'])
            elif 'attention.wk.weight' in k:
                layer_params['attention_wk'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['attention_wk'].shape == (config['dim'], config['n_kv_heads'] * config['head_dim'])
            elif 'attention.wv.weight' in k:
                layer_params['attention_wv'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['attention_wv'].shape == (config['dim'], config['n_kv_heads'] * config['head_dim'])
            elif 'attention.wo.weight' in k:
                layer_params['attention_wo'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['attention_wo'].shape == (config['n_heads'] * config['head_dim'], config['dim'])
            elif 'feed_forward.w1.weight' in k:
                layer_params['ffn_w1'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['ffn_w1'].shape == (config['dim'], config['hidden_dim'])
            elif 'feed_forward.w2.weight' in k:
                layer_params['ffn_w2'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['ffn_w2'].shape == (config['hidden_dim'], config['dim'])
            elif 'feed_forward.w3.weight' in k:
                layer_params['ffn_w3'] = jnp.array(torch_to_numpy(v)).T
                assert layer_params['ffn_w3'].shape == (config['dim'], config['hidden_dim'])
        elif 'tok_embeddings.weight' in k:
            params['token_embedding'] = jnp.array(torch_to_numpy(v))
            assert params['token_embedding'].shape == (config['vocab_size'], config['dim'])
        elif 'norm.weight' in k:
            params['norm'] = jnp.array(torch_to_numpy(v))
            assert params['norm'].shape == (config['dim'],)
        elif 'output.weight' in k:
            params['output'] = jnp.array(torch_to_numpy(v)).T
            assert params['output'].shape == (config['dim'], config['vocab_size'])

    return params, config

def generate(params: Dict[str, Any], config: Dict[str, Any], tokenizer: Any, prompt: str, max_new_tokens: int = 20, temperature: float = 0.7) -> str:
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="jax").squeeze()

    # Initialize the model's state
    x = input_ids[None, :]  # Add batch dimension explicitly

    # Precompute freqs_cis for the entire possible length
    max_seq_len = config['sliding_window']
    freqs_cis = precompute_freqs_cis(
        config['head_dim'],
        max_seq_len,
        config['rope_theta']
    )

    # Generate tokens
    for _ in range(max_new_tokens):
        # Ensure x doesn't exceed the model's context length
        x = x[:, -config['sliding_window']:]

        # Forward pass through the model
        logits = transformer(params, x, freqs_cis[:x.shape[1]], config)

        # Get the logits for the last token
        next_token_logits = logits[0, -1]  # Remove batch dimension

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Sample the next token
        next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_logits)

        # Append the new token to the sequence
        x = jnp.concatenate([x, next_token[None, None]], axis=1)

        # If we've generated an EOS token, stop
        if next_token == tokenizer.eos_token_id:
            break

    # Decode the generated sequence
    generated_text = tokenizer.decode(x[0, len(input_ids):], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    # Set up paths and configurations
    model_path = "../sagasu/mistral-7B-v0.2/"

    # Load the model and configuration
    print("Loading model...")
    params, config = load_model(model_path)

    # Update or add necessary configuration parameters
    config['max_seq_len'] = 4096  # Update this to match the model's max sequence length
    config['rope_theta'] = 1000000.0  # Make sure this matches the value used during training

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    # Test text generation
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

    # Interactive mode
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == 'exit':
            break

        output = generate(params, config, tokenizer, user_prompt, max_new_tokens=100, temperature=0.7)
        print(f"\nGenerated text: {output}")

    print("Thank you for using the language model. Goodbye!")