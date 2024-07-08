import torch
import jax
import jax.numpy as jnp
import numpy as np

# PyTorch implementation
def pt_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def pt_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# JAX implementation
def jax_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)
    freqs_LD = jnp.outer(t, freqs).astype(jnp.float32)
    cos_LD, sin_LD = jnp.cos(freqs_LD), jnp.sin(freqs_LD)
    return jnp.stack([cos_LD, sin_LD], axis=-1)

def jax_apply_rotary_emb(
    xq_BLHK: jnp.ndarray,
    xk_BLHK: jnp.ndarray,
    freqs_cis_LK2: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def rotate(x_BLHK):
        x1_BLHK, x2_BLHK = jnp.split(x_BLHK, 2, axis=-1)
        freqs_cos_LK1, freqs_sin_LK2 = jnp.split(freqs_cis_LK2, 2, axis=-1)
        freqs_cos_LK = freqs_cos_LK1.squeeze(axis=2)
        freqs_sin_LK = freqs_sin_LK2.squeeze(axis=2)

        freqs_cos_LK = freqs_cos_LK[:x_BLHK.shape[1]]
        freqs_sin_LK = freqs_sin_LK[:x_BLHK.shape[1]]

        freqs_cos_1L1K = jnp.expand_dims(freqs_cos_LK, axis=(0, 2))
        freqs_sin_1L1K = jnp.expand_dims(freqs_sin_LK, axis=(0, 2))

        return jnp.concatenate([
            x1_BLHK * freqs_cos_1L1K - x2_BLHK * freqs_sin_1L1K,
            x1_BLHK * freqs_sin_1L1K + x2_BLHK * freqs_cos_1L1K
        ], axis=-1)

    return rotate(xq_BLHK), rotate(xk_BLHK)

# Test functions
def test_precompute_freqs_cis():
    dim, end = 128, 64
    pt_freqs = pt_precompute_freqs_cis(dim, end)
    jax_freqs = jax_precompute_freqs_cis(dim, end)

    pt_cos, pt_sin = pt_freqs.real, pt_freqs.imag
    jax_cos, jax_sin = jax_freqs[..., 0], jax_freqs[..., 1]

    assert np.allclose(pt_cos.numpy(), jax_cos, atol=1e-5), "Cosine values do not match"
    assert np.allclose(pt_sin.numpy(), jax_sin, atol=1e-5), "Sine values do not match"
    print("precompute_freqs_cis test passed")

def test_apply_rotary_emb():
    B, L, H, K = 2, 32, 4, 128
    dim, end = K, L

    # Generate random input tensors
    pt_xq = torch.randn(B, L, H, K)
    pt_xk = torch.randn(B, L, H, K)
    jax_xq = jnp.array(pt_xq.numpy())
    jax_xk = jnp.array(pt_xk.numpy())

    # Precompute frequencies
    pt_freqs = pt_precompute_freqs_cis(dim, end)
    jax_freqs = jax_precompute_freqs_cis(dim, end)

    # Apply rotary embeddings
    pt_xq_out, pt_xk_out = pt_apply_rotary_emb(pt_xq, pt_xk, pt_freqs)
    jax_xq_out, jax_xk_out = jax_apply_rotary_emb(jax_xq, jax_xk, jax_freqs)

    assert np.allclose(pt_xq_out.numpy(), jax_xq_out, atol=1e-5), "xq outputs do not match"
    assert np.allclose(pt_xk_out.numpy(), jax_xk_out, atol=1e-5), "xk outputs do not match"
    print("apply_rotary_emb test passed")

if __name__ == "__main__":
    test_precompute_freqs_cis()
    test_apply_rotary_emb()
    print("All tests passed!")
