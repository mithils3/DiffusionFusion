from __future__ import annotations

from math import pi

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """Project a spatial tensor to patch tokens with a lightweight bottleneck."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 768,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        img_size_tuple = (img_size, img_size)
        patch_size_tuple = (patch_size, patch_size)
        self.img_size = img_size_tuple
        self.patch_size = patch_size_tuple
        self.num_patches = (
            (img_size_tuple[0] // patch_size_tuple[0])
            * (img_size_tuple[1] // patch_size_tuple[1])
        )

        self.proj1 = nn.Conv2d(
            in_chans,
            pca_dim,
            kernel_size=patch_size_tuple,
            stride=patch_size_tuple,
            bias=False,
        )
        self.proj2 = nn.Conv2d(
            pca_dim,
            embed_dim,
            kernel_size=1,
            stride=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                "Input image size "
                f"({height}x{width}) does not match model ({self.img_size[0]}x{self.img_size[1]})."
            )
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * inner_dim, bias=bias)
        self.w3 = nn.Linear(inner_dim, out_dim or dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


def broadcat(tensors: tuple[torch.Tensor, ...], dim: int = -1) -> torch.Tensor:
    num_tensors = len(tensors)
    shape_len = len(tensors[0].shape)
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(t.shape) for t in tensors]))
    expandable_dims = [(i, values) for i, values in enumerate(dims) if i != dim]
    if not all(len(set(values)) <= 2 for _, values in expandable_dims):
        raise ValueError("Invalid dimensions for broadcastable concatenation.")
    max_dims = [(axis, max(values)) for axis, values in expandable_dims]
    expanded_dims = [(axis, (value,) * num_tensors) for axis, value in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[values for _, values in expanded_dims]))
    expanded_tensors = [
        tensor.expand(*shape) for tensor, shape in zip(tensors, expandable_shapes)
    ]
    return torch.cat(expanded_tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


class VisionRotaryEmbeddingFast(nn.Module):
    """2D rotary embeddings flattened into image-token order."""

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: int | None = None,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
        num_freqs: int = 1,
    ) -> None:
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown frequency mode `{freqs_for}`.")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        self.register_buffer(
            "freqs_cos",
            freqs.cos().reshape(-1, freqs.shape[-1]),
            persistent=False,
        )
        self.register_buffer(
            "freqs_sin",
            freqs.sin().reshape(-1, freqs.shape[-1]),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs_cos = self.freqs_cos.to(device=x.device, dtype=x.dtype)
        freqs_sin = self.freqs_sin.to(device=x.device, dtype=x.dtype)
        return x * freqs_cos + rotate_half(x) * freqs_sin


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


__all__ = [
    "BottleneckPatchEmbed",
    "RMSNorm",
    "SwiGLUFFN",
    "VisionRotaryEmbeddingFast",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "modulate",
]
