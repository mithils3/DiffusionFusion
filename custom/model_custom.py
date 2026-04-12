from __future__ import annotations

import math

import torch
from torch import nn

from .layers import (
    BottleneckPatchEmbed,
    RMSNorm,
    SwiGLUFFN,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    modulate,
)


SUPPORTED_CUSTOM_MODELS = (
    "CustomDiT-B/2-4C",
    "CustomDiT-B/4-4C",
    "CustomDiT-L/2-4C",
    "CustomDiT-L/4-4C",
)


def _format_supported_model_ids() -> str:
    return ", ".join(SUPPORTED_CUSTOM_MODELS)


def validate_model_name(model_name: str) -> None:
    if model_name in SUPPORTED_CUSTOM_MODELS:
        return
    if model_name.startswith("JiT-"):
        raise ValueError(
            f"Model `{model_name}` uses the removed JiT backbone. "
            f"This migration is a clean break; use one of {_format_supported_model_ids()} "
            "and retrain or evaluate a CustomDiT checkpoint."
        )
    raise ValueError(
        f"Unsupported model `{model_name}`. Supported models: {_format_supported_model_ids()}."
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def _maybe_drop(self, labels: torch.Tensor, train: bool) -> torch.Tensor:
        if not train or self.dropout_prob <= 0:
            return labels
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        drop_ids = drop_ids & (labels != self.num_classes)
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: torch.Tensor, train: bool) -> torch.Tensor:
        labels = self._maybe_drop(labels, train=train)
        return self.embedding_table(labels)


class CustomAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"Attention dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _apply_stream_rope(
        x: torch.Tensor,
        ropes: tuple[VisionRotaryEmbeddingFast, ...] | None,
        stream_lengths: tuple[int, ...],
    ) -> torch.Tensor:
        if ropes is None:
            return x
        if len(ropes) != len(stream_lengths):
            raise ValueError("Each token stream must have a matching rotary embedding.")
        pieces: list[torch.Tensor] = []
        offset = 0
        for rope, length in zip(ropes, stream_lengths, strict=True):
            next_offset = offset + length
            pieces.append(rope(x[:, :, offset:next_offset]))
            offset = next_offset
        if offset != x.shape[2]:
            raise ValueError(
                f"Stream lengths sum to {offset}, but attention sequence length is {x.shape[2]}."
            )
        return torch.cat(pieces, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        ropes: tuple[VisionRotaryEmbeddingFast, ...] | None,
        stream_lengths: tuple[int, ...],
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._apply_stream_rope(q, ropes, stream_lengths)
        k = self._apply_stream_rope(k, ropes, stream_lengths)
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return self.proj_drop(self.proj(x))


class CustomLightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = CustomAttention(
            hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ropes: tuple[VisionRotaryEmbeddingFast, ...],
        stream_lengths: tuple[int, ...],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            ropes=ropes,
            stream_lengths=stream_lengths,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class LatentFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DinoFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, dino_hidden_size: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, dino_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class CustomLightningDiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        dino_hidden_size: int = 768,
        dino_patches: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"CustomLightningDiT hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        head_dim = hidden_size // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"CustomLightningDiT attention head dimension ({head_dim}) must be even for rotary embeddings."
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.input_size = input_size
        self.dino_hidden_size = dino_hidden_size
        self.dino_patches = dino_patches

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_prob=class_dropout_prob,
        )

        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )
        self.latent_num_patches = self.x_embedder.num_patches
        self.dino_num_patches = dino_patches * dino_patches
        self.dino_embedder = (
            nn.Identity()
            if dino_hidden_size == hidden_size
            else nn.Linear(dino_hidden_size, hidden_size, bias=True)
        )

        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, self.latent_num_patches, hidden_size),
            requires_grad=False,
        )
        self.dino_pos_embed = nn.Parameter(
            torch.zeros(1, self.dino_num_patches, hidden_size),
            requires_grad=False,
        )
        self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dino_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

        rope_dim = head_dim // 2
        self.latent_rope = VisionRotaryEmbeddingFast(
            dim=rope_dim,
            pt_seq_len=input_size // patch_size,
        )
        self.dino_rope = VisionRotaryEmbeddingFast(
            dim=rope_dim,
            pt_seq_len=dino_patches,
        )

        self.blocks = nn.ModuleList(
            [
                CustomLightningDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        self.latent_final_layer = LatentFinalLayer(hidden_size, patch_size, self.out_channels)
        self.dino_final_layer = DinoFinalLayer(hidden_size, dino_hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        latent_grid = int(self.latent_num_patches**0.5)
        dino_grid = int(self.dino_num_patches**0.5)
        latent_pos_embed = get_2d_sincos_pos_embed(self.hidden_size, latent_grid)
        dino_pos_embed = get_2d_sincos_pos_embed(self.hidden_size, dino_grid)
        self.latent_pos_embed.data.copy_(torch.from_numpy(latent_pos_embed).float().unsqueeze(0))
        self.dino_pos_embed.data.copy_(torch.from_numpy(dino_pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        if self.x_embedder.proj2.bias is not None:
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        if isinstance(self.dino_embedder, nn.Linear):
            wd = self.dino_embedder.weight.data
            nn.init.xavier_uniform_(wd.view(wd.shape[0], -1))
            nn.init.constant_(self.dino_embedder.bias, 0)

        nn.init.normal_(self.latent_type_embed, std=0.02)
        nn.init.normal_(self.dino_type_embed, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.latent_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.latent_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.latent_final_layer.linear.weight, 0)
        nn.init.constant_(self.latent_final_layer.linear.bias, 0)

        nn.init.constant_(self.dino_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.dino_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.dino_final_layer.linear.weight, 0)
        nn.init.constant_(self.dino_final_layer.linear.bias, 0)

    def _prepare_dino_tokens(self, dino_features: torch.Tensor) -> torch.Tensor:
        if dino_features.ndim == 4:
            dino_tokens = dino_features.flatten(2).transpose(1, 2)
        elif dino_features.ndim == 3:
            dino_tokens = dino_features
        else:
            raise ValueError(
                f"DINO features must be rank-3 or rank-4, got shape {tuple(dino_features.shape)}."
            )

        if dino_tokens.shape[1] != self.dino_num_patches:
            raise ValueError(
                f"Expected {self.dino_num_patches} DINO tokens, got {dino_tokens.shape[1]}."
            )
        if dino_tokens.shape[-1] != self.dino_hidden_size:
            raise ValueError(
                f"Expected DINO hidden size {self.dino_hidden_size}, got {dino_tokens.shape[-1]}."
            )

        return self.dino_embedder(dino_tokens)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Expected a square latent token grid, got {x.shape[1]} tokens.")
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(
        self,
        latent: torch.Tensor,
        dino_features: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb

        latent_tokens = self.x_embedder(latent)
        dino_tokens = self._prepare_dino_tokens(dino_features)

        latent_tokens = latent_tokens + self.latent_pos_embed + self.latent_type_embed
        dino_tokens = dino_tokens + self.dino_pos_embed + self.dino_type_embed

        stream_lengths = (latent_tokens.shape[1], dino_tokens.shape[1])
        x = torch.cat([latent_tokens, dino_tokens], dim=1)
        ropes = (self.latent_rope, self.dino_rope)

        for block in self.blocks:
            x = block(x, c, ropes=ropes, stream_lengths=stream_lengths)

        latent_tokens, dino_tokens = x.split(stream_lengths, dim=1)
        latent_out = self.unpatchify(self.latent_final_layer(latent_tokens, c))
        dino_out = self.dino_final_layer(dino_tokens, c).transpose(1, 2).reshape(
            -1,
            self.dino_hidden_size,
            self.dino_patches,
            self.dino_patches,
        )
        return latent_out, dino_out


def CustomDiT_B_2_4C(**kwargs) -> CustomLightningDiT:
    kwargs.setdefault("input_size", 32)
    kwargs.setdefault("in_channels", 4)
    kwargs.setdefault("patch_size", 2)
    return CustomLightningDiT(
        depth=12,
        hidden_size=1024,
        num_heads=16,
        **kwargs,
    )


def CustomDiT_B_4_4C(**kwargs) -> CustomLightningDiT:
    kwargs.setdefault("input_size", 32)
    kwargs.setdefault("in_channels", 4)
    kwargs.setdefault("patch_size", 4)
    return CustomLightningDiT(
        depth=12,
        hidden_size=1024,
        num_heads=16,
        **kwargs,
    )


def CustomDiT_L_2_4C(**kwargs) -> CustomLightningDiT:
    kwargs.setdefault("input_size", 32)
    kwargs.setdefault("in_channels", 4)
    kwargs.setdefault("patch_size", 2)
    return CustomLightningDiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        **kwargs,
    )


def CustomDiT_L_4_4C(**kwargs) -> CustomLightningDiT:
    kwargs.setdefault("input_size", 32)
    kwargs.setdefault("in_channels", 4)
    kwargs.setdefault("patch_size", 4)
    return CustomLightningDiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        **kwargs,
    )


CustomDiT_models = {
    "CustomDiT-B/2-4C": CustomDiT_B_2_4C,
    "CustomDiT-B/4-4C": CustomDiT_B_4_4C,
    "CustomDiT-L/2-4C": CustomDiT_L_2_4C,
    "CustomDiT-L/4-4C": CustomDiT_L_4_4C,
}


def build_custom_dit(model_name: str, **kwargs) -> CustomLightningDiT:
    validate_model_name(model_name)
    return CustomDiT_models[model_name](**kwargs)


__all__ = [
    "CustomDiT_B_2_4C",
    "CustomDiT_B_4_4C",
    "CustomDiT_L_2_4C",
    "CustomDiT_L_4_4C",
    "CustomDiT_models",
    "CustomLightningDiT",
    "SUPPORTED_CUSTOM_MODELS",
    "build_custom_dit",
    "validate_model_name",
]
