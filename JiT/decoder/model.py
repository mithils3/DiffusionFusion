import math

import torch
from torch import nn

from JiT.model_jit import RMSNorm, SwiGLUFFN
from JiT.util.model_util import get_2d_sincos_pos_embed


class Decoder(nn.Module):
    """Transformer decoder that reconstructs an image from concatenated EVA and DINO tokens."""

    def __init__(
        self,
        patch_size: int,
        eva_hidden_size: int,
        dino_hidden_size: int,
        hidden_size: int,
        out_channels: int,
        depth: int,
        attn_drop=0.0,
        proj_drop=0.0,
        num_heads=8,
        mlp_ratio=4.0,
        output_image_size: int = 224,
    ) -> None:
        super().__init__()
        self.eva_hidden_size = eva_hidden_size
        self.dino_hidden_size = dino_hidden_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.output_image_size = output_image_size

        if output_image_size % patch_size != 0:
            raise ValueError(
                f"output_image_size={output_image_size} must be divisible by patch_size={patch_size}."
            )
        self.grid_size = output_image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.eva_embedder = nn.Identity() if eva_hidden_size == hidden_size else nn.Linear(
            eva_hidden_size, hidden_size, bias=True
        )
        self.dino_embedder = nn.Identity() if dino_hidden_size == hidden_size else nn.Linear(
            dino_hidden_size, hidden_size, bias=True
        )
        self.eva_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.dino_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        self.eva_stream_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dino_stream_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.final_layer = DecoderFinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
        )
        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=self.attn_drop if (
                    self.depth // 4 * 3 > i >= self.depth // 4) else 0.0,
                proj_drop=self.proj_drop if (self.depth // 4 * 3 > i >= self.depth // 4) else 0.0)
            for i in range(self.depth)
        ])
        self.init_weights()

    def unpatchify(self, x, p):
        """
        Reassemble a sequence of per-patch pixel predictions into an image tensor.

        Args:
            x: Tensor of shape ``[B, T, p * p * C]`` where each token stores one
                flattened ``p x p`` RGB patch.
            p: Spatial patch size used by the output head.

        Returns:
            Tensor of shape ``[B, C, H, W]``.
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def tokens_to_image(self, tokens):
        """Map decoder tokens ``[B, T, D]`` to an RGB image ``[B, C, H, W]``."""
        patch_pixels = self.final_layer(tokens)
        return self.unpatchify(patch_pixels, self.patch_size)

    def init_weights(self):
        """Initialize decoder embeddings, projections, and output head."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.eva_pos_embed.data.copy_(pos_embed_tensor)
        self.dino_pos_embed.data.copy_(pos_embed_tensor)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        nn.init.normal_(self.eva_stream_embed, std=0.02)
        nn.init.normal_(self.dino_stream_embed, std=0.02)
        if isinstance(self.eva_embedder, nn.Linear):
            we = self.eva_embedder.weight.data
            nn.init.xavier_uniform_(we.view(we.shape[0], -1))
            nn.init.constant_(self.eva_embedder.bias, 0)
        if isinstance(self.dino_embedder, nn.Linear):
            wd = self.dino_embedder.weight.data
            nn.init.xavier_uniform_(wd.view(wd.shape[0], -1))
            nn.init.constant_(self.dino_embedder.bias, 0)

        # Scale residual-stream projections for stability across depth.
        residual_scale = 1 / math.sqrt(2 * max(self.depth, 1))
        for block in self.blocks:
            block.attn.proj.weight.data.mul_(residual_scale)
            block.attn.proj_cross.weight.data.mul_(residual_scale)
            block.mlp.w3.weight.data.mul_(residual_scale)

        # Keep the pixel head small so reconstruction starts near zero/mean.
        final_std = 0.02 / math.sqrt(2 * max(self.depth, 1))
        nn.init.normal_(self.final_layer.linear.weight, std=final_std)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _prepare_feature_tokens(
        self,
        features: torch.Tensor,
        *,
        expected_hidden_size: int,
        stream_name: str,
    ) -> torch.Tensor:
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim != 3:
            raise ValueError(
                f"{stream_name} features must be rank-3 or rank-4, got shape {tuple(features.shape)}."
            )

        if features.shape[-1] != expected_hidden_size:
            raise ValueError(
                f"Expected {stream_name} channel width {expected_hidden_size}, got {features.shape[-1]}."
            )
        if features.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} {stream_name} tokens, got {features.shape[1]}."
            )
        return features

    def _build_context_tokens(
        self,
        eva_tokens: torch.Tensor,
        dino_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if eva_tokens.shape[0] != dino_tokens.shape[0]:
            raise ValueError(
                "EVA and DINO batch sizes must match before context concatenation, got "
                f"{eva_tokens.shape[0]} and {dino_tokens.shape[0]}."
            )

        # With a 16x16 grid this yields 256 EVA tokens + 256 DINO tokens = 512 context tokens.
        eva_tokens = eva_tokens + self.eva_pos_embed + self.eva_stream_embed
        dino_tokens = dino_tokens + self.dino_pos_embed + self.dino_stream_embed
        return torch.cat([eva_tokens, dino_tokens], dim=1)

    def forward(self, eva, dino):
        """
        Decode a pair of aligned context streams into an output image.

        Args:
            eva: EVA features of shape ``[B, D, H, W]`` or EVA tokens ``[B, T, D]``.
            dino: DINO features of shape ``[B, D, H, W]`` or DINO tokens ``[B, T, D]``.

        Returns:
            Reconstructed image tensor of shape ``[B, C, H, W]``.
        """
        eva_tokens = self._prepare_feature_tokens(
            eva,
            expected_hidden_size=self.eva_hidden_size,
            stream_name="EVA",
        )
        dino_tokens = self._prepare_feature_tokens(
            dino,
            expected_hidden_size=self.dino_hidden_size,
            stream_name="DINO",
        )
        eva_tokens = self.eva_embedder(eva_tokens)
        dino_tokens = self.dino_embedder(dino_tokens)

        # Use one shared set of learnable query slots and broadcast it across the batch.
        x = self.query_tokens.expand(eva_tokens.shape[0], -1, -1)
        x = x + self.query_pos_embed
        ctx_tokens = self._build_context_tokens(eva_tokens, dino_tokens)
        for block in self.blocks:
            x = block(x, ctx_tokens)
        return self.tokens_to_image(x)

    def generate(self, eva, dino):
        """Inference entrypoint expected by decoder evaluation."""
        return self.forward(eva, dino)

class CrossAttention(nn.Module):
    """Self-attention over decoder queries followed by cross-attention into context."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q_cross_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_cross_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.cross_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_cross = nn.Linear(dim, dim)
        self.proj_cross_drop = nn.Dropout(proj_drop)

    def forward(self, x_in, ctx):
        """
        Update decoder queries using both self-attention and context attention.

        Args:
            x_in: Decoder query tokens of shape ``[B, N, D]``.
            ctx: Context tokens of shape ``[B, M, D]``.

        Returns:
            Updated decoder tokens of shape ``[B, N, D]``.
        """
        B, N, C = x_in.shape
        B_ctx, N_ctx, C_ctx = ctx.shape
        head_dim = C // self.num_heads

        if B != B_ctx:
            raise ValueError(
                f"Batch size mismatch between decoder tokens ({B}) and context ({B_ctx})."
            )
        if C != C_ctx:
            raise ValueError(
                f"Hidden size mismatch between decoder tokens ({C}) and context ({C_ctx})."
            )

        # self-attn
        q, k, v = self.qkv(x_in).chunk(3, dim=-1)
        q = self.q_norm(q.reshape(B, N, self.num_heads,
                                  head_dim).transpose(1, 2))
        k = self.k_norm(k.reshape(B, N, self.num_heads,
                                  head_dim).transpose(1, 2))
        v = v.reshape(B, N, self.num_heads, head_dim).transpose(1, 2)

        self_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.
        )
        self_out = self_out.transpose(1, 2).reshape(B, N, C)
        x = x_in + self.proj_drop(self.proj(self_out))

        # cross-attn
        q_cross = self.cross_q(x)
        k_cross, v_cross = self.cross_kv(ctx).chunk(2, dim=-1)

        q_cross = self.q_cross_norm(q_cross.reshape(
            B, N, self.num_heads, head_dim).transpose(1, 2))
        k_cross = self.k_cross_norm(k_cross.reshape(
            B_ctx, N_ctx, self.num_heads, head_dim).transpose(1, 2))
        v_cross = v_cross.reshape(
            B_ctx, N_ctx, self.num_heads, head_dim).transpose(1, 2)

        cross_out = torch.nn.functional.scaled_dot_product_attention(
            q_cross, k_cross, v_cross, dropout_p=self.attn_drop.p if self.training else 0.
        )
        cross_out = cross_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj_cross_drop(self.proj_cross(cross_out))

        return x


class DecoderBlock(nn.Module):
    """Single decoder block: attention-based token mixing followed by a SwiGLU FFN."""

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
        self.attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, ctx):
        """Refine query tokens with attention and then a per-token feed-forward update."""
        # Attention already applies its own residual updates for self- and cross-attention.
        x = self.attn(self.norm1(x), ctx)
        x = x + self.mlp(self.norm2(x))

        return x


class DecoderFinalLayer(nn.Module):
    """Project decoder tokens back to flattened image patches."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )

    def forward(self, x):
        """Convert tokens ``[B, T, D]`` into patch pixels ``[B, T, p * p * C]``."""
        x = self.norm_final(x)
        x = self.linear(x)
        return x
