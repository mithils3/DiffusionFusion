from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.spectral_norm import SpectralNorm


def _extract_checkpoint_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected discriminator checkpoint payload to be a state dict mapping, got {type(payload).__name__}."
        )
    return payload


def _is_classifier_key(key: str) -> bool:
    return key.startswith(("head.", "fc.", "classifier."))


def _normalize_backbone_name(model_name: str) -> str:
    return model_name[len("timm/"):] if model_name.startswith("timm/") else model_name


class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.ratio = 1.0 / np.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    """Local batch norm used by the released RAE discriminator implementation."""

    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = x.float()
        groups = int(np.ceil(x.size(0) / self.virtual_bs))
        x = x.view(groups, -1, x.size(-2), x.size(-1))
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


RECIPES: dict[str, dict[str, object]] = {
    "S_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "embed_dim": 384,
    },
    "S_8": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 8,
        "embed_dim": 384,
    },
    "B_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "embed_dim": 768,
    },
}


def make_block(
    channels: int,
    *,
    kernel_size: int,
    norm_type: str,
    norm_eps: float,
    using_spec_norm: bool,
) -> nn.Module:
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "gn":
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError(f"Unknown norm_type '{norm_type}'")

    conv = SpectralConv1d if using_spec_norm else nn.Conv1d
    return nn.Sequential(
        conv(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class TimmDinoBackbone(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool,
        checkpoint_path: str | None,
        input_size: int,
        key_depths: Iterable[int],
        freeze_backbone: bool,
        feature_dim: int | None,
        recipe: str,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required to build the decoder discriminator backbone.") from exc

        if recipe not in RECIPES:
            raise KeyError(f"Unknown discriminator recipe `{recipe}`. Available: {sorted(RECIPES)}")

        normalized_model_name = _normalize_backbone_name(model_name)
        use_pretrained = bool(pretrained and checkpoint_path is None)
        self.backbone = timm.create_model(normalized_model_name, pretrained=use_pretrained, num_classes=0)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        recipe_cfg = RECIPES[recipe]
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_size = int(patch_size[0])
        if patch_size is None:
            raise AttributeError(f"Backbone `{normalized_model_name}` is missing patch_embed.patch_size.")
        expected_patch_size = int(recipe_cfg["patch_size"])
        if patch_size != expected_patch_size:
            raise ValueError(
                f"Backbone `{normalized_model_name}` has patch size {patch_size}, expected {expected_patch_size} for recipe {recipe}."
            )

        embed_dim = int(getattr(self.backbone, "num_features"))
        expected_embed_dim = int(recipe_cfg["embed_dim"])
        if embed_dim != expected_embed_dim:
            raise ValueError(
                f"Backbone `{normalized_model_name}` has feature dim {embed_dim}, expected {expected_embed_dim} for recipe {recipe}."
            )
        if feature_dim is not None and embed_dim != int(feature_dim):
            raise ValueError(f"Configured discriminator feature_dim={feature_dim} does not match backbone dim {embed_dim}.")

        self.embed_dim = embed_dim
        self.input_size = int(input_size)
        max_depth = int(recipe_cfg["depth"])
        self.key_depths = tuple(sorted({int(depth) for depth in key_depths if 0 <= int(depth) < max_depth}))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0) or 0)
        self.freeze_backbone = bool(freeze_backbone)

        data_config = timm.data.resolve_model_data_config(self.backbone)
        mean = torch.tensor(data_config.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(data_config.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        # The decoder GAN feeds images in [-1, 1], while timm backbones expect normalized [0, 1] inputs.
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1), persistent=False)

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Decoder discriminator checkpoint not found: {checkpoint_path}")

        state_dict = _extract_checkpoint_state_dict(torch.load(path, map_location="cpu"))
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        missing_keys = [key for key in incompatible.missing_keys if not _is_classifier_key(key)]
        unexpected_keys = [key for key in incompatible.unexpected_keys if not _is_classifier_key(key)]
        if missing_keys or unexpected_keys:
            problems = []
            if missing_keys:
                problems.append(f"missing keys: {missing_keys}")
            if unexpected_keys:
                problems.append(f"unexpected keys: {unexpected_keys}")
            raise RuntimeError(
                f"Decoder discriminator checkpoint {checkpoint_path} is incompatible with "
                f"{self.backbone.__class__.__name__}; {'; '.join(problems)}."
            )

    def train(self, mode: bool = True) -> "TimmDinoBackbone":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> list[Tensor]:
        images = images.float()
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        images = images * self.x_scale + self.x_shift

        # Keep the backbone parameters frozen via requires_grad_(False), but do
        # not wrap the forward in no_grad(): the generator adversarial step
        # still needs gradients to flow from discriminator logits back to the
        # reconstructed image.
        outputs = self.backbone.forward_intermediates(
            images,
            indices=self.key_depths,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )

        final_tokens, intermediates = outputs
        if final_tokens.ndim != 3:
            raise ValueError(f"Expected final backbone tokens to be 3D, got shape {tuple(final_tokens.shape)}.")
        if self.num_prefix_tokens > 0:
            final_tokens = final_tokens[:, self.num_prefix_tokens :, :]
        activations = [final_tokens.transpose(1, 2).contiguous()]
        for feature_map in intermediates:
            activations.append(feature_map.flatten(2))
        return activations


class DinoPatchDiscriminator(nn.Module):
    """RAE-style discriminator with a frozen timm DINO backbone and per-depth patch heads."""

    def __init__(
        self,
        *,
        backbone_model_name: str = "timm/vit_small_patch8_224.dino",
        checkpoint_path: str | None = None,
        input_size: int = 224,
        feature_dim: int = 384,
        kernel_size: int = 9,
        norm_type: str = "bn",
        using_spec_norm: bool = True,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        recipe: str = "S_8",
        key_depths: Iterable[int] = (2, 5, 8, 11),
        norm_eps: float = 1e-6,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        self.dino = TimmDinoBackbone(
            model_name=backbone_model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            input_size=input_size,
            key_depths=key_depths,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
            recipe=recipe,
        )
        dino_channels = self.dino.embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(
                        dino_channels,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    ResidualBlock(
                        make_block(
                            dino_channels,
                            kernel_size=kernel_size,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
                        )
                    ),
                    (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_channels, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(self.dino.key_depths) + 1)
            ]
        )

    def classify(self, images: Tensor) -> Tensor:
        activations = self.dino(images)
        batch_size = images.shape[0]
        outputs = []
        for head, activation in zip(self.heads, activations, strict=True):
            outputs.append(head(activation).view(batch_size, -1))
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        fake_images: Tensor,
        real_images: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        fake_logits = self.classify(fake_images)
        if real_images is None:
            return fake_logits
        real_logits = self.classify(real_images)
        return fake_logits, real_logits
