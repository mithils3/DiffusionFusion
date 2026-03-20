import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

_DEFAULT_BACKBONE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_BACKBONE_STD = (0.229, 0.224, 0.225)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "ln":
        return ChannelLayerNorm2d(channels)
    raise ValueError(f"Unsupported discriminator norm_type: {norm_type}")


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


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


def _load_checkpoint(module: nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Decoder discriminator checkpoint not found: {checkpoint_path}"
        )

    state_dict = _extract_checkpoint_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key for key in incompatible.missing_keys if not _is_classifier_key(key)
    ]
    unexpected_keys = [
        key for key in incompatible.unexpected_keys if not _is_classifier_key(key)
    ]
    if missing_keys or unexpected_keys:
        problems = []
        if missing_keys:
            problems.append(f"missing keys: {missing_keys}")
        if unexpected_keys:
            problems.append(f"unexpected keys: {unexpected_keys}")
        raise RuntimeError(
            f"Decoder discriminator checkpoint {checkpoint_path} is incompatible with "
            f"{module.__class__.__name__}; {'; '.join(problems)}."
        )


def _tokens_to_feature_map(
    features: Any,
    *,
    num_prefix_tokens: int = 0,
    patch_grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if isinstance(features, dict):
        for key in (
            "x_norm_patchtokens",
            "patch_tokens",
            "last_hidden_state",
            "tokens",
            "feature_map",
        ):
            value = features.get(key)
            if value is not None:
                features = value
                break

    if isinstance(features, (list, tuple)):
        features = features[-1]

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"Unsupported discriminator backbone output type: {type(features).__name__}"
        )

    if features.ndim == 4:
        # Swin-style backbones often return NHWC feature maps; conv heads expect NCHW.
        if features.shape[-1] > max(features.shape[1], features.shape[2]):
            return features.permute(0, 3, 1, 2).contiguous()
        return features

    if features.ndim != 3:
        raise ValueError(
            f"Expected 3D or 4D discriminator backbone output, got shape {tuple(features.shape)}."
        )

    batch_size, token_count, channels = features.shape
    grid_tokens = features

    if patch_grid_size is not None:
        grid_h, grid_w = int(patch_grid_size[0]), int(patch_grid_size[1])
        expected_token_count = grid_h * grid_w
        if token_count == expected_token_count:
            return grid_tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)
        if num_prefix_tokens > 0 and token_count - num_prefix_tokens == expected_token_count:
            grid_tokens = features[:, num_prefix_tokens:, :]
            return grid_tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)

    grid_size = int(math.sqrt(token_count))
    if grid_size * grid_size != token_count:
        for prefix_count in range(max(1, num_prefix_tokens), min(token_count, 16) + 1):
            remaining = token_count - prefix_count
            grid_size = int(math.sqrt(remaining))
            if grid_size * grid_size == remaining:
                grid_tokens = features[:, prefix_count:, :]
                token_count = remaining
                break

    if grid_size * grid_size != token_count:
        raise ValueError(
            f"Could not reshape backbone tokens with shape {tuple(features.shape)} into a square grid."
        )

    return grid_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)


class DinoPatchDiscriminator(nn.Module):
    """Patch-based DINO discriminator that returns a spatial real/fake logit map."""

    def __init__(
        self,
        backbone_model_name: str = "timm/vit_small_patch16_dinov3.lvd1689m",
        checkpoint_path: str | None = None,
        input_size: int = 224,
        feature_dim: int = 384,
        kernel_size: int = 9,
        norm_type: str = "gn",
        using_spec_norm: bool = True,
        freeze_backbone: bool = False,
        backbone: nn.Module | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        if backbone is None:
            self.backbone, backbone_mean, backbone_std = self._build_backbone(
                backbone_model_name=backbone_model_name,
                checkpoint_path=checkpoint_path,
                pretrained=pretrained,
            )
        else:
            self.backbone = backbone
            backbone_mean = _DEFAULT_BACKBONE_MEAN
            backbone_std = _DEFAULT_BACKBONE_STD
        self.register_buffer(
            "backbone_mean",
            torch.tensor(backbone_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "backbone_std",
            torch.tensor(backbone_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.head = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                using_spec_norm,
            ),
            _make_norm(norm_type, feature_dim),
            nn.GELU(),
            _maybe_spectral_norm(
                nn.Conv2d(feature_dim, 1, kernel_size=1, bias=True),
                using_spec_norm,
            ),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _build_backbone(
        backbone_model_name: str,
        checkpoint_path: str | None,
        pretrained: bool,
    ) -> tuple[nn.Module, tuple[float, float, float], tuple[float, float, float]]:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required to build the decoder discriminator backbone."
            ) from exc

        use_checkpoint = bool(checkpoint_path)

        backbone = timm.create_model(
            backbone_model_name,
            pretrained=pretrained and not use_checkpoint,
            num_classes=0,
        )
        data_config = timm.data.resolve_model_data_config(backbone)
        if use_checkpoint:
            _load_checkpoint(backbone, checkpoint_path)
        mean = tuple(data_config.get("mean", _DEFAULT_BACKBONE_MEAN))
        std = tuple(data_config.get("std", _DEFAULT_BACKBONE_STD))
        return backbone, mean, std

    def _normalize_backbone_input(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        # Decoder training works in normalized image space; remap to the DINO backbone's input space.
        images = images.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
        return (images - self.backbone_mean) / self.backbone_std

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_backbone_input(images)
        if normalized.shape[-2:] != (self.input_size, self.input_size):
            normalized = F.interpolate(
                normalized,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone.forward_features(normalized)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0) or 0)
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_grid_size = getattr(patch_embed, "grid_size", None)
        feature_map = _tokens_to_feature_map(
            features,
            num_prefix_tokens=prefix_tokens,
            patch_grid_size=patch_grid_size,
        )
        if feature_map.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected discriminator feature dim {self.feature_dim}, got {feature_map.shape[1]}."
            )
        return feature_map

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_features(images)
        return self.head(feature_map)

    def forward(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fake_logits = self.logits(fake_images)
        if real_images is None:
            return fake_logits
        real_logits = self.logits(real_images)
        return fake_logits, real_logits
