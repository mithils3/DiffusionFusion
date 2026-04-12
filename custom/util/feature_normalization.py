import torch
import torch.nn.functional as F


def normalize_feature_map_tokens(
    feature_map: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Apply per-token normalization across channel dimensions.

    Args:
        feature_map: Tensor shaped ``[B, C, H, W]`` containing spatial patch features.
        eps: Numerical stability epsilon for the layer norm.

    Returns:
        Tensor with the same shape and dtype as ``feature_map`` where every
        spatial token has been independently normalized across channels.
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected DINO feature map of shape [B, C, H, W], got {tuple(feature_map.shape)}."
        )

    batch_size, channels, height, width = feature_map.shape
    tokens = feature_map.float().flatten(2).transpose(1, 2)
    tokens = F.layer_norm(tokens, (channels,), eps=eps)
    normalized = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
    return normalized.to(dtype=feature_map.dtype)


def normalize_dino_feature_map_tokens(
    feature_map: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Backward-compatible alias for DINO feature normalization."""
    return normalize_feature_map_tokens(feature_map, eps=eps)
