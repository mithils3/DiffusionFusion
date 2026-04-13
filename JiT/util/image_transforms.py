from collections.abc import Sequence

from torchvision import transforms

from .crop import center_crop_arr


def _as_float_tuple(values: Sequence[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)


def build_center_crop_normalize_transform(
    image_size: int,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Build a deterministic ADM-style center-crop transform.

    We keep the geometric preprocessing identical across VAE features, DINO
    features, decoder targets, and FID reference image export. Optional
    normalization is applied after cropping and tensor conversion.
    """

    mean = _as_float_tuple(mean)
    std = _as_float_tuple(std)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must either both be set or both be omitted.")

    transform_steps: list[object] = [
        transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)
