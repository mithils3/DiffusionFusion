import torch
from torch import nn


def _apply_cutout(images: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0.0:
        return images

    batch_size, _channels, height, width = images.shape
    cutout_height = max(1, int(height * fraction))
    cutout_width = max(1, int(width * fraction))
    for index in range(batch_size):
        top = torch.randint(
            0,
            max(height - cutout_height + 1, 1),
            (1,),
            device=images.device,
        ).item()
        left = torch.randint(
            0,
            max(width - cutout_width + 1, 1),
            (1,),
            device=images.device,
        ).item()
        images[index, :, top:top + cutout_height, left:left + cutout_width] = 0.0
    return images


class DiscriminatorAugment(nn.Module):
    """Lightweight discriminator augmentation matching the decoder architecture plan."""

    def __init__(
        self,
        prob: float = 1.0,
        cutout: float = 0.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        horizontal_flip: bool = True,
    ) -> None:
        super().__init__()
        self.prob = prob
        self.cutout = cutout
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.horizontal_flip = horizontal_flip

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob <= 0.0:
            return images

        batch_size = images.shape[0]
        apply_mask = torch.rand(batch_size, device=images.device) < self.prob
        if not torch.any(apply_mask):
            return images

        augmented = images.clone()
        selected = augmented[apply_mask]

        if self.horizontal_flip:
            flip_mask = torch.rand(selected.shape[0], device=selected.device) < 0.5
            if torch.any(flip_mask):
                selected[flip_mask] = torch.flip(selected[flip_mask], dims=(-1,))

        if self.brightness > 0.0:
            brightness = selected.new_empty(selected.shape[0], 1, 1, 1).uniform_(
                1.0 - self.brightness,
                1.0 + self.brightness,
            )
            selected = selected * brightness

        if self.contrast > 0.0:
            contrast = selected.new_empty(selected.shape[0], 1, 1, 1).uniform_(
                1.0 - self.contrast,
                1.0 + self.contrast,
            )
            mean = selected.mean(dim=(2, 3), keepdim=True)
            selected = (selected - mean) * contrast + mean

        if self.saturation > 0.0 and selected.shape[1] == 3:
            saturation = selected.new_empty(selected.shape[0], 1, 1, 1).uniform_(
                1.0 - self.saturation,
                1.0 + self.saturation,
            )
            grayscale = selected.mean(dim=1, keepdim=True)
            selected = (selected - grayscale) * saturation + grayscale

        selected = _apply_cutout(selected, self.cutout)
        selected = selected.to(dtype=augmented.dtype)
        augmented[apply_mask] = selected
        return augmented
