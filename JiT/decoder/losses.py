from dataclasses import dataclass

import torch
from torch import nn


def l1_reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(reconstructed - target))


def mse_reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((reconstructed - target) ** 2)


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = torch.relu(1.0 - real_logits).mean()
    fake_loss = torch.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def r1_gradient_penalty(
    real_logits: torch.Tensor,
    real_images: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty: penalize squared gradient norm on real images."""
    (grad_real,) = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
    )
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


def vanilla_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _prepare_lpips_input(
    images: torch.Tensor,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
) -> torch.Tensor:
    images = images.float()
    if image_mean is not None and image_std is not None:
        image_mean = image_mean.to(device=images.device, dtype=images.dtype)
        image_std = image_std.to(device=images.device, dtype=images.dtype)
        images = images * image_std + image_mean
    images = images.clamp(0.0, 1.0)
    return images.mul(2.0).sub(1.0)


class LPIPSLoss(nn.Module):
    """Thin LPIPS wrapper that accepts optionally normalized image tensors."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "lpips is not installed. Install it to use perceptual decoder loss."
            ) from exc
        self.metric = lpips.LPIPS(net=net)

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        image_mean: torch.Tensor | None = None,
        image_std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reconstructed = _prepare_lpips_input(
            reconstructed, image_mean, image_std)
        target = _prepare_lpips_input(target, image_mean, image_std)
        return self.metric(reconstructed, target).mean()


@dataclass(frozen=True)
class DecoderLossBreakdown:
    reconstruction: torch.Tensor
    mse: torch.Tensor
    perceptual: torch.Tensor
    adversarial: torch.Tensor
    total: torch.Tensor


def build_decoder_loss_breakdown(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    fake_logits: torch.Tensor | None = None,
    perceptual_loss_module: nn.Module | None = None,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
    perceptual_weight: float = 1.0,
    adversarial_weight: float = 0.75,
    use_perceptual: bool = True,
    use_adversarial: bool = True,
) -> DecoderLossBreakdown:
    reconstruction = l1_reconstruction_loss(reconstructed, target)
    mse = mse_reconstruction_loss(reconstructed, target)
    perceptual = zero_loss_like(reconstruction)
    adversarial = zero_loss_like(reconstruction)

    if use_perceptual and perceptual_loss_module is not None:
        perceptual = perceptual_loss_module(
            reconstructed,
            target,
            image_mean=image_mean,
            image_std=image_std,
        )

    if use_adversarial and fake_logits is not None:
        adversarial = vanilla_generator_loss(fake_logits)

    total = reconstruction + perceptual_weight * perceptual + adversarial_weight * adversarial
    return DecoderLossBreakdown(
        reconstruction=reconstruction,
        mse=mse,
        perceptual=perceptual,
        adversarial=adversarial,
        total=total,
    )
