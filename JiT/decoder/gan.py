from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import JiT.util.misc as misc

from .augment import DiscriminatorAugment
from .config import DecoderLossConfig, default_decoder_plan_config
from .discriminator import DinoPatchDiscriminator
from .losses import LPIPSLoss


def _get_arg_value(args, names: tuple[str, ...], default):
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


@dataclass
class DecoderGanTrainingState:
    loss_config: DecoderLossConfig
    discriminator: nn.Module
    discriminator_optimizer: torch.optim.Optimizer
    perceptual_loss: nn.Module | None
    discriminator_augment: nn.Module | None
    noise_tau: float
    disc_lr: float
    disc_min_lr: float
    disc_warmup_epochs: int
    disc_total_epochs: int
    disc_lr_schedule: str


def set_requires_grad(module: nn.Module | None, flag: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def apply_noise_augmentation(
    latent: torch.Tensor,
    dino: torch.Tensor,
    noise_tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise_tau <= 0.0:
        return latent, dino
    latent = latent + noise_tau * torch.randn_like(latent)
    dino = dino + noise_tau * torch.randn_like(dino)
    return latent, dino


def images_to_minus_one_to_one(
    images: torch.Tensor,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
) -> torch.Tensor:
    image_mean = image_mean.to(device=images.device, dtype=images.dtype)
    image_std = image_std.to(device=images.device, dtype=images.dtype)
    images = images * image_std + image_mean
    images = images.clamp(0.0, 1.0)
    return images.mul(2.0).sub(1.0)


def get_decoder_last_layer(model: nn.Module) -> torch.nn.Parameter:
    decoder = getattr(model, "decoder", model)
    final_layer = getattr(decoder, "final_layer", None)
    linear = getattr(final_layer, "linear", None)
    if linear is not None and hasattr(linear, "weight"):
        return linear.weight

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise ValueError("Cannot compute adaptive GAN weight for a model without parameters.")
    return named_parameters[-1][1]


def calculate_adaptive_weight(
    reconstruction_loss: torch.Tensor,
    generator_loss: torch.Tensor,
    last_layer: torch.Tensor,
    max_d_weight: float,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    reconstruction_grad = torch.autograd.grad(
        reconstruction_loss,
        last_layer,
        retain_graph=True,
        allow_unused=True,
    )[0]
    generator_grad = torch.autograd.grad(
        generator_loss,
        last_layer,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if reconstruction_grad is None or generator_grad is None:
        return last_layer.new_tensor(1.0)

    d_weight = torch.norm(reconstruction_grad) / (torch.norm(generator_grad) + eps)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def build_decoder_gan_training_state(
    args,
    device: torch.device,
) -> DecoderGanTrainingState:
    plan = default_decoder_plan_config()
    loss_defaults = plan.gan.loss
    disc_defaults = plan.gan.disc
    train_defaults = plan.training
    decoder_defaults = plan.decoder

    loss_config = DecoderLossConfig(
        disc_loss=_get_arg_value(args, ("decoder_disc_loss", "disc_loss"), loss_defaults.disc_loss),
        gen_loss=_get_arg_value(args, ("decoder_gen_loss", "gen_loss"), loss_defaults.gen_loss),
        disc_weight=float(_get_arg_value(args, ("decoder_disc_weight", "disc_weight"), loss_defaults.disc_weight)),
        perceptual_weight=float(
            _get_arg_value(args, ("decoder_perceptual_weight", "perceptual_weight"), loss_defaults.perceptual_weight)
        ),
        adaptive_weight=bool(
            _get_arg_value(args, ("decoder_adaptive_weight", "adaptive_d_weight"), loss_defaults.adaptive_weight)
        ),
        disc_start=int(_get_arg_value(args, ("decoder_disc_start", "disc_start"), loss_defaults.disc_start)),
        disc_upd_start=int(
            _get_arg_value(args, ("decoder_disc_upd_start", "disc_upd_start"), loss_defaults.disc_upd_start)
        ),
        lpips_start=int(_get_arg_value(args, ("decoder_lpips_start", "lpips_start"), loss_defaults.lpips_start)),
        max_d_weight=float(_get_arg_value(args, ("decoder_max_d_weight", "max_d_weight"), loss_defaults.max_d_weight)),
        disc_updates=int(_get_arg_value(args, ("decoder_disc_updates", "disc_updates"), loss_defaults.disc_updates)),
    )

    disc_arch = disc_defaults.arch
    discriminator = DinoPatchDiscriminator(
        **disc_arch.build_kwargs(),
        pretrained=bool(_get_arg_value(args, ("decoder_disc_pretrained",), True)),
    ).to(device)
    if misc.is_dist_avail_and_initialized():
        device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
        discriminator = DDP(
            discriminator,
            device_ids=device_ids,
            broadcast_buffers=False,
        )
    disc_optimizer_cfg = disc_defaults.optimizer
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(_get_arg_value(args, ("decoder_disc_lr", "disc_lr"), disc_optimizer_cfg.lr)),
        betas=tuple(_get_arg_value(args, ("decoder_disc_betas", "disc_betas"), disc_optimizer_cfg.betas)),
        weight_decay=float(
            _get_arg_value(args, ("decoder_disc_weight_decay", "disc_weight_decay"), disc_optimizer_cfg.weight_decay)
        ),
    )

    lpips_net = str(_get_arg_value(args, ("decoder_lpips_net", "lpips_net"), "vgg"))
    perceptual_loss = LPIPSLoss(net=lpips_net).to(device)

    augment_cfg = disc_defaults.augment
    discriminator_augment = DiscriminatorAugment(
        **augment_cfg.build_kwargs()
    ).to(device)

    return DecoderGanTrainingState(
        loss_config=loss_config,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        perceptual_loss=perceptual_loss,
        discriminator_augment=discriminator_augment,
        noise_tau=float(_get_arg_value(args, ("decoder_noise_tau", "noise_tau"), decoder_defaults.noise_tau)),
        disc_lr=float(_get_arg_value(args, ("decoder_disc_lr", "disc_lr"), disc_optimizer_cfg.lr)),
        disc_min_lr=float(_get_arg_value(args, ("decoder_disc_min_lr", "disc_min_lr"), train_defaults.scheduler.final_lr)),
        disc_warmup_epochs=int(
            _get_arg_value(args, ("decoder_disc_warmup_epochs", "warmup_epochs"), train_defaults.scheduler.warmup_epochs)
        ),
        disc_total_epochs=int(_get_arg_value(args, ("decoder_disc_epochs", "epochs"), train_defaults.epochs)),
        disc_lr_schedule=str(_get_arg_value(args, ("decoder_disc_lr_schedule", "lr_schedule"), train_defaults.scheduler.type)),
    )
