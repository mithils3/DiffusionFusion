from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import JiT.util.misc as misc

from .augment import DiscriminatorAugment
from .config import DecoderLossConfig
from .discriminator import DinoPatchDiscriminator
from .losses import LPIPSLoss


@dataclass
class DecoderGanTrainingState:
    loss_config: DecoderLossConfig
    discriminator: nn.Module
    discriminator_optimizer: torch.optim.Optimizer
    perceptual_loss: nn.Module | None
    discriminator_augment: nn.Module
    noise_tau: float
    disc_lr: float
    disc_min_lr: float
    disc_warmup_epochs: int
    disc_total_epochs: int
    disc_lr_schedule: str
    disc_lr_epoch_offset: float
    discriminator_step: int = 0


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def apply_noise_augmentation(
    eva: torch.Tensor,
    noise_tau: float,
) -> torch.Tensor:
    if noise_tau <= 0.0:
        return eva

    batch_size = eva.shape[0]
    eva_sigma = eva.new_empty(batch_size).normal_(mean=0.0, std=noise_tau).abs_()

    eva_sigma = eva_sigma.view(batch_size, *([1] * (eva.ndim - 1)))

    eva = eva + eva_sigma * torch.randn_like(eva)
    return eva


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
    if linear is None or not hasattr(linear, "weight"):
        raise AttributeError(
            "Adaptive decoder GAN weighting expects decoder.final_layer.linear.weight."
        )
    return linear.weight


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
    )[0]
    generator_grad = torch.autograd.grad(
        generator_loss,
        last_layer,
        retain_graph=True,
    )[0]

    d_weight = torch.norm(reconstruction_grad) / (torch.norm(generator_grad) + eps)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def build_decoder_gan_training_state(
    args,
    device: torch.device,
) -> DecoderGanTrainingState:
    plan = args.decoder_plan
    disc_defaults = plan.gan.disc

    loss_config = DecoderLossConfig(
        disc_loss=args.decoder_disc_loss,
        gen_loss=args.decoder_gen_loss,
        disc_weight=float(args.decoder_disc_weight),
        perceptual_weight=float(args.decoder_perceptual_weight),
        adaptive_weight=bool(args.decoder_adaptive_weight),
        disc_start=int(args.decoder_disc_start),
        disc_upd_start=int(args.decoder_disc_upd_start),
        adversarial_warmup_epochs=float(args.decoder_adversarial_warmup_epochs),
        lpips_start=int(args.decoder_lpips_start),
        max_d_weight=float(args.decoder_max_d_weight),
        disc_updates=int(args.decoder_disc_updates),
        r1_weight=float(plan.gan.loss.r1_weight),
        r1_interval=int(plan.gan.loss.r1_interval),
    )

    discriminator = DinoPatchDiscriminator(
        backbone_model_name=args.decoder_disc_backbone_model_name,
        checkpoint_path=args.decoder_disc_ckpt_path,
        input_size=int(args.decoder_disc_input_size),
        feature_dim=int(args.decoder_disc_feature_dim),
        kernel_size=int(args.decoder_disc_kernel_size),
        norm_type=args.decoder_disc_norm_type,
        using_spec_norm=bool(args.decoder_disc_using_spec_norm),
        freeze_backbone=bool(args.decoder_disc_freeze_backbone),
        pretrained=bool(args.decoder_disc_pretrained),
    ).to(device)
    if misc.is_dist_avail_and_initialized():
        device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
        discriminator = DDP(
            discriminator,
            device_ids=device_ids,
            broadcast_buffers=False,
        )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(args.decoder_disc_lr),
        betas=args.decoder_disc_betas,
        weight_decay=float(args.decoder_disc_weight_decay),
    )

    perceptual_loss = None
    if loss_config.perceptual_weight > 0.0:
        perceptual_loss = LPIPSLoss(net=args.decoder_lpips_net).to(device)

    augment_config = disc_defaults.augment
    discriminator_augment = DiscriminatorAugment(
        prob=augment_config.prob,
        cutout=augment_config.cutout,
        brightness=augment_config.brightness,
        contrast=augment_config.contrast,
        saturation=augment_config.saturation,
        horizontal_flip=augment_config.horizontal_flip,
    ).to(device)

    return DecoderGanTrainingState(
        loss_config=loss_config,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        perceptual_loss=perceptual_loss,
        discriminator_augment=discriminator_augment,
        noise_tau=float(args.decoder_noise_tau),
        disc_lr=float(args.decoder_disc_lr),
        disc_min_lr=float(args.decoder_disc_min_lr),
        disc_warmup_epochs=int(args.decoder_disc_warmup_epochs),
        disc_total_epochs=int(args.decoder_disc_epochs),
        disc_lr_schedule=str(args.decoder_disc_lr_schedule),
        disc_lr_epoch_offset=float(loss_config.disc_upd_start),
        discriminator_step=0,
    )
