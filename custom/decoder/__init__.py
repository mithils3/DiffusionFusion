"""Decoder components for the custom diffusion stack."""

from .augment import DiscriminatorAugment
from .config import (
    DecoderGanConfig,
    DecoderLossConfig,
    DecoderModelConfig,
    DecoderPlanConfig,
    DecoderTrainingConfig,
    DiscriminatorArchConfig,
    DiscriminatorAugmentConfig,
    DiscriminatorConfig,
    OptimizerConfig,
    SchedulerConfig,
    load_decoder_plan_config,
)
from .gan import (
    DecoderGanTrainingState,
    apply_noise_augmentation,
    build_decoder_gan_training_state,
    calculate_adaptive_weight,
    get_decoder_last_layer,
    images_to_minus_one_to_one,
    set_requires_grad,
)
from .discriminator import DinoPatchDiscriminator
from .losses import (
    DecoderLossBreakdown,
    LPIPSLoss,
    build_decoder_loss_breakdown,
    hinge_discriminator_loss,
    l1_reconstruction_loss,
    mse_reconstruction_loss,
    vanilla_generator_loss,
    zero_loss_like,
)
from .model import Decoder, DecoderReconstructionModel, Small

__all__ = [
    "Decoder",
    "DecoderGanConfig",
    "DecoderGanTrainingState",
    "DecoderLossBreakdown",
    "DecoderLossConfig",
    "DecoderModelConfig",
    "DecoderPlanConfig",
    "DecoderReconstructionModel",
    "DecoderTrainingConfig",
    "DinoPatchDiscriminator",
    "DiscriminatorArchConfig",
    "DiscriminatorAugment",
    "DiscriminatorAugmentConfig",
    "DiscriminatorConfig",
    "LPIPSLoss",
    "OptimizerConfig",
    "SchedulerConfig",
    "Small",
    "apply_noise_augmentation",
    "build_decoder_gan_training_state",
    "calculate_adaptive_weight",
    "build_decoder_loss_breakdown",
    "get_decoder_last_layer",
    "hinge_discriminator_loss",
    "images_to_minus_one_to_one",
    "l1_reconstruction_loss",
    "load_decoder_plan_config",
    "mse_reconstruction_loss",
    "set_requires_grad",
    "vanilla_generator_loss",
    "zero_loss_like",
]
