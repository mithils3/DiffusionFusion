from dataclasses import dataclass, field
from typing import Literal


DiscLossName = Literal["hinge"]
GenLossName = Literal["vanilla"]
NormType = Literal["bn", "gn", "ln"]
SchedulerType = Literal["cosine"]


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 2.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    type: SchedulerType = "cosine"
    warmup_epochs: int = 1
    decay_end_epoch: int = 16
    base_lr: float = 2.0e-4
    final_lr: float = 2.0e-5
    warmup_from_zero: bool = True


@dataclass(frozen=True)
class DecoderModelConfig:
    hidden_size: int = 1152
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    latent_patch_size: int = 2
    output_image_size: int = 256
    noise_tau: float = 0.8


@dataclass(frozen=True)
class DecoderTrainingConfig:
    epochs: int = 16
    ema_decay: float = 0.9978
    global_batch_size: int = 512
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class DiscriminatorArchConfig:
    backbone_model_name: str = "vit_small_patch8_224.dino"
    dino_ckpt_path: str = "dino_vit_small_patch8_224.pth"
    input_size: int = 224
    feature_dim: int = 384
    ks: int = 9
    norm_type: NormType = "bn"
    using_spec_norm: bool = True
    recipe: str = "S_8"
    freeze_backbone: bool = False

    def build_kwargs(self) -> dict[str, object]:
        return {
            "backbone_model_name": self.backbone_model_name,
            "checkpoint_path": self.dino_ckpt_path,
            "input_size": self.input_size,
            "feature_dim": self.feature_dim,
            "kernel_size": self.ks,
            "norm_type": self.norm_type,
            "using_spec_norm": self.using_spec_norm,
            "freeze_backbone": self.freeze_backbone,
        }


@dataclass(frozen=True)
class DiscriminatorAugmentConfig:
    prob: float = 1.0
    cutout: float = 0.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    horizontal_flip: bool = True

    def build_kwargs(self) -> dict[str, object]:
        return {
            "prob": self.prob,
            "cutout": self.cutout,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "horizontal_flip": self.horizontal_flip,
        }


@dataclass(frozen=True)
class DiscriminatorConfig:
    arch: DiscriminatorArchConfig = field(default_factory=DiscriminatorArchConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augment: DiscriminatorAugmentConfig = field(default_factory=DiscriminatorAugmentConfig)


@dataclass(frozen=True)
class DecoderLossConfig:
    disc_loss: DiscLossName = "hinge"
    gen_loss: GenLossName = "vanilla"
    disc_weight: float = 0.75
    perceptual_weight: float = 1.0
    adaptive_weight: bool = True
    disc_start: int = 8
    disc_upd_start: int = 6
    lpips_start: int = 0
    max_d_weight: float = 10000.0
    disc_updates: int = 1

    def perceptual_enabled(self, epoch: int) -> bool:
        return epoch >= self.lpips_start

    def discriminator_updates_enabled(self, epoch: int) -> bool:
        return epoch >= self.disc_upd_start

    def adversarial_enabled(self, epoch: int) -> bool:
        return epoch >= self.disc_start


@dataclass(frozen=True)
class DecoderGanConfig:
    disc: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    loss: DecoderLossConfig = field(default_factory=DecoderLossConfig)


@dataclass(frozen=True)
class DecoderPlanConfig:
    decoder: DecoderModelConfig = field(default_factory=DecoderModelConfig)
    training: DecoderTrainingConfig = field(default_factory=DecoderTrainingConfig)
    gan: DecoderGanConfig = field(default_factory=DecoderGanConfig)


def default_decoder_plan_config() -> DecoderPlanConfig:
    return DecoderPlanConfig()
