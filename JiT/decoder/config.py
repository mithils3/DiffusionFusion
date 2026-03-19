from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:
    yaml = None


DiscLossName = Literal["hinge"]
GenLossName = Literal["vanilla"]
NormType = Literal["bn", "gn", "ln"]
SchedulerType = Literal["cosine"]


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 2.0e-4
    betas: tuple[float, float] = (0.5, 0.9)
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
    dino_hidden_size: int = 768
    hidden_size: int = 1152
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    latent_patch_size: int = 2
    output_image_size: int = 256
    noise_tau: float = 0.4


@dataclass(frozen=True)
class DecoderTrainingConfig:
    epochs: int = 16
    ema_decay: float = 0.9978
    global_batch_size: int = 512
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class DiscriminatorArchConfig:
    backbone_model_name: str = "timm/vit_small_patch16_dinov3.lvd1689m"
    dino_ckpt_path: str | None = None
    input_size: int = 256
    feature_dim: int = 384
    ks: int = 9
    norm_type: NormType = "gn"
    using_spec_norm: bool = True
    recipe: str = "S_16"
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
    adversarial_warmup_epochs: float = 1.0
    lpips_start: int = 0
    max_d_weight: float = 4.0
    disc_updates: int = 1
    r1_weight: float = 0.0

    def perceptual_enabled(self, epoch: int) -> bool:
        return epoch >= self.lpips_start

    def discriminator_updates_enabled(self, epoch: int) -> bool:
        return epoch >= self.disc_upd_start

    def adversarial_enabled(self, epoch: int) -> bool:
        return epoch >= self.disc_start

    def adversarial_scale(self, epoch_progress: float) -> float:
        if epoch_progress < self.disc_start:
            return 0.0
        if self.adversarial_warmup_epochs <= 0.0:
            return 1.0
        progress = (epoch_progress - self.disc_start) / self.adversarial_warmup_epochs
        return float(min(1.0, max(0.0, progress)))


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


def _coerce_betas(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected exactly 2 beta values, got {value}.")
        return float(value[0]), float(value[1])
    raise TypeError(f"Unsupported optimizer betas value: {value!r}")


def _build_optimizer_config(data: dict[str, Any] | None, defaults: OptimizerConfig) -> OptimizerConfig:
    if not data:
        return defaults
    return OptimizerConfig(
        lr=float(data.get("lr", defaults.lr)),
        betas=_coerce_betas(data.get("betas"), defaults.betas),
        weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
    )


def _build_scheduler_config(data: dict[str, Any] | None, defaults: SchedulerConfig) -> SchedulerConfig:
    if not data:
        return defaults
    return SchedulerConfig(
        type=str(data.get("type", defaults.type)),
        warmup_epochs=int(data.get("warmup_epochs", defaults.warmup_epochs)),
        decay_end_epoch=int(data.get("decay_end_epoch", defaults.decay_end_epoch)),
        base_lr=float(data.get("base_lr", defaults.base_lr)),
        final_lr=float(data.get("final_lr", defaults.final_lr)),
        warmup_from_zero=bool(data.get("warmup_from_zero", defaults.warmup_from_zero)),
    )


def _build_decoder_model_config(
    data: dict[str, Any] | None,
    defaults: DecoderModelConfig,
) -> DecoderModelConfig:
    if not data:
        return defaults
    return DecoderModelConfig(
        dino_hidden_size=int(data.get("dino_hidden_size", defaults.dino_hidden_size)),
        hidden_size=int(data.get("hidden_size", defaults.hidden_size)),
        depth=int(data.get("depth", defaults.depth)),
        num_heads=int(data.get("num_heads", defaults.num_heads)),
        mlp_ratio=float(data.get("mlp_ratio", defaults.mlp_ratio)),
        patch_size=int(data.get("patch_size", defaults.patch_size)),
        latent_patch_size=int(data.get("latent_patch_size", defaults.latent_patch_size)),
        output_image_size=int(data.get("output_image_size", defaults.output_image_size)),
        noise_tau=float(data.get("noise_tau", defaults.noise_tau)),
    )


def _build_decoder_training_config(
    data: dict[str, Any] | None,
    defaults: DecoderTrainingConfig,
) -> DecoderTrainingConfig:
    if not data:
        return defaults
    return DecoderTrainingConfig(
        epochs=int(data.get("epochs", defaults.epochs)),
        ema_decay=float(data.get("ema_decay", defaults.ema_decay)),
        global_batch_size=int(data.get("global_batch_size", defaults.global_batch_size)),
        optimizer=_build_optimizer_config(data.get("optimizer"), defaults.optimizer),
        scheduler=_build_scheduler_config(data.get("scheduler"), defaults.scheduler),
    )


def _build_discriminator_arch_config(
    data: dict[str, Any] | None,
    defaults: DiscriminatorArchConfig,
) -> DiscriminatorArchConfig:
    if not data:
        return defaults
    return DiscriminatorArchConfig(
        backbone_model_name=str(data.get("backbone_model_name", defaults.backbone_model_name)),
        dino_ckpt_path=data.get("dino_ckpt_path", defaults.dino_ckpt_path),
        input_size=int(data.get("input_size", defaults.input_size)),
        feature_dim=int(data.get("feature_dim", defaults.feature_dim)),
        ks=int(data.get("ks", defaults.ks)),
        norm_type=str(data.get("norm_type", defaults.norm_type)),
        using_spec_norm=bool(data.get("using_spec_norm", defaults.using_spec_norm)),
        recipe=str(data.get("recipe", defaults.recipe)),
        freeze_backbone=bool(data.get("freeze_backbone", defaults.freeze_backbone)),
    )


def _build_discriminator_augment_config(
    data: dict[str, Any] | None,
    defaults: DiscriminatorAugmentConfig,
) -> DiscriminatorAugmentConfig:
    if not data:
        return defaults
    return DiscriminatorAugmentConfig(
        prob=float(data.get("prob", defaults.prob)),
        cutout=float(data.get("cutout", defaults.cutout)),
        brightness=float(data.get("brightness", defaults.brightness)),
        contrast=float(data.get("contrast", defaults.contrast)),
        saturation=float(data.get("saturation", defaults.saturation)),
        horizontal_flip=bool(data.get("horizontal_flip", defaults.horizontal_flip)),
    )


def _build_discriminator_config(
    data: dict[str, Any] | None,
    defaults: DiscriminatorConfig,
) -> DiscriminatorConfig:
    if not data:
        return defaults
    return DiscriminatorConfig(
        arch=_build_discriminator_arch_config(data.get("arch"), defaults.arch),
        optimizer=_build_optimizer_config(data.get("optimizer"), defaults.optimizer),
        augment=_build_discriminator_augment_config(data.get("augment"), defaults.augment),
    )


def _build_decoder_loss_config(
    data: dict[str, Any] | None,
    defaults: DecoderLossConfig,
) -> DecoderLossConfig:
    if not data:
        return defaults
    return DecoderLossConfig(
        disc_loss=str(data.get("disc_loss", defaults.disc_loss)),
        gen_loss=str(data.get("gen_loss", defaults.gen_loss)),
        disc_weight=float(data.get("disc_weight", defaults.disc_weight)),
        perceptual_weight=float(data.get("perceptual_weight", defaults.perceptual_weight)),
        adaptive_weight=bool(data.get("adaptive_weight", defaults.adaptive_weight)),
        disc_start=int(data.get("disc_start", defaults.disc_start)),
        disc_upd_start=int(data.get("disc_upd_start", defaults.disc_upd_start)),
        adversarial_warmup_epochs=float(
            data.get("adversarial_warmup_epochs", defaults.adversarial_warmup_epochs)
        ),
        lpips_start=int(data.get("lpips_start", defaults.lpips_start)),
        max_d_weight=float(data.get("max_d_weight", defaults.max_d_weight)),
        disc_updates=int(data.get("disc_updates", defaults.disc_updates)),
        r1_weight=float(data.get("r1_weight", defaults.r1_weight)),
    )


def _build_decoder_gan_config(
    data: dict[str, Any] | None,
    defaults: DecoderGanConfig,
) -> DecoderGanConfig:
    if not data:
        return defaults
    return DecoderGanConfig(
        disc=_build_discriminator_config(data.get("disc"), defaults.disc),
        loss=_build_decoder_loss_config(data.get("loss"), defaults.loss),
    )


def load_decoder_plan_config(path: str | None = None) -> DecoderPlanConfig:
    defaults = default_decoder_plan_config()
    if path is None:
        return defaults
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load decoder config files. Install `pyyaml` or omit --config."
        )

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Decoder config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError(
            f"Decoder config file must contain a mapping at the top level, got {type(payload).__name__}."
        )

    return DecoderPlanConfig(
        decoder=_build_decoder_model_config(payload.get("decoder"), defaults.decoder),
        training=_build_decoder_training_config(payload.get("training"), defaults.training),
        gan=_build_decoder_gan_config(payload.get("gan"), defaults.gan),
    )
