from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal


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
    backbone_model_name: str = "timm/vit_small_patch8_224.dino"
    dino_ckpt_path: str | None = None
    input_size: int = 224
    feature_dim: int = 384
    ks: int = 9
    norm_type: NormType = "bn"
    using_spec_norm: bool = True
    freeze_backbone: bool = True
    recipe: str = "S_8"
    key_depths: tuple[int, ...] = (2, 5, 8, 11)
    norm_eps: float = 1.0e-6


@dataclass(frozen=True)
class DiscriminatorAugmentConfig:
    prob: float = 1.0
    cutout: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    horizontal_flip: bool = False


@dataclass(frozen=True)
class DiscriminatorConfig:
    arch: DiscriminatorArchConfig = field(default_factory=DiscriminatorArchConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augment: DiscriminatorAugmentConfig = field(default_factory=DiscriminatorAugmentConfig)


@dataclass(frozen=True)
class DecoderLossConfig:
    disc_loss: DiscLossName = "hinge"
    gen_loss: GenLossName = "vanilla"
    disc_weight: float = 1.5
    perceptual_weight: float = 1.0
    adaptive_weight: bool = True
    disc_start: int = 6
    disc_upd_start: int = 4
    adversarial_warmup_epochs: float = 2.0
    lpips_start: int = 0
    max_d_weight: float = 5.0
    disc_updates: int = 1

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


def _coerce_betas(value: Any) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected exactly 2 beta values, got {value}.")
        return float(value[0]), float(value[1])
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected exactly 2 beta values, got {value}.")
        return float(value[0]), float(value[1])
    raise TypeError(f"Unsupported optimizer betas value: {value!r}")


def _require_mapping(path: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a mapping, got {type(data).__name__}.")
    return data


def _coerce_scalar(raw_value: Any, default_value: Any, *, path: str) -> Any:
    if isinstance(default_value, bool):
        if not isinstance(raw_value, bool):
            raise TypeError(f"{path} must be a boolean, got {raw_value!r}.")
        return raw_value
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, str):
        return str(raw_value)
    if default_value is None:
        if raw_value is None or isinstance(raw_value, str):
            return raw_value
        raise TypeError(f"{path} must be a string or null, got {raw_value!r}.")
    return raw_value


def _merge_dataclass(defaults: Any, data: dict[str, Any], *, path: str) -> Any:
    mapping = _require_mapping(path, data)
    allowed_keys = {field.name for field in fields(defaults)}
    unknown_keys = sorted(set(mapping) - allowed_keys)
    if unknown_keys:
        raise KeyError(f"Unsupported keys in {path}: {', '.join(unknown_keys)}.")

    values = {}
    for dataclass_field in fields(defaults):
        field_name = dataclass_field.name
        default_value = getattr(defaults, field_name)
        if field_name not in mapping:
            values[field_name] = default_value
            continue

        raw_value = mapping[field_name]
        field_path = f"{path}.{field_name}"
        if is_dataclass(default_value):
            values[field_name] = _merge_dataclass(
                default_value,
                _require_mapping(field_path, raw_value),
                path=field_path,
            )
        elif field_name == "betas":
            values[field_name] = _coerce_betas(raw_value)
        else:
            values[field_name] = _coerce_scalar(raw_value, default_value, path=field_path)
    return type(defaults)(**values)


def load_decoder_plan_config(path: str | None = None) -> DecoderPlanConfig:
    defaults = DecoderPlanConfig()
    if path is None:
        return defaults

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load decoder config files. Install `pyyaml` or omit --config."
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Decoder config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        raise ValueError(f"Decoder config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Decoder config file must contain a mapping at the top level, got {type(payload).__name__}."
        )

    return _merge_dataclass(defaults, payload, path="decoder config")
