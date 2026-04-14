import argparse
from typing import Callable

import torch
import torch.nn as nn
from JiT.model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.net: nn.Module = JiT_models[args.model](
            input_size=args.latent_size,
            in_channels=4,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            dino_hidden_size=args.dino_hidden_size,
            dino_patches=args.dino_patches,
            dual_time_cond=getattr(args, "latent_forcing", False),
        )
        self.latent_size: int = args.latent_size
        self.latent_in_chans: int = 4
        self.num_classes: int = args.class_num
        self.dino_patches = args.dino_patches
        self.dino_hidden_size = args.dino_hidden_size
        self.label_drop_prob: float = args.label_drop_prob
        self.P_mean: float = args.P_mean
        self.P_std: float = args.P_std
        self.t_eps: float = args.t_eps
        self.inference_t_eps: float = getattr(
            args, "inference_t_eps", min(self.t_eps, 1e-5)
        )
        self.noise_scale: float = args.noise_scale
        self.latent_loss_weight: float = getattr(args, "latent_loss_weight", 1.0)
        self.dino_loss_weight: float = getattr(args, "dino_loss_weight", 1.0)
        self.latent_forcing: bool = getattr(args, "latent_forcing", False)
        self.lf_order: str = getattr(args, "lf_order", "dino_first")
        self.lf_phase_prob: float = float(getattr(args, "lf_phase_prob", 0.4))
        self.lf_dino_mean: float = float(getattr(args, "lf_dino_mean", -1.2))
        self.lf_dino_std: float = float(getattr(args, "lf_dino_std", 1.0))
        self.lf_latent_mean: float = float(getattr(args, "lf_latent_mean", -0.8))
        self.lf_latent_std: float = float(getattr(args, "lf_latent_std", 0.8))
        self.lf_context_beta: float = float(getattr(args, "lf_context_beta", 0.25))

        # ema
        self.ema_decay1: float = args.ema_decay1
        self.ema_decay2: float = args.ema_decay2
        self.ema_params1: list[torch.Tensor] | None = None
        self.ema_params2: list[torch.Tensor] | None = None

        # generation hyper params
        self.method: str = args.sampling_method
        self.steps: int = args.num_sampling_steps
        self.cfg_scale: float = args.cfg
        self.cfg_interval: tuple[float, float] = (
            args.interval_min, args.interval_max)
        self.lf_first_phase_steps: int = int(
            getattr(args, "lf_first_phase_steps", max(1, self.steps // 2))
        )

        if self.lf_order not in {"dino_first", "latent_first"}:
            raise ValueError(
                f"Unsupported latent forcing order `{self.lf_order}`."
            )
        if not 0.0 < self.lf_phase_prob < 1.0:
            raise ValueError("--lf_phase_prob must lie in (0, 1).")
        if not 0.0 <= self.lf_context_beta <= 1.0:
            raise ValueError("--lf_context_beta must lie in [0, 1].")
        if self.latent_forcing and not (0 < self.lf_first_phase_steps < self.steps):
            raise ValueError(
                "--lf_first_phase_steps must be in [1, num_sampling_steps - 1] "
                "when latent forcing is enabled."
            )

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(
            labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(
            labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _sample_t_with_stats(
        self,
        n: int,
        mean: float,
        std: float,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        z = torch.randn(n, device=device) * std + mean
        return torch.sigmoid(z)

    @staticmethod
    def _view_time(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.view(-1, *([1] * (ref.ndim - 1)))

    @staticmethod
    def _batch_time(value: float, batch_size: int, device: torch.device, ref: torch.Tensor) -> torch.Tensor:
        return torch.full((batch_size,), value, device=device).view(-1, *([1] * (ref.ndim - 1)))

    def forward(self, latent: torch.Tensor, dino: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped = self.drop_labels(labels) if self.training else labels
        if self.latent_forcing:
            return self._forward_latent_forcing(latent, dino, labels_dropped)

        t = self.sample_t(latent.size(
            0), device=latent.device).view(-1, *([1] * (latent.ndim - 1)))
        e_latent = torch.randn_like(latent) * self.noise_scale
        e_dino = torch.randn_like(dino) * self.noise_scale
        z_latent = t * latent + (1 - t) * e_latent
        z_dino = t * dino + (1 - t) * e_dino
        v_latent = (latent - z_latent) / (1 - t).clamp_min(self.t_eps)
        v_dino = (dino - z_dino) / (1 - t).clamp_min(self.t_eps)

        latent_pred, dino_pred = self.net(
            z_latent, z_dino, t.flatten(), labels_dropped)
        v_latent_pred = (latent_pred - z_latent) / \
            (1 - t).clamp_min(self.t_eps)
        v_dino_pred = (dino_pred - z_dino) / (1 - t).clamp_min(self.t_eps)

        # L2 loss on v targets while keeping the network output in x-space.
        loss_latent = ((v_latent - v_latent_pred) ** 2).mean()
        loss_dino = ((v_dino - v_dino_pred) ** 2).mean()
        loss = (
            self.latent_loss_weight * loss_latent
            + self.dino_loss_weight * loss_dino
        )

        return loss

    def _forward_latent_forcing(
        self,
        latent: torch.Tensor,
        dino: torch.Tensor,
        labels_dropped: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latent.size(0)
        device = latent.device
        first_phase = torch.rand((), device=device) < self.lf_phase_prob
        sampled_lat = self._sample_t_with_stats(
            batch_size, self.lf_latent_mean, self.lf_latent_std, device=device
        )
        sampled_dino = self._sample_t_with_stats(
            batch_size, self.lf_dino_mean, self.lf_dino_std, device=device
        )
        context_lat = 1.0 - self.lf_context_beta * torch.rand(batch_size, device=device)
        context_dino = 1.0 - self.lf_context_beta * torch.rand(batch_size, device=device)

        if self.lf_order == "dino_first":
            t_lat = torch.where(first_phase, torch.zeros_like(sampled_lat), sampled_lat)
            t_dino = torch.where(first_phase, sampled_dino, context_dino)
            dino_active = first_phase.to(dtype=sampled_dino.dtype)
        else:
            t_lat = torch.where(first_phase, sampled_lat, context_lat)
            t_dino = torch.where(first_phase, torch.zeros_like(sampled_dino), sampled_dino)
            dino_active = (~first_phase).to(dtype=sampled_dino.dtype)

        t_lat = self._view_time(t_lat, latent)
        t_dino = self._view_time(t_dino, dino)

        e_latent = torch.randn_like(latent) * self.noise_scale
        e_dino = torch.randn_like(dino) * self.noise_scale
        z_latent = t_lat * latent + (1 - t_lat) * e_latent
        z_dino = t_dino * dino + (1 - t_dino) * e_dino

        v_latent = (latent - z_latent) / (1 - t_lat).clamp_min(self.t_eps)
        v_dino = (dino - z_dino) / (1 - t_dino).clamp_min(self.t_eps)

        latent_pred, dino_pred = self.net(
            z_latent,
            z_dino,
            t_lat.flatten(),
            labels_dropped,
            t_dino.flatten(),
        )
        v_latent_pred = (latent_pred - z_latent) / (1 - t_lat).clamp_min(self.t_eps)
        v_dino_pred = (dino_pred - z_dino) / (1 - t_dino).clamp_min(self.t_eps)

        loss_latent = self.latent_loss_weight * ((v_latent - v_latent_pred) ** 2).mean()
        loss_dino = self.dino_loss_weight * ((v_dino - v_dino_pred) ** 2).mean()
        return dino_active * loss_dino + (1.0 - dino_active) * loss_latent

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        bsz = labels.size(0)
        z_latent = self.noise_scale * \
            torch.randn(bsz, self.latent_in_chans, self.latent_size,
                        self.latent_size, device=device)
        z_dino = self.noise_scale * \
            torch.randn(bsz, self.dino_hidden_size,
                        self.dino_patches, self.dino_patches, device=device)
        if self.latent_forcing:
            return self._generate_latent_forcing(z_latent, z_dino, labels)

        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        stepper: Callable[..., tuple[torch.Tensor, torch.Tensor]]
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = self._batch_time(float(timesteps[i]), bsz, device, z_latent)
            t_next = self._batch_time(float(timesteps[i + 1]), bsz, device, z_latent)
            z_latent, z_dino = stepper(
                z_latent,
                z_dino,
                t,
                t_next,
                t,
                t_next,
                labels,
                t_guidance=t,
                t_guidance_next=t_next,
            )
        # Land on the model's guided x-prediction directly so the final step
        # is not shortened by the training-time velocity clamp near t=1.
        z_latent, z_dino = self._forward_sample_xpred(
            z_latent,
            z_dino,
            self._batch_time(float(timesteps[-2]), bsz, device, z_latent),
            labels,
            t_dino=self._batch_time(float(timesteps[-2]), bsz, device, z_dino),
            t_guidance=self._batch_time(float(timesteps[-2]), bsz, device, z_latent),
        )
        return z_latent, z_dino

    @torch.no_grad()
    def _cfg_scale_interval(self, t: torch.Tensor) -> torch.Tensor:
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        return torch.where(interval_mask, self.cfg_scale, 1.0)

    @torch.no_grad()
    def _generate_latent_forcing(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lf_order == "dino_first":
            z_latent, z_dino = self._run_cascaded_phase(
                z_latent,
                z_dino,
                labels,
                active_stream="dino",
                fixed_stream_time=0.0,
                steps=self.lf_first_phase_steps,
                progress_start=0.0,
                progress_end=0.5,
            )
            z_latent, z_dino = self._run_cascaded_phase(
                z_latent,
                z_dino,
                labels,
                active_stream="latent",
                fixed_stream_time=1.0,
                steps=self.steps - self.lf_first_phase_steps,
                progress_start=0.5,
                progress_end=1.0,
            )
        else:
            z_latent, z_dino = self._run_cascaded_phase(
                z_latent,
                z_dino,
                labels,
                active_stream="latent",
                fixed_stream_time=0.0,
                steps=self.lf_first_phase_steps,
                progress_start=0.0,
                progress_end=0.5,
            )
            z_latent, z_dino = self._run_cascaded_phase(
                z_latent,
                z_dino,
                labels,
                active_stream="dino",
                fixed_stream_time=1.0,
                steps=self.steps - self.lf_first_phase_steps,
                progress_start=0.5,
                progress_end=1.0,
            )
        return z_latent, z_dino

    @torch.no_grad()
    def _run_cascaded_phase(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        labels: torch.Tensor,
        *,
        active_stream: str,
        fixed_stream_time: float,
        steps: int,
        progress_start: float,
        progress_end: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if steps <= 0:
            return z_latent, z_dino

        device = labels.device
        batch_size = labels.size(0)
        phase = torch.linspace(0.0, 1.0, steps + 1, device=device)
        progress = torch.linspace(progress_start, progress_end, steps + 1, device=device)

        stepper: Callable[..., tuple[torch.Tensor, torch.Tensor]]
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(steps - 1):
            t_active = float(phase[i])
            t_active_next = float(phase[i + 1])
            t_guidance = float(progress[i])
            t_guidance_next = float(progress[i + 1])

            if active_stream == "latent":
                t_lat = self._batch_time(t_active, batch_size, device, z_latent)
                t_lat_next = self._batch_time(t_active_next, batch_size, device, z_latent)
                t_dino = self._batch_time(fixed_stream_time, batch_size, device, z_dino)
                t_dino_next = t_dino
            else:
                t_lat = self._batch_time(fixed_stream_time, batch_size, device, z_latent)
                t_lat_next = t_lat
                t_dino = self._batch_time(t_active, batch_size, device, z_dino)
                t_dino_next = self._batch_time(t_active_next, batch_size, device, z_dino)

            z_latent, z_dino = stepper(
                z_latent,
                z_dino,
                t_lat,
                t_lat_next,
                t_dino,
                t_dino_next,
                labels,
                t_guidance=self._batch_time(t_guidance, batch_size, device, z_latent),
                t_guidance_next=self._batch_time(t_guidance_next, batch_size, device, z_latent),
            )

        t_final = float(phase[-2])
        guidance_final = float(progress[-2])
        if active_stream == "latent":
            t_lat = self._batch_time(t_final, batch_size, device, z_latent)
            t_dino = self._batch_time(fixed_stream_time, batch_size, device, z_dino)
        else:
            t_lat = self._batch_time(fixed_stream_time, batch_size, device, z_latent)
            t_dino = self._batch_time(t_final, batch_size, device, z_dino)

        latent_pred, dino_pred = self._forward_sample_xpred(
            z_latent,
            z_dino,
            t_lat,
            labels,
            t_dino=t_dino,
            t_guidance=self._batch_time(guidance_final, batch_size, device, z_latent),
        )
        if active_stream == "latent":
            z_latent = latent_pred
        else:
            z_dino = dino_pred
        return z_latent, z_dino

    @torch.no_grad()
    def _forward_sample_xpred(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t_lat: torch.Tensor,
        labels: torch.Tensor,
        t_dino: torch.Tensor | None = None,
        t_guidance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if t_dino is None:
            t_dino = t_lat
        if t_guidance is None:
            t_guidance = torch.maximum(t_lat, t_dino)
        # conditional
        latent_cond, dino_cond = self.net(
            z_latent, z_dino, t_lat.flatten(), labels, t_dino.flatten())

        # unconditional
        latent_uncond, dino_uncond = self.net(
            z_latent, z_dino, t_lat.flatten(), torch.full_like(labels, self.num_classes), t_dino.flatten())
        cfg_scale_interval = self._cfg_scale_interval(t_guidance)
        latent_pred = latent_uncond + cfg_scale_interval * \
            (latent_cond - latent_uncond)
        dino_pred = dino_uncond + cfg_scale_interval * \
            (dino_cond - dino_uncond)
        return latent_pred, dino_pred

    @torch.no_grad()
    def _forward_sample(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t_lat: torch.Tensor,
        t_dino: torch.Tensor,
        labels: torch.Tensor,
        t_guidance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_pred, dino_pred = self._forward_sample_xpred(
            z_latent, z_dino, t_lat, labels, t_dino=t_dino, t_guidance=t_guidance)
        denom_lat = (1.0 - t_lat).clamp_min(self.inference_t_eps)
        denom_dino = (1.0 - t_dino).clamp_min(self.inference_t_eps)
        v_latent = (latent_pred - z_latent) / denom_lat
        v_dino = (dino_pred - z_dino) / denom_dino
        return v_latent, v_dino

    @torch.no_grad()
    def _euler_step(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t_lat: torch.Tensor,
        t_lat_next: torch.Tensor,
        t_dino: torch.Tensor,
        t_dino_next: torch.Tensor,
        labels: torch.Tensor,
        t_guidance: torch.Tensor | None = None,
        t_guidance_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t_guidance_next
        v_latent, v_dino = self._forward_sample(
            z_latent, z_dino, t_lat, t_dino, labels, t_guidance=t_guidance
        )
        z_latent_next = z_latent + (t_lat_next - t_lat) * v_latent
        z_dino_next = z_dino + (t_dino_next - t_dino) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def _heun_step(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t_lat: torch.Tensor,
        t_lat_next: torch.Tensor,
        t_dino: torch.Tensor,
        t_dino_next: torch.Tensor,
        labels: torch.Tensor,
        t_guidance: torch.Tensor | None = None,
        t_guidance_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_latent_t, v_dino_t = self._forward_sample(
            z_latent, z_dino, t_lat, t_dino, labels, t_guidance=t_guidance)

        z_latent_euler = z_latent + (t_lat_next - t_lat) * v_latent_t
        z_dino_euler = z_dino + (t_dino_next - t_dino) * v_dino_t
        v_latent_next, v_dino_next = self._forward_sample(
            z_latent_euler,
            z_dino_euler,
            t_lat_next,
            t_dino_next,
            labels,
            t_guidance=t_guidance_next,
        )

        v_latent = 0.5 * (v_latent_t + v_latent_next)
        v_dino = 0.5 * (v_dino_t + v_dino_next)
        z_latent_next = z_latent + (t_lat_next - t_lat) * v_latent
        z_dino_next = z_dino + (t_dino_next - t_dino) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def update_ema(self) -> None:
        assert self.ema_params1 is not None and self.ema_params2 is not None
        for targ1, targ2, src in zip(self.ema_params1, self.ema_params2, self.parameters()):
            targ1.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
            targ2.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
