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
        self.dino_time_shift: float = float(getattr(args, "dino_time_shift", 0.0))

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

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(
            labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(
            labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def dino_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.dino_time_shift == 0.0:
            return t

        eps = torch.finfo(t.dtype).eps
        shifted = torch.sigmoid(torch.logit(t.clamp(eps, 1.0 - eps)) + self.dino_time_shift)
        shifted = torch.where(t <= 0.0, torch.zeros_like(shifted), shifted)
        shifted = torch.where(t >= 1.0, torch.ones_like(shifted), shifted)
        return shifted

    @staticmethod
    def _batch_time(value: float, batch_size: int, device: torch.device, ref: torch.Tensor) -> torch.Tensor:
        return torch.full((batch_size,), value, device=device).view(-1, *([1] * (ref.ndim - 1)))

    def _net_forward(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        dino_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.net, "supports_dino_time", False):
            return self.net(
                z_latent,
                z_dino,
                t.flatten(),
                labels,
                dino_t.flatten() if dino_t is not None else None,
            )
        return self.net(z_latent, z_dino, t.flatten(), labels)

    def forward(self, latent: torch.Tensor, dino: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(latent.size(
            0), device=latent.device).view(-1, *([1] * (latent.ndim - 1)))
        dino_t = self.dino_time(t)
        e_latent = torch.randn_like(latent) * self.noise_scale
        e_dino = torch.randn_like(dino) * self.noise_scale
        z_latent = t * latent + (1 - t) * e_latent
        z_dino = dino_t * dino + (1 - dino_t) * e_dino
        v_latent = (latent - z_latent) / (1 - t).clamp_min(self.t_eps)
        v_dino = (dino - z_dino) / (1 - dino_t).clamp_min(self.t_eps)

        latent_pred, dino_pred = self._net_forward(
            z_latent, z_dino, t, labels_dropped, dino_t)
        v_latent_pred = (latent_pred - z_latent) / \
            (1 - t).clamp_min(self.t_eps)
        v_dino_pred = (dino_pred - z_dino) / (1 - dino_t).clamp_min(self.t_eps)

        # L2 loss on v targets while keeping the network output in x-space.
        loss_latent = ((v_latent - v_latent_pred) ** 2).mean()
        loss_dino = ((v_dino - v_dino_pred) ** 2).mean()
        loss = (
            self.latent_loss_weight * loss_latent
            + self.dino_loss_weight * loss_dino
        )

        return loss

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
            dino_t = self.dino_time(t)
            dino_t_next = self.dino_time(t_next)
            z_latent, z_dino = stepper(
                z_latent, z_dino, t, t_next, labels, dino_t, dino_t_next)
        # Land on the model's guided x-prediction directly so the final step
        # is not shortened by the training-time velocity clamp near t=1.
        z_latent, z_dino = self._forward_sample_xpred(
            z_latent,
            z_dino,
            self._batch_time(float(timesteps[-2]), bsz, device, z_latent),
            labels,
        )
        return z_latent, z_dino

    @torch.no_grad()
    def _cfg_scale_interval(self, t: torch.Tensor) -> torch.Tensor:
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        return torch.where(interval_mask, self.cfg_scale, 1.0)

    @torch.no_grad()
    def _forward_sample_xpred(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        dino_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dino_t is None:
            dino_t = self.dino_time(t)
        # conditional
        latent_cond, dino_cond = self._net_forward(z_latent, z_dino, t, labels, dino_t)

        # unconditional
        latent_uncond, dino_uncond = self._net_forward(
            z_latent,
            z_dino,
            t,
            torch.full_like(labels, self.num_classes),
            dino_t,
        )
        cfg_scale_interval = self._cfg_scale_interval(t)
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
        t: torch.Tensor,
        labels: torch.Tensor,
        dino_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dino_t is None:
            dino_t = self.dino_time(t)
        latent_pred, dino_pred = self._forward_sample_xpred(
            z_latent, z_dino, t, labels, dino_t)
        denom_lat = (1.0 - t).clamp_min(self.inference_t_eps)
        denom_dino = (1.0 - dino_t).clamp_min(self.inference_t_eps)
        v_latent = (latent_pred - z_latent) / denom_lat
        v_dino = (dino_pred - z_dino) / denom_dino
        return v_latent, v_dino

    @torch.no_grad()
    def _euler_step(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        dino_t: torch.Tensor | None = None,
        dino_t_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dino_t is None:
            dino_t = self.dino_time(t)
        if dino_t_next is None:
            dino_t_next = self.dino_time(t_next)
        v_latent, v_dino = self._forward_sample(z_latent, z_dino, t, labels, dino_t)
        z_latent_next = z_latent + (t_next - t) * v_latent
        z_dino_next = z_dino + (dino_t_next - dino_t) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def _heun_step(
        self,
        z_latent: torch.Tensor,
        z_dino: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        dino_t: torch.Tensor | None = None,
        dino_t_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dino_t is None:
            dino_t = self.dino_time(t)
        if dino_t_next is None:
            dino_t_next = self.dino_time(t_next)
        v_latent_t, v_dino_t = self._forward_sample(
            z_latent, z_dino, t, labels, dino_t)

        z_latent_euler = z_latent + (t_next - t) * v_latent_t
        z_dino_euler = z_dino + (dino_t_next - dino_t) * v_dino_t
        v_latent_next, v_dino_next = self._forward_sample(
            z_latent_euler,
            z_dino_euler,
            t_next,
            labels,
            dino_t_next,
        )

        v_latent = 0.5 * (v_latent_t + v_latent_next)
        v_dino = 0.5 * (v_dino_t + v_dino_next)
        z_latent_next = z_latent + (t_next - t) * v_latent
        z_dino_next = z_dino + (dino_t_next - dino_t) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def update_ema(self) -> None:
        assert self.ema_params1 is not None and self.ema_params2 is not None
        for targ1, targ2, src in zip(self.ema_params1, self.ema_params2, self.parameters()):
            targ1.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
            targ2.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
