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
        self.noise_scale: float = args.noise_scale

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

    def forward(self, latent: torch.Tensor, dino: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(latent.size(
            0), device=latent.device).view(-1, *([1] * (latent.ndim - 1)))
        e_latent = torch.randn_like(latent) * self.noise_scale
        e_dino = torch.randn_like(dino) * self.noise_scale
        z_latent = t * latent + (1 - t) * e_latent
        z_dino = t * dino + (1 - t) * e_dino
        v_latent_target = (latent - z_latent) / (1 - t).clamp_min(self.t_eps)
        v_dino_target = (dino - z_dino) / (1 - t).clamp_min(self.t_eps)

        v_latent_pred, v_dino_pred = self.net(
            z_latent, z_dino, t.flatten(), labels_dropped)

        # L2 loss on native velocity targets; streams keep separate reductions
        # because their spatial resolutions differ.
        loss_latent = ((v_latent_target - v_latent_pred) ** 2).mean()
        loss_dino = ((v_dino_target - v_dino_pred) ** 2).mean()
        loss = loss_latent + loss_dino

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
        timesteps = torch.linspace(
            0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z_latent.ndim)).expand(-1, bsz, -1, -1, -1)

        stepper: Callable[..., tuple[torch.Tensor, torch.Tensor]]
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z_latent, z_dino = stepper(z_latent, z_dino, t, t_next, labels)
        # last step euler
        z_latent, z_dino = self._euler_step(
            z_latent, z_dino, timesteps[-2], timesteps[-1], labels)
        return z_latent, z_dino

    @torch.no_grad()
    def _forward_sample(self, z_latent: torch.Tensor, z_dino: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # conditional
        v_latent_cond, v_dino_cond = self.net(
            z_latent, z_dino, t.flatten(), labels)

        # unconditional
        v_latent_uncond, v_dino_uncond = self.net(
            z_latent, z_dino, t.flatten(), torch.full_like(labels, self.num_classes))

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        v_latent = v_latent_uncond + cfg_scale_interval * \
            (v_latent_cond - v_latent_uncond)
        v_dino = v_dino_uncond + cfg_scale_interval * \
            (v_dino_cond - v_dino_uncond)
        return v_latent, v_dino

    @torch.no_grad()
    def _euler_step(self, z_latent: torch.Tensor, z_dino: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_latent, v_dino = self._forward_sample(z_latent, z_dino, t, labels)
        z_latent_next = z_latent + (t_next - t) * v_latent
        z_dino_next = z_dino + (t_next - t) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def _heun_step(self, z_latent: torch.Tensor, z_dino: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_latent_t, v_dino_t = self._forward_sample(
            z_latent, z_dino, t, labels)

        z_latent_euler = z_latent + (t_next - t) * v_latent_t
        z_dino_euler = z_dino + (t_next - t) * v_dino_t
        v_latent_next, v_dino_next = self._forward_sample(
            z_latent_euler, z_dino_euler, t_next, labels)

        v_latent = 0.5 * (v_latent_t + v_latent_next)
        v_dino = 0.5 * (v_dino_t + v_dino_next)
        z_latent_next = z_latent + (t_next - t) * v_latent
        z_dino_next = z_dino + (t_next - t) * v_dino
        return z_latent_next, z_dino_next

    @torch.no_grad()
    def update_ema(self) -> None:
        assert self.ema_params1 is not None and self.ema_params2 is not None
        for targ1, targ2, src in zip(self.ema_params1, self.ema_params2, self.parameters()):
            targ1.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
            targ2.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
