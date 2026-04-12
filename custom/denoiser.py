import argparse
import torch
import torch.nn as nn
from custom.model_custom import build_custom_dit
from custom.transport import Sampler, create_transport


class Denoiser(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.net: nn.Module = build_custom_dit(
            args.model,
            input_size=args.latent_size,
            in_channels=4,
            num_classes=args.class_num,
            class_dropout_prob=args.label_drop_prob,
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
        self.P_mean: float = args.P_mean
        self.P_std: float = args.P_std
        self.t_eps: float = args.t_eps
        self.inference_t_eps: float = getattr(
            args, "inference_t_eps", min(self.t_eps, 1e-5)
        )
        self.noise_scale: float = args.noise_scale

        # ema
        self.ema_decay1: float = args.ema_decay1
        self.ema_decay2: float = args.ema_decay2
        self.ema_params1: list[torch.Tensor] | None = None
        self.ema_params2: list[torch.Tensor] | None = None

        # generation hyper params
        self.method: str = str(args.sampling_method).lower()
        self.steps: int = args.num_sampling_steps
        self.cfg_scale: float = args.cfg
        self.cfg_interval: tuple[float, float] = (
            args.interval_min, args.interval_max)
        self.timestep_shift: float = getattr(args, "timestep_shift", 0.3)
        self.sampling_atol: float = getattr(args, "sampling_atol", 1e-6)
        self.sampling_rtol: float = getattr(args, "sampling_rtol", 1e-3)
        self.transport = create_transport(path_type="Linear", prediction="velocity")
        self.sampler = Sampler(self.transport)

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, latent: torch.Tensor, dino: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t = self.sample_t(latent.size(
            0), device=latent.device).view(-1, *([1] * (latent.ndim - 1)))
        e_latent = torch.randn_like(latent) * self.noise_scale
        e_dino = torch.randn_like(dino) * self.noise_scale
        z_latent = t * latent + (1 - t) * e_latent
        z_dino = t * dino + (1 - t) * e_dino
        v_latent = (latent - z_latent) / (1 - t).clamp_min(self.t_eps)
        v_dino = (dino - z_dino) / (1 - t).clamp_min(self.t_eps)

        latent_pred, dino_pred = self.net(
            z_latent, z_dino, t.flatten(), labels)
        v_latent_pred = (latent_pred - z_latent) / \
            (1 - t).clamp_min(self.t_eps)
        v_dino_pred = (dino_pred - z_dino) / (1 - t).clamp_min(self.t_eps)

        # L2 loss on v targets while keeping the network output in x-space.
        loss_latent = ((v_latent - v_latent_pred) ** 2).mean()
        loss_dino = ((v_dino - v_dino_pred) ** 2).mean()
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
        sample_fn = self.sampler.sample_ode(
            sampling_method=self.method,
            num_steps=self.steps,
            atol=self.sampling_atol,
            rtol=self.sampling_rtol,
            reverse=False,
            timestep_shift=self.timestep_shift,
        )
        return sample_fn(
            (z_latent, z_dino),
            self._transport_velocity_model,
            labels=labels,
        )

    def _expand_t(self, t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t.view(-1, *([1] * (reference.ndim - 1)))
        return t

    def _flatten_t(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t
        return t.view(t.shape[0], -1)[:, 0]

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_expanded = self._expand_t(t, z_latent)
        t_flat = self._flatten_t(t_expanded)
        latent_in = torch.cat([z_latent, z_latent], dim=0)
        dino_in = torch.cat([z_dino, z_dino], dim=0)
        t_in = torch.cat([t_flat, t_flat], dim=0)
        labels_in = torch.cat([labels, torch.full_like(labels, self.num_classes)], dim=0)
        latent_out, dino_out = self.net(latent_in, dino_in, t_in, labels_in)
        latent_cond, latent_uncond = latent_out.chunk(2, dim=0)
        dino_cond, dino_uncond = dino_out.chunk(2, dim=0)
        cfg_scale_interval = self._cfg_scale_interval(t_expanded)
        latent_pred = latent_uncond + cfg_scale_interval * \
            (latent_cond - latent_uncond)
        dino_pred = dino_uncond + cfg_scale_interval * \
            (dino_cond - dino_uncond)
        return latent_pred, dino_pred

    @torch.no_grad()
    def _forward_sample(self, z_latent: torch.Tensor, z_dino: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_expanded = self._expand_t(t, z_latent)
        latent_pred, dino_pred = self._forward_sample_xpred(
            z_latent, z_dino, t_expanded, labels)
        denom = (1.0 - t_expanded).clamp_min(self.inference_t_eps)
        v_latent = (latent_pred - z_latent) / denom
        v_dino = (dino_pred - z_dino) / denom
        return v_latent, v_dino

    def _transport_velocity_model(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_latent, z_dino = state
        return self._forward_sample(z_latent, z_dino, t, labels)

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
