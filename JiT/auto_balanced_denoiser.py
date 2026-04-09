import argparse

import torch

from JiT.denoiser import Denoiser


class AutoBalancedDenoiser(Denoiser):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.supports_loss_components: bool = True

    def forward(
        self,
        latent: torch.Tensor,
        dino: torch.Tensor,
        labels: torch.Tensor,
        return_loss_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        labels_dropped = self.drop_labels(labels) if self.training else labels

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

        loss_latent_raw = ((v_latent - v_latent_pred) ** 2).mean()
        loss_dino_raw = ((v_dino - v_dino_pred) ** 2).mean()
        if return_loss_components:
            return loss_latent_raw, loss_dino_raw
        return loss_latent_raw + loss_dino_raw
