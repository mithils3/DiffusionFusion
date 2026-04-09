import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from JiT import auto_balanced_denoiser as auto_denoiser_module
from JiT import denoiser as denoiser_module


class _ZeroPredictor(torch.nn.Module):
    def forward(self, z_latent, z_dino, t, labels):
        del t, labels
        return torch.zeros_like(z_latent), torch.zeros_like(z_dino)


class AutoBalancedDenoiserTests(unittest.TestCase):
    def _build_args(self, **overrides):
        args = dict(
            model="test-model",
            latent_size=1,
            class_num=1000,
            attn_dropout=0.0,
            proj_dropout=0.0,
            dino_hidden_size=1,
            dino_patches=1,
            label_drop_prob=0.0,
            P_mean=0.0,
            P_std=1.0,
            t_eps=0.05,
            noise_scale=0.0,
            ema_decay1=0.9999,
            ema_decay2=0.9996,
            sampling_method="heun",
            num_sampling_steps=50,
            cfg=1.0,
            interval_min=0.0,
            interval_max=1.0,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_forward_can_return_raw_stream_losses(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ZeroPredictor()},
            clear=False,
        ):
            model = auto_denoiser_module.AutoBalancedDenoiser(self._build_args())

        latent = torch.ones(1, 1, 1, 1)
        dino = torch.ones(1, 1, 1, 1)
        labels = torch.zeros(1, dtype=torch.long)

        with patch.object(model, "sample_t", return_value=torch.tensor([0.5])):
            loss_latent_raw, loss_dino_raw = model(
                latent,
                dino,
                labels,
                return_loss_components=True,
            )

        self.assertAlmostEqual(loss_latent_raw.item(), 4.0, places=5)
        self.assertAlmostEqual(loss_dino_raw.item(), 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
