import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from JiT import denoiser as denoiser_module


class _ConstantPredictor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, z_latent, z_dino, t, labels):
        del t, labels
        latent = torch.full_like(z_latent, self.value)
        dino = torch.full_like(z_dino, -self.value)
        return latent, dino


class _FinalStepPredictor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, z_latent, z_dino, t, labels):
        del labels
        final_mask = (t > 0.97).view(-1, 1, 1, 1)
        latent = torch.where(final_mask, torch.full_like(z_latent, self.value), z_latent)
        dino = torch.where(final_mask, torch.full_like(z_dino, -self.value), z_dino)
        return latent, dino


class DenoiserSamplingTests(unittest.TestCase):
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
            inference_t_eps=1e-5,
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

    def test_forward_sample_uses_inference_t_eps_near_terminal_time(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(10.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args())

        z_latent = torch.zeros(1, 1, 1, 1)
        z_dino = torch.zeros(1, 1, 1, 1)
        t = torch.full((1, 1, 1, 1), 0.98)
        labels = torch.zeros(1, dtype=torch.long)

        v_latent, v_dino = model._forward_sample(z_latent, z_dino, t, labels)

        self.assertAlmostEqual(v_latent.item(), 500.0, places=3)
        self.assertAlmostEqual(v_dino.item(), -500.0, places=3)

    def test_generate_uses_guided_x_prediction_for_final_step(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _FinalStepPredictor(10.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args())

        labels = torch.zeros(1, dtype=torch.long)
        latent, dino = model.generate(labels)

        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 10.0)))
        self.assertTrue(torch.allclose(dino, torch.full_like(dino, -10.0)))


if __name__ == "__main__":
    unittest.main()
