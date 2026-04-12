import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from custom import denoiser as denoiser_module
from custom import model_custom


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
            model="CustomDiT-B/2-4C",
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
            sampling_method="euler",
            num_sampling_steps=50,
            sampling_atol=1e-6,
            sampling_rtol=1e-3,
            cfg=1.0,
            interval_min=0.1,
            interval_max=1.0,
            timestep_shift=0.0,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_forward_sample_uses_inference_t_eps_near_terminal_time(self):
        with patch.dict(
            model_custom.CustomDiT_models,
            {"CustomDiT-B/2-4C": lambda **_kwargs: _ConstantPredictor(10.0)},
        ):
            model = denoiser_module.Denoiser(self._build_args())

        z_latent = torch.zeros(1, 1, 1, 1)
        z_dino = torch.zeros(1, 1, 1, 1)
        t = torch.full((1, 1, 1, 1), 0.98)
        labels = torch.zeros(1, dtype=torch.long)

        v_latent, v_dino = model._forward_sample(z_latent, z_dino, t, labels)

        self.assertAlmostEqual(v_latent.item(), 500.0, places=3)
        self.assertAlmostEqual(v_dino.item(), -500.0, places=3)

    def test_generate_runs_through_transport_ode_sampler(self):
        with patch.dict(
            model_custom.CustomDiT_models,
            {"CustomDiT-B/2-4C": lambda **_kwargs: _FinalStepPredictor(10.0)},
        ):
            model = denoiser_module.Denoiser(self._build_args())

        labels = torch.zeros(1, dtype=torch.long)
        latent, dino = model.generate(labels)

        self.assertEqual(latent.shape, (1, 4, 1, 1))
        self.assertEqual(dino.shape, (1, 1, 1, 1))
        self.assertTrue(torch.isfinite(latent).all())
        self.assertTrue(torch.isfinite(dino).all())

    def test_legacy_removed_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "removed JiT backbone"):
            denoiser_module.Denoiser(self._build_args(model="JiT-B/2-4C"))


class CustomLightningDiTSmokeTests(unittest.TestCase):
    def test_custom_backbone_preserves_latent_and_dino_shapes(self):
        model = model_custom.build_custom_dit(
            "CustomDiT-B/2-4C",
            input_size=32,
            in_channels=4,
            num_classes=1000,
            class_dropout_prob=0.1,
            dino_hidden_size=768,
            dino_patches=16,
        )
        latent = torch.randn(1, 4, 32, 32)
        dino = torch.randn(1, 768, 16, 16)
        t = torch.tensor([0.5])
        y = torch.tensor([5], dtype=torch.long)

        with torch.no_grad():
            latent_out, dino_out = model(latent, dino, t, y)

        self.assertEqual(latent_out.shape, latent.shape)
        self.assertEqual(dino_out.shape, dino.shape)


if __name__ == "__main__":
    unittest.main()
