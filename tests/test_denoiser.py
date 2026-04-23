import math
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


class _ConstantDinoTimePredictor(torch.nn.Module):
    supports_dino_time = True

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.seen_t = None
        self.seen_dino_t = None

    def forward(self, z_latent, z_dino, t, labels, dino_t=None):
        del labels
        self.seen_t = t.detach().clone()
        self.seen_dino_t = dino_t.detach().clone()
        return torch.full_like(z_latent, self.value), torch.full_like(z_dino, self.value)


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
            dino_time_shift=None,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_default_dino_time_shift_matches_rae_dim_ratio(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(0.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(
                    latent_size=32,
                    dino_hidden_size=768,
                    dino_patches=16,
                )
            )

        rae_shift = math.sqrt((768 * 16 * 16) / (4 * 32 * 32))
        self.assertAlmostEqual(math.exp(model.dino_time_shift), rae_shift)
        self.assertAlmostEqual(model.dino_time(torch.tensor([0.5])).item(), rae_shift / (1.0 + rae_shift))

    def test_dino_time_shift_is_logit_space_and_preserves_endpoints(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(0.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(dino_time_shift=math.log(3.0))
            )

        t = torch.tensor([0.0, 0.5, 1.0]).view(-1, 1, 1, 1)
        dino_t = model.dino_time(t)

        expected = torch.tensor([0.0, 0.75, 1.0]).view(-1, 1, 1, 1)
        self.assertTrue(torch.allclose(dino_t, expected))

    def test_forward_sample_passes_dino_time_to_dual_time_model(self):
        predictor = _ConstantDinoTimePredictor(1.0)
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: predictor},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        z_latent = torch.zeros(1, 1, 1, 1)
        z_dino = torch.zeros(1, 1, 1, 1)
        t = torch.full((1, 1, 1, 1), 0.25)
        dino_t = torch.full((1, 1, 1, 1), 0.5)
        labels = torch.zeros(1, dtype=torch.long)

        model._forward_sample_xpred(z_latent, z_dino, t, labels, dino_t)

        self.assertTrue(torch.equal(predictor.seen_t, torch.tensor([0.25])))
        self.assertTrue(torch.equal(predictor.seen_dino_t, torch.tensor([0.5])))

    def test_euler_step_uses_separate_dino_time_delta(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantDinoTimePredictor(1.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(sampling_method="euler", dino_time_shift=0.0)
            )

        z_latent = torch.zeros(1, 1, 1, 1)
        z_dino = torch.zeros(1, 1, 1, 1)
        t = torch.full((1, 1, 1, 1), 0.0)
        t_next = torch.full((1, 1, 1, 1), 0.5)
        dino_t = torch.full((1, 1, 1, 1), 0.0)
        dino_t_next = torch.full((1, 1, 1, 1), 0.75)
        labels = torch.zeros(1, dtype=torch.long)

        latent_next, dino_next = model._euler_step(
            z_latent, z_dino, t, t_next, labels, dino_t, dino_t_next)

        self.assertTrue(torch.allclose(latent_next, torch.full_like(latent_next, 0.5)))
        self.assertTrue(torch.allclose(dino_next, torch.full_like(dino_next, 0.75)))

    def test_forward_sample_uses_inference_t_eps_near_terminal_time(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(10.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

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
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        labels = torch.zeros(1, dtype=torch.long)
        latent, dino = model.generate(labels)

        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 10.0)))
        self.assertTrue(torch.allclose(dino, torch.full_like(dino, -10.0)))


if __name__ == "__main__":
    unittest.main()
