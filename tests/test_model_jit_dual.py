import unittest

import torch

from JiT.model_jit import JiT_models


class JiTDualStreamTests(unittest.TestCase):
    def _build_model(self, **overrides):
        kwargs = dict(
            input_size=8,
            dino_patches=4,
            in_context_len=4,
        )
        kwargs.update(overrides)
        return JiT_models["JiT-Dual-B/2-4C-896"](**kwargs)

    def test_dual_variant_exposes_early_context_and_periodic_fusion(self):
        model = self._build_model()

        self.assertEqual(model.in_context_start, 0)
        self.assertEqual(model.cross_fusion_layers, (4, 8))
        self.assertTrue(model.supports_dino_time)
        self.assertEqual(model.latent_in_context_posemb.shape, (1, 4, 896))
        self.assertEqual(model.dino_in_context_posemb.shape, (1, 4, 896))

    def test_active_registry_only_exposes_dual_training_model(self):
        self.assertEqual(set(JiT_models), {"JiT-Dual-B/2-4C-896"})

    def test_dual_variant_keeps_latent_and_dino_towers_separate(self):
        model = self._build_model()

        self.assertNotEqual(
            model.latent_blocks[0].attn.qkv.weight.data_ptr(),
            model.dino_blocks[0].attn.qkv.weight.data_ptr(),
        )
        self.assertNotEqual(
            model.latent_blocks[0].mlp.w12.weight.data_ptr(),
            model.dino_blocks[0].mlp.w12.weight.data_ptr(),
        )

    def test_dual_variant_forward_preserves_external_shapes(self):
        model = self._build_model()
        latent = torch.randn(1, 4, 8, 8)
        dino = torch.randn(1, 768, 4, 4)
        t = torch.tensor([1.0])
        y = torch.tensor([5], dtype=torch.long)
        dino_t = torch.tensor([0.25])

        with torch.no_grad():
            latent_out, dino_out = model(latent, dino, t, y, dino_t=dino_t)

        self.assertEqual(latent_out.shape, latent.shape)
        self.assertEqual(dino_out.shape, dino.shape)


if __name__ == "__main__":
    unittest.main()
