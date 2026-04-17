import argparse
import unittest

import numpy as np
import torch

from JiT.eval import diffusion_decoder


class _FakeDecoder:
    def __init__(self, mean, std):
        self._output_mean = mean
        self._output_std = std


class DiffusionDecoderTests(unittest.TestCase):
    def test_decoder_images_to_uint8_inverts_saved_normalization(self):
        decoder = _FakeDecoder(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        images = torch.tensor(
            [[[[0.0]], [[1.0]], [[-1.0]]]],
            dtype=torch.float32,
        )

        uint8_images = diffusion_decoder._decoder_images_to_uint8(images, decoder)

        expected = np.array([[[[124, 173, 46]]]], dtype=np.uint8)
        self.assertTrue(np.array_equal(uint8_images, expected))

    def test_resolve_decoder_output_stats_prefers_checkpoint_values(self):
        ckpt_args = argparse.Namespace(
            image_mean=[0.1, 0.2, 0.3],
            image_std=[0.4, 0.5, 0.6],
            image_model_name="ignored-model-name",
        )

        mean, std = diffusion_decoder._resolve_decoder_output_stats(ckpt_args)

        self.assertEqual(mean, (0.1, 0.2, 0.3))
        self.assertEqual(std, (0.4, 0.5, 0.6))


if __name__ == "__main__":
    unittest.main()
