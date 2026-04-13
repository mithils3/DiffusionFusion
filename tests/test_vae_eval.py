import argparse
import unittest
from types import SimpleNamespace

from JiT.eval import vae_eval as vae_eval_module


class VaeEvalHelperTests(unittest.TestCase):
    def test_resolve_checkpoint_path_prefers_explicit_checkpoint(self):
        resolved = vae_eval_module.resolve_checkpoint_path(
            "~/custom-checkpoint.pth",
            "/tmp/ignored-resume-dir",
        )

        self.assertTrue(str(resolved).endswith("custom-checkpoint.pth"))

    def test_resolve_checkpoint_path_uses_resume_dir_checkpoint_last(self):
        resolved = vae_eval_module.resolve_checkpoint_path(
            None,
            "/tmp/jit-run",
        )

        self.assertEqual(str(resolved), "/tmp/jit-run/checkpoint-last.pth")

    def test_load_checkpoint_args_accepts_namespace_payload(self):
        payload_args = argparse.Namespace(model="JiT-B/2-4C", cfg=2.9)

        loaded_args = vae_eval_module.load_checkpoint_args({"args": payload_args})

        self.assertIsInstance(loaded_args, argparse.Namespace)
        self.assertEqual(loaded_args.model, "JiT-B/2-4C")
        self.assertEqual(loaded_args.cfg, 2.9)
        self.assertIsNot(loaded_args, payload_args)

    def test_load_checkpoint_args_accepts_dict_payload(self):
        loaded_args = vae_eval_module.load_checkpoint_args(
            {"args": {"model": "JiT-B/2-4C", "cfg": 2.9}}
        )

        self.assertIsInstance(loaded_args, argparse.Namespace)
        self.assertEqual(loaded_args.model, "JiT-B/2-4C")
        self.assertEqual(loaded_args.cfg, 2.9)

    def test_select_checkpoint_key_auto_prefers_model_ema1(self):
        args = SimpleNamespace(checkpoint_key="auto")

        key = vae_eval_module.select_checkpoint_key(
            args,
            {"model": {}, "model_ema1": {}, "model_ema2": {}},
        )

        self.assertEqual(key, "model_ema1")

    def test_apply_generation_overrides_updates_sampling_fields(self):
        checkpoint_args = argparse.Namespace(
            cfg=1.0,
            interval_min=0.0,
            interval_max=1.0,
            num_sampling_steps=50,
            sampling_method="heun",
            noise_scale=1.0,
            inference_t_eps=1e-5,
            class_num=1000,
            seed=0,
        )
        cli_args = SimpleNamespace(
            cfg=2.9,
            interval_min=0.1,
            interval_max=None,
            num_sampling_steps=25,
            sampling_method="euler",
            noise_scale=None,
            inference_t_eps=1e-4,
            class_num=None,
            seed=123,
        )

        eval_args = vae_eval_module.apply_generation_overrides(checkpoint_args, cli_args)

        self.assertEqual(eval_args.cfg, 2.9)
        self.assertEqual(eval_args.interval_min, 0.1)
        self.assertEqual(eval_args.interval_max, 1.0)
        self.assertEqual(eval_args.num_sampling_steps, 25)
        self.assertEqual(eval_args.sampling_method, "euler")
        self.assertEqual(eval_args.noise_scale, 1.0)
        self.assertEqual(eval_args.inference_t_eps, 1e-4)
        self.assertEqual(eval_args.class_num, 1000)
        self.assertEqual(eval_args.seed, 123)

    def test_load_generation_decoder_requires_checkpoint_for_decoder_backend(self):
        args = SimpleNamespace(
            decode_backend="decoder",
            decoder_checkpoint=None,
            decoder_checkpoint_key="auto",
            vae_pretrained_path="stabilityai/sdxl-vae",
            local_files_only=False,
        )

        with self.assertRaisesRegex(ValueError, "requires --decoder-checkpoint"):
            vae_eval_module.load_generation_decoder(args, device="cpu")


if __name__ == "__main__":
    unittest.main()
