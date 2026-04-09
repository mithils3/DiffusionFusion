import io
import unittest
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch

from JiT.engine_jit import _iter_accumulation_groups, train_one_epoch


class _ToyJitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.ema_updates = 0

    def forward(self, latent, dino, labels):
        del labels
        return self.weight * (latent.float().mean() + dino.float().mean())

    def update_ema(self):
        self.ema_updates += 1


class _ToyJitModelWithLossComponents(_ToyJitModel):
    def __init__(self):
        super().__init__()
        self.supports_loss_components = True

    def forward(self, latent, dino, labels, return_loss_components=False):
        del labels
        loss_latent_raw = self.weight * latent.float().mean()
        loss_dino_raw = 0.5 * self.weight * dino.float().mean()
        if return_loss_components:
            return loss_latent_raw, loss_dino_raw
        return loss_latent_raw + loss_dino_raw


class _FakeWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, payload, **kwargs):
        self.logged.append((dict(payload), dict(kwargs)))


class EngineJitAccumulationTests(unittest.TestCase):
    def test_iter_accumulation_groups_keeps_partial_tail(self):
        groups = [
            (list(group), group_size)
            for group, group_size in _iter_accumulation_groups(range(5), 2, 5)
        ]
        self.assertEqual(groups, [([0, 1], 2), ([2, 3], 2), ([4], 1)])

    def test_train_one_epoch_logs_after_optimizer_step_when_accumulating(self):
        model = _ToyJitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = [
            {
                "latent": torch.ones(2, 2),
                "dino": torch.full((2, 2), 0.25),
                "y": torch.zeros(2, dtype=torch.long),
            },
            {
                "latent": torch.full((2, 2), 2.0),
                "dino": torch.full((2, 2), 0.5),
                "y": torch.ones(2, dtype=torch.long),
            },
            {
                "latent": torch.full((2, 2), 3.0),
                "dino": torch.full((2, 2), 0.75),
                "y": torch.full((2,), 2, dtype=torch.long),
            },
        ]
        args = SimpleNamespace(
            accum_iter=2,
            epochs=1,
            log_freq=1,
            lr=0.1,
            lr_schedule="constant",
            min_lr=0.0,
            warmup_epochs=0,
        )
        output = io.StringIO()

        def fake_autocast(*_args, **_kwargs):
            return nullcontext()

        with patch("JiT.engine_jit.torch.autocast", fake_autocast):
            with redirect_stdout(output):
                train_one_epoch(
                    model=model,
                    model_without_ddp=model,
                    data_loader=data_loader,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    epoch=0,
                    log_writer=None,
                    args=args,
                    steps_per_epoch=3,
                    optimizer_steps_per_epoch=2,
                    wandb_run=None,
                )

        logged = output.getvalue()
        self.assertEqual(model.ema_updates, 2)
        self.assertIn("[0/2]", logged)
        self.assertIn("[1/2]", logged)
        self.assertIn("loss:", logged)
        self.assertIn("lr:", logged)

    def test_train_one_epoch_logs_monotonic_custom_wandb_step(self):
        model = _ToyJitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = [
            {
                "latent": torch.ones(2, 2),
                "dino": torch.full((2, 2), 0.25),
                "y": torch.zeros(2, dtype=torch.long),
            },
            {
                "latent": torch.full((2, 2), 2.0),
                "dino": torch.full((2, 2), 0.5),
                "y": torch.ones(2, dtype=torch.long),
            },
            {
                "latent": torch.full((2, 2), 3.0),
                "dino": torch.full((2, 2), 0.75),
                "y": torch.full((2,), 2, dtype=torch.long),
            },
        ]
        args = SimpleNamespace(
            accum_iter=2,
            epochs=1,
            log_freq=1,
            lr=0.1,
            lr_schedule="constant",
            min_lr=0.0,
            warmup_epochs=0,
        )
        wandb_run = _FakeWandbRun()

        def fake_autocast(*_args, **_kwargs):
            return nullcontext()

        with patch("JiT.engine_jit.torch.autocast", fake_autocast):
            train_one_epoch(
                model=model,
                model_without_ddp=model,
                data_loader=data_loader,
                optimizer=optimizer,
                device=torch.device("cpu"),
                epoch=0,
                log_writer=None,
                args=args,
                steps_per_epoch=3,
                optimizer_steps_per_epoch=2,
                wandb_run=wandb_run,
            )

        self.assertEqual(len(wandb_run.logged), 2)
        first_payload, first_kwargs = wandb_run.logged[0]
        second_payload, second_kwargs = wandb_run.logged[1]
        self.assertEqual(first_kwargs, {})
        self.assertEqual(second_kwargs, {})
        self.assertEqual(first_payload["trainer/global_step"], 0)
        self.assertEqual(second_payload["trainer/global_step"], 1)
        self.assertIn("train/loss", first_payload)
        self.assertIn("train/epoch_progress", second_payload)

    def test_train_one_epoch_logs_loss_components_when_available(self):
        model = _ToyJitModelWithLossComponents()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = [
            {
                "latent": torch.full((2, 2), 2.0),
                "dino": torch.full((2, 2), 4.0),
                "y": torch.zeros(2, dtype=torch.long),
            },
        ]
        args = SimpleNamespace(
            accum_iter=1,
            epochs=1,
            log_freq=1,
            lr=0.1,
            lr_schedule="constant",
            min_lr=0.0,
            warmup_epochs=0,
        )
        wandb_run = _FakeWandbRun()

        def fake_autocast(*_args, **_kwargs):
            return nullcontext()

        with patch("JiT.engine_jit.torch.autocast", fake_autocast):
            train_one_epoch(
                model=model,
                model_without_ddp=model,
                data_loader=data_loader,
                optimizer=optimizer,
                device=torch.device("cpu"),
                epoch=0,
                log_writer=None,
                args=args,
                steps_per_epoch=1,
                optimizer_steps_per_epoch=1,
                wandb_run=wandb_run,
            )

        payload, _kwargs = wandb_run.logged[0]
        self.assertAlmostEqual(payload["train/loss"], 4.0, places=5)
        self.assertAlmostEqual(payload["train/loss_latent_raw"], 2.0, places=5)
        self.assertAlmostEqual(payload["train/loss_dino_raw"], 2.0, places=5)
        self.assertAlmostEqual(payload["train/loss_latent_weighted"], 2.0, places=5)
        self.assertAlmostEqual(payload["train/loss_dino_weighted"], 2.0, places=5)
        self.assertAlmostEqual(payload["train/latent_weight"], 1.0, places=5)
        self.assertAlmostEqual(payload["train/dino_weight"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
