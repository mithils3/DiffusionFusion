# Latent Forcing in JiT

## What We Implemented

We adapted the core Latent Forcing idea to JiT:

- use two timesteps instead of one
- train with a cascaded schedule
- denoise the semantic branch first
- denoise the rendering branch second

For this JiT codebase, the closest mapping is:

- paper "latent" branch -> JiT `dino`
- paper "pixel" branch -> JiT `latent`

So the default schedule is:

1. denoise `dino` first
2. denoise `latent` second

This is exposed as:

- `--latent_forcing`
- `--lf_order dino_first`

with `latent_first` available as a clean ablation.

## Why DINO First

The Stanford Latent Forcing paper argues that generation improves when high-level structure appears before the detail-rendering branch. In our setup:

- DINO features are the semantic scratchpad
- SDXL latents are the rendering/detail branch that the downstream decoder consumes

That makes `dino_first` the closest analogue to the paper's best-performing direction.

## Training Schedule

The model now supports two time variables:

- `t_lat`
- `t_dino`

Under latent forcing, training samples one of two phases:

### Phase 1

The first branch is active and the second branch is fully noisy.

With `dino_first`:

- sample `t_dino` from a logit-normal distribution
- set `t_lat = 0`
- optimize only the DINO loss

### Phase 2

The second branch is active and the first branch is almost clean, but receives a little training noise.

With `dino_first`:

- sample `t_lat` from a logit-normal distribution
- set `t_dino ~ U[1 - beta, 1]`
- optimize only the latent loss

This follows the paper's finding that a small amount of train-time corruption on the earlier branch helps the later branch generalize better.

## Sampling Schedule

Sampling is also cascaded:

- `25` Heun steps for the first branch
- `25` Heun steps for the second branch

With `dino_first`:

1. denoise DINO while latent stays fixed at full noise
2. denoise latent while DINO stays fixed at clean context

The first-phase branch lands on direct `x` prediction at the end of its phase, then the second phase uses that cleaned branch as context.

## Paper-Style Defaults

The new training script uses these defaults:

- model: `JiT-B/2-4C`
- epochs: `100`
- effective batch size: `1024`
- lr schedule: `constant`
- Heun steps: `50`
- first-phase probability: `0.4`
- DINO timestep sampler: `mu=-1.2`, `sigma=1.0`
- latent timestep sampler: `mu=-0.8`, `sigma=0.8`
- context noise beta: `0.25`
- loss weights: `latent=1.0`, `dino=0.333`

## Main Files

- `JiT/model_jit.py`: dual-time conditioning support
- `JiT/denoiser.py`: cascaded training and sampling logic
- `JiT/main_jit.py`: new CLI flags
- `sbatch/jit_latent_forcing_dino_first.sbatch`: paper-style training entrypoint
