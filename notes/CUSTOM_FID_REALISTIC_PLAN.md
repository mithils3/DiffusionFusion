# Custom FID Realistic Recovery Plan

## Bottom line

Decoder work is not the main bottleneck anymore.

The current repo is not training the same model family as the JiT paper:
- Paper JiT: pixel-space, single-stream, self-contained transformer, no tokenizer, no DINO features, no learned decoder.
- This repo: SDXL-VAE latents + DINOv3 features + a dual-stream denoiser + a separate learned decoder.

Because of that, reaching the paper's reported FID with the current hybrid branch is unlikely without a major reset. The realistic strategy is:
1. Reproduce paper-level JiT with a paper-faithful pixel-space branch.
2. Independently push the current hybrid branch as far as it will go.

## What the JiT paper actually reports

Primary sources:
- Paper: https://arxiv.org/abs/2511.13720
- Official repo: https://github.com/LTH14/JiT

Reported ImageNet 256x256 FID-50K from the paper's scaling tables:
- JiT-B/16: 4.37 at 200 epochs, 3.66 at 600 epochs
- JiT-L/16: 2.79 at 200 epochs, 2.36 at 600 epochs
- JiT-H/16: 2.29 at 200 epochs, 1.86 at 600 epochs

Paper recipe details that matter:
- `x`-prediction with `v`-loss
- 50-step Heun sampler
- Linear timesteps in `[0.0, 1.0]`
- CFG sweep in `[1.0, 4.0]` with CFG interval `[0.1, 1.0]`
- Batch size 1024
- LR `2e-4`, constant, with 5 warmup epochs
- EMA sweep `{0.9996, 0.9998, 0.9999}`
- 200-600 epochs depending on the table
- Pixel space at 256 with patch size 16, giving 256 image tokens plus 32 in-context class tokens

## What this repo is actually doing now

Code-grounded facts from the current repo:
- The active training script uses `CustomDiT-B/2-4C`, not `JiT-B/16`.
- Inputs are `4x32x32` SDXL-VAE latents plus `768x16x16` DINOv3 features.
- The transformer sees 256 latent tokens plus 256 DINO tokens, then injects 32 class tokens, so the sequence is materially larger than paper JiT.
- End-to-end FID is measured after a separate learned decoder.
- The denoiser already follows the paper's key `x`-prediction plus `v`-loss pattern and uses Heun-50 sampling, logit-normal time sampling, and CFG interval guidance.
- Training already uses cosine LR in the current code path and `sbatch/custom_training.sbatch`, so the old constant-LR plateau diagnosis is stale for the current branch.
- Online eval during training still defaults to 5k images.
- VAE latents are extracted with `latent_dist.sample()`, not posterior mean.
- DINO features are extracted at 256, not native 224.
- Eval hard-codes `ema_params1` even though training tracks two EMAs.
- The decoder-only reconstruction note reports rFID `0.387`, so decoder changes alone are unlikely to close a 10-point end-to-end FID gap.

## Why 13-14 FID at 80 epochs is not surprising

1. This is a different model family than paper JiT.
The paper does not pay the cost of denoising two modalities and then learning a decoder.

2. The current branch is undertrained relative to the paper.
80 epochs is still far below the paper's 200-600 epoch results.

3. The targets are noisier than they need to be.
Sampled VAE latents freeze posterior noise per image, and DINOv3 is being used off its native 224 input size.

4. The dual-stream objective is blind to stream balance.
`loss = loss_latent + loss_dino` with no diagnostics or weighting is risky in a model that mixes a 4-channel latent stream with a 768-channel semantic stream.

5. Checkpoint selection is weaker than the paper.
The paper sweeps EMA and CFG. This repo uses one EMA at eval and 5k online FID during training.

6. The shared transformer gets no explicit modality tags.
Latent and DINO tokens share the same positional embedding and RoPE treatment even though they play very different roles.

## Realistic target setting

### Track A: paper-level FID

If the target is "match JiT paper levels", the realistic path is to reproduce paper JiT first.

Success target:
- Reproduce paper-faithful JiT-B/16 at ImageNet 256 with 50k FID within about 0.2-0.4 of the paper number.
- Only then compare future changes against that baseline.

### Track B: current hybrid branch

If we stay on the current latent+DINO+decoder design, a realistic near-term target is:
- Move from 13-14 at 80 epochs to high single digits after disciplined training and data cleanup.
- Getting all the way to about 3.7 like paper JiT is unlikely without a large architecture change.

## Recommended plan

### Phase 0: truth-check the benchmark

Goal: stop optimizing against a fuzzy target.

Actions:
- Run 50k FID on existing checkpoints at epochs 20, 40, 60, and 80.
- Sweep CFG in `{1.5, 2.0, 2.5, 2.9, 3.5}`.
- Sweep CFG interval in `{(0.0, 1.0), (0.1, 0.9), (0.1, 1.0)}`.
- Evaluate both EMA decays already tracked by the code instead of only `ema_params1`.

Why this matters:
- A noisy 5k online FID can hide real improvements or make plateaus look worse than they are.
- EMA and CFG selection are low-cost gains that the paper explicitly sweeps.

Expected upside:
- About 0.5-1.5 FID if the current checkpoints are being evaluated off the best EMA/CFG combination.

### Phase 1: clean up the targets before spending more training budget

Goal: make the hybrid task as learnable as possible.

Actions:
- Re-extract VAE latents with posterior mean instead of `latent_dist.sample()`.
- Re-extract DINO features at native 224 and upsample to 16x16 before saving.
- Keep the decoder fixed while doing these denoiser experiments so the denoiser is isolated.

Why this matters:
- Sampled latents add permanent target noise.
- DINO at 256 relies on position interpolation and is likely giving weaker semantic targets than necessary.

Expected upside:
- About 0.5-2.0 FID.

### Phase 2: make the dual-stream objective observable and controllable

Goal: stop training blind.

Actions:
- Log `loss_latent` and `loss_dino` separately.
- Log per-stream gradient norms at least periodically.
- Add explicit stream weights and sweep:
  - `(latent_weight, dino_weight) = (1.0, 1.0)`
  - `(1.0, 0.5)`
  - `(1.0, 0.25)`
  - `(2.0, 1.0)`
- Add learned type embeddings for latent tokens and DINO tokens.

Why this matters:
- The current model is not the paper's single-stream objective.
- If DINO converges faster and dominates optimization, the shared backbone may never fully serve the latent stream that actually drives pixel FID.
- Type embeddings are a cheap fix for a real ambiguity in the current sequence layout.

Expected upside:
- About 1.0-3.0 FID if stream competition is a major bottleneck.

### Phase 3: commit to a real training horizon

Goal: stop judging the branch at 80 epochs.

Actions:
- Train the best Phase 1+2 recipe to at least 200 epochs.
- Keep effective batch 1024.
- Continue using the current cosine schedule if it is producing smoother optimization than constant LR.
- Save and 50k-evaluate checkpoints every 20-40 epochs, not just at the end.

Why this matters:
- Even if the hybrid branch never reaches paper-level JiT, 80 epochs is too early to make a final call after changing targets and loss balance.

Expected upside:
- Another 1.0-3.0 FID relative to the cleaned-up 80-epoch recipe.

### Phase 4: if the goal is still paper-level FID, reset the architecture

Goal: stop asking a hybrid latent+DINO system to prove the paper's thesis.

Actions:
- Create a paper-faithful pixel-space JiT branch or use the official `LTH14/JiT` repo as the reproducibility baseline.
- Train or at least evaluate official JiT-B/16 settings on your cluster first.
- Do not use the SDXL-VAE, DINO, or the custom decoder in this baseline.
- Only after that baseline is working should you test whether any hybrid additions help.

Why this matters:
- The paper's best claim is that a plain pixel-space transformer with `x`-prediction works.
- The current branch changes the data space, conditioning structure, target semantics, and output head all at once.
- If paper-level FID is the bar, the shortest honest route is to re-enter the paper's hypothesis class.

Expected upside:
- This is the only path I would call credible for getting into the paper's FID range.

## What I would do next

1. Stop spending time on decoder-first changes for now.
2. Run a full 50k sweep over existing 20/40/60/80 checkpoints with EMA and CFG/interval variations.
3. Re-extract mean latents and native-resolution DINO features.
4. Add per-stream loss logging, stream weights, and token-type embeddings.
5. Launch a 200-epoch hybrid run with those fixes.
6. In parallel or immediately after, start a paper-faithful JiT-B/16 reproduction branch.

## Decision rule

- If the cleaned-up hybrid branch is still worse than about 8-10 FID by 200 epochs, stop treating it as a paper-level JiT path.
- At that point the right move is a paper-faithful pixel-space JiT reproduction, not more decoder tuning.

## Sources and local files audited

Sources:
- JiT paper: https://arxiv.org/abs/2511.13720
- Official JiT repo: https://github.com/LTH14/JiT

Local files audited for this note:
- `sbatch/custom_training.sbatch`
- `sbatch/custom_eval.sbatch`
- `custom/main_custom.py`
- `custom/engine_custom.py`
- `custom/denoiser.py`
- `custom/model_custom.py`
- `custom/feature_extraction/vae.py`
- `custom/feature_extraction/dinov3.py`
- `custom/decoder/RESULTS.md`
