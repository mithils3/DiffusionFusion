# Custom Dual JiT: path from ~8 FID-50K to ~5 FID-50K at 80 epochs

Date: 2026-04-23

## Current repo read

- Training script: `sbatch/jit_training.sbatch`
  - Model: `JiT-Dual-B/2-4C-896`
  - Epoch budget: 80
  - Effective batch size: `128 * accum_iter 2 * 4 GPUs = 1024`
  - LR: `blr=5e-5`, so actual LR should be `2e-4`
  - Schedule: default run name says constant, and `LR_SCHEDULE=constant`
  - Sampling/eval: Heun, 50 steps, CFG `2.9`, interval `[0.1, 1.0]`
  - Eval currently defaults to `NUM_IMAGES=10000`, so final FID claims must still be rerun at 50K.

- Model code: `JiT/model_jit.py`
  - Baseline shared-trunk `JiT-B/2-4C` has about 235M trainable params.
  - Custom dual stream `JiT-Dual-B/2-4C-896` has about 376M trainable params.
  - `JiT-L/2-4C` has about 462M trainable params.
  - Dual stream separates latent and DINO towers, adds context at block 0, and fuses bidirectionally at blocks 4 and 8.
  - Dual model supports separate DINO time conditioning through `supports_dino_time=True`.

- Denoiser code: `JiT/denoiser.py`
  - Training samples a single latent time `t = sigmoid(N(P_mean, P_std))`, currently `P_mean=-0.8`, `P_std=0.8`.
  - DINO time is then shifted in logit space by `log(sqrt(dino_dim / latent_dim))`.
  - With DINO `[768,16,16]` and SD-VAE latent `[4,32,32]`, `sqrt(196608 / 4096) = sqrt(48) = 6.93`, so `t=0.5` maps to `dino_t~=0.874`.
  - The latent branch itself still uses the old JiT/SD-VAE schedule, not a dimension-aware schedule.
  - Loss is equal-weighted latent velocity MSE + DINO velocity MSE.

- Feature/decoder code:
  - DINO features are per-token layer-normalized during extraction.
  - SDXL-VAE latents are not dataset-normalized beyond VAE scaling factor.
  - Decoder has RAE-style noise augmentation (`decoder_noise_tau`, default config says `0.2`; earlier dataclass default is `0.4`, so check the actual checkpoint args).
  - The decoder rFID in `JiT/decoder/RESULTS.md` is strong enough that the denoiser/sampler is the likely bottleneck, not reconstruction.

## Literature signal

- RAE / DiT with Representation Autoencoders says high-dimensional representation latents need three things: transformer width at least comparable to token dimension, dimension-dependent noise/time shift, and noise-augmented decoder training. Their project page reports standard DiT-XL on DINOv2-B RAE latents at 4.28 gFID after 80 epochs, and DiT-DH-XL at 2.16 after 80 epochs. Sources: arXiv 2510.11690 and project page.
  - https://arxiv.org/abs/2510.11690
  - https://rae-dit.github.io/

- The RAE schedule shift formula is:
  - `t_m = alpha * t_n / (1 + (alpha - 1) * t_n)`
  - `alpha = sqrt(m / n)`, with base `n=4096`
  - This is already conceptually in our DINO shift, but only as `dino_t = sigmoid(logit(t) + log(alpha))`.
  - For the combined dual target, the effective dimension is `m = latent_dim + dino_dim = 4096 + 196608 = 200704`, so `alpha = sqrt(49) ~= 7.0`.

- Latent Forcing is very relevant because our model is already a latent+image/latent dual denoiser. LF finds that ordering matters: denoise the semantic latent first, then use it as a scratchpad for the pixel/image stream. Their conditional 80-epoch FID-50K table: JiT 5.64 guided, JiT+REPA 4.57, LF-DiT DINOv2 4.18. Source:
  - https://arxiv.org/abs/2602.11401

- REPA is still useful for fast convergence in latent DiT, but for JiT-like high-dimensional prediction it is risky as a direct add-on. The March 2026 PixelREPA paper reports vanilla REPA can hurt JiT late in training through diversity collapse, while masked/adapter-constrained PixelREPA improves JiT-B/16 FID from 3.66 to 3.17. Source:
  - https://arxiv.org/abs/2603.14366

## Highest-probability path to 5 FID

### 1. First evaluate the current time-shift run correctly

The recent DINO time-shift change may already recover a lot. Do not compare the old 8 FID against 10K or a stale CFG.

Run:

- Final eval at 50K images, not 10K.
- Sweep both checkpoint EMAs (`model_ema1`, `model_ema2`) if supported by eval.
- Sweep CFG wider than current bounds.

Concrete sweep:

- `CFG_VALUES="2.5,2.9,3.3,3.7,4.1,4.5,5.0,5.5,6.0,7.0,8.0,10.0"`
- `INTERVAL_MIN` in `{0.05, 0.10, 0.12, 0.15, 0.20}`
- `INTERVAL_MAX=1.0`
- Keep Heun 50 first, then only test 75/100 steps on the best 2 configs.

Why: LightningDiT/RAE-style high-dimensional latents often need much higher CFG than SD-VAE latents, especially with CFG interval. The repo sweep currently caps at `4.5`, which may be leaving easy FID on the table.

Expected impact: possible immediate 8 -> 5-6 if time shift helped and CFG was under-tuned.

### 2. Add proper global timestep shift, not only DINO-relative time shift

Current code samples latent `t` from the old `P_mean=-0.8/P_std=0.8`, then shifts DINO forward. This creates an ordered trajectory, but the global schedule is still calibrated for 4096-dim SD-VAE latents.

Add CLI args:

- `--time_shift`
- `--time_shift_base_dim 4096`
- `--time_shift_target_dim auto|latent|dino|combined`

Implement:

- Sample base `t_base = sigmoid(N(P_mean, P_std))`.
- Shift latent time too:
  - `t = shift_time(t_base, alpha_latent)` where `alpha_latent` is usually 1.0 for SD-VAE latents.
  - For dual generation, test `alpha_global = sqrt((latent_dim + dino_dim) / 4096) ~= 7.0`.
- Derive DINO time either from that global `t` or from `t_base`; test both.

Candidate schedules:

- A: current repo: `latent_t=t_base`, `dino_t=shift(t_base, alpha_dino)`.
- B: RAE-combined: `latent_t=shift(t_base, alpha_combined)`, `dino_t=latent_t`.
- C: LF-style ordered: `latent_t=t_base`, `dino_t=shift(t_base, alpha_dino)`, but train with `dino_loss_weight > latent_loss_weight` early.
- D: cascaded-ish: sample either DINO step or latent step; when training DINO, set latent noisy/full-noise and latent loss 0; when training latent, keep DINO clean or lightly noised.

Expected impact: high. RAE reports schedule shift as the difference between failure-ish FID and ~4.8 before decoder noise augmentation. Our current DINO shift is close, but not the same as shifting the actual global corruption distribution.

### 3. Move architecture closer to RAE-DH or Latent Forcing

Current dual-stream is powerful, but it doubles most trunk compute and keeps only two cross-fusion layers. Two better variants are worth implementing:

Variant A: Additive-token LF model

- Embed latent tokens and DINO tokens separately.
- Add them into one 256-token sequence instead of keeping 512+context tokens.
- Use two time embeddings, one for latent, one for DINO.
- Add two output heads.
- Optionally split the last 4 blocks into two output experts as LF tried.

Why: Latent Forcing gets the multi-tokenizer benefit without doubling sequence length. Their paper says minimal architecture changes and ordering are the main win.

Variant B: Wide shallow DINO head

- Keep current 896-wide dual trunk.
- Add a 2-layer, 2048-wide DINO/dual denoising head after the trunk, similar to RAE DiT-DH.
- Let the main trunk stay moderate width, while the head handles high-dimensional DINO token denoising.

Why: RAE says width should match token dimension; our 896 hidden size exceeds DINOv2-B token dim 768, so the core passes the basic threshold, but the final DINO projection is still a single linear head from 896 to 768. A shallow wide head is exactly the RAE trick that improves 4.28 -> 2.16 at 80 epochs.

Expected impact: medium-high, but needs a full 80-epoch rerun.

### 4. Tune dual loss weighting

Equal `latent_loss_weight=1.0` and `dino_loss_weight=1.0` may not be optimal. DINO has 48x more scalar dimensions, but because each loss uses `.mean()`, both branches have equal aggregate weight, not equal per-scalar or equal usefulness.

Grid:

- `dino_loss_weight`: `0.25, 0.5, 1.0, 2.0`
- `latent_loss_weight`: keep `1.0`

Watch:

- DINO branch train loss
- latent branch train loss
- final decoder sample quality

Hypothesis: if generated DINO is acting as decoder semantic control, undertraining it hurts FID more than overtraining it. But if DINO loss steals capacity from the SD-VAE latent, FID will worsen. This is cheap to determine with 10-20 epoch pilots.

### 5. Add training-time DINO noise during latent/pixel steps

Latent Forcing found that small training-time latent noise during pixel steps prevents overfitting/cascaded error, but inference noise is harmful.

Repo translation:

- Add `--dino_condition_noise_max`.
- During latent reconstruction-heavy steps, keep `dino_t` slightly below 1 or add explicit feature noise to DINO conditioning.
- Do not add this noise at inference.

Candidate:

- `dino_condition_noise_max=0.10` and `0.25`
- Only apply when `t > 0.5` or on a randomly selected 10-25% of batches.

Expected impact: medium, especially if samples look semantically right but have decoder artifacts or brittle details.

### 6. Normalize SD-VAE latents or at least measure their stats

DINO tokens are layer-normalized. SD-VAE latents are not normalized beyond `vae.config.scaling_factor`. This mismatch means a single `noise_scale=1.0` and shared loss scale may not match both streams.

Add a small stats script over the latent shards:

- global mean/std for SD-VAE latents
- per-channel mean/std
- DINO mean/std after token LN

If latent std is far from 1, add optional dataset normalization metadata and inverse it before decoder, or set `noise_scale`/time shift to match actual latent variance.

Expected impact: low-medium, but it derisks all schedule work.

### 7. Do not make plain REPA the main bet

Plain REPA is tempting, but PixelREPA warns it can fail for JiT as training proceeds. If adding representation alignment, use a constrained version:

- shallow adapter
- partial token masking
- stop-gradient DINO target
- turn off or decay alignment late
- track per-class diversity/recall, not just early FID

Since this repo already directly denoises DINO features, REPA is also less conceptually necessary than in a pure SD-VAE model.

## Concrete implementation queue

1. Eval/sweep current checkpoint:
   - 50K FID
   - wider CFG up to 10
   - interval min sweep
   - EMA1 vs EMA2

2. Add schedule-shift controls in `JiT/denoiser.py` and args in `JiT/main_jit.py`:
   - reusable `shift_time(t, alpha)`
   - `latent_time(t_base)`
   - `dino_time(t_base)` with selectable modes
   - tests matching RAE formula and endpoint behavior

3. Add `--dino_loss_weight`/`--latent_loss_weight` to sbatch sweeps if not already surfaced in the training script.

4. Add a `jit_time_schedule_sweep.sbatch` for 10-20 epoch pilots:
   - current
   - combined shift
   - stronger DINO shift
   - reduced/increased DINO loss

5. If current-shift + CFG sweep is still above 6 FID, implement the LF additive-token variant. It is probably the most repo-compatible architectural change.

6. If LF-style schedule works but plateaus above 5, implement the RAE-DH wide shallow head.

## My best bet

For this exact repo, the fastest route to ~5 FID-50K without increasing epochs is:

1. Evaluate the untested time-shift checkpoint with a much wider CFG/interval sweep.
2. If it is not already near 5, add RAE-style global timestep shift controls and test combined effective dimension.
3. If a full retrain is allowed, switch the dual architecture toward Latent Forcing ordering: DINO semantics denoise earlier and condition the image/latent branch, with either additive 256-token fusion or stronger late output experts.

The current dual-stream model has enough parameter count and width. The likely missing pieces are schedule/order calibration and sampler guidance, not simply more epochs.
