# JiT Dual-Stream Diffusion: End-to-End FID Deep Dive

**Date:** 2026-04-09
**Baseline:** FID ~13, 80 epochs, JiT-B/2-4C, 4 GPU, eff. batch 1024, LR 2e-4 constant

---

## System Architecture Summary

The pipeline has three stages:

1. **Feature extraction** (offline, once):
   - SDXL-VAE encodes 256x256 images to 4x32x32 latents (scaled by 0.13025).
   - DINOv3 (ViT-B/16 at 256x256, non-native) extracts 768x16x16 feature maps, per-token LayerNorm'd.

2. **Denoiser** (JiT-B/2-4C, ~131M params):
   - Jointly denoises latent (4x32x32) and DINO (768x16x16) via shared transformer with dual-stream attention.
   - patch_size=2 on latents gives 16x16=256 patches; DINO is also 16x16=256 patches.
   - Sequence: `[latent_256, dino_256]` = 512 tokens, with 32 in-context class tokens injected at block 4 (total 544 for blocks 4-11).
   - Flow matching: `z_t = t*x + (1-t)*noise`, t in [0,1], model predicts x, loss computed on v = (x-z)/(1-t).

3. **Decoder** (trained separately, ~257M params):
   - Cross-attention transformer that takes generated (latent, dino) and produces 256x256 RGB.
   - Trained with L1 + LPIPS + GAN.
   - Reconstruction FID (clean inputs): ~1.2.

End-to-end FID = denoiser quality + decoder degradation. Current split: denoiser contributes ~10-11 FID, decoder adds ~1-2.

---

## Top Findings (Ordered by Impact)

### Finding 1: t_eps Clamping Causes Severe ODE Undershoot at Inference

**Confidence: HIGH | Expected FID impact: 1-3 points**

The velocity computation clamps `(1-t)` to `t_eps=0.05` everywhere, including inference:

```python
# denoiser.py:120-122
v_latent_cond = (latent_cond - z_latent) / (1.0 - t).clamp_min(self.t_eps)
```

With 50-step Heun and uniform timesteps `linspace(0, 1, 51)`, the last ODE steps at t >= 0.96 have `(1-t) <= 0.04 < t_eps=0.05`, so velocity is artificially reduced.

**Exact math for the final Euler step (t=0.98 -> 1.0):**
```
dt = 0.02
v_clamped = (x_pred - z) / 0.05        # instead of / 0.02
z_final   = z + 0.02 * v_clamped
          = z + 0.02 * (x_pred - z) / 0.05
          = z + 0.4 * (x_pred - z)
          = 0.6 * z + 0.4 * x_pred      # should be x_pred
```

The final sample retains 60% of the residual noise/error from t=0.98. The penultimate Heun step (t=0.96->0.98) is also affected, with velocities at ~60-80% of true magnitude. Combined, the last ~4% of the ODE trajectory is severely under-traveled.

The model WAS trained with the same clamping, so the x-predictions partially compensate. But the compensation is incomplete because the v-loss objective with clamping is not equivalent to optimizing for correct final-step behavior.

**Evidence:**
- `denoiser.py:68-69` (training target clamping)
- `denoiser.py:73-75` (training prediction clamping)
- `denoiser.py:120-122` (inference clamping, conditional path)
- `denoiser.py:128-129` (inference clamping, unconditional path)
- `main_jit.py:86` (t_eps=0.05 default)

**Fix:** At inference, after the final ODE step, use the model's x-prediction directly instead of the v-to-z conversion. The model outputs x_pred natively; the v conversion is a post-hoc transformation that the clamping corrupts. Alternatively, use a much smaller t_eps (e.g., 1e-5) for inference only, or reduce the number of clamped steps by using a non-uniform timestep schedule that avoids the t_eps zone.

---

### Finding 2: Constant Learning Rate With No Decay

**Confidence: HIGH | Expected FID impact: 1-3 points (at current epoch count)**

The training uses `--lr_schedule constant` (default in `main_jit.py:76`). After 5-epoch warmup, the learning rate stays at 2e-4 forever.

```python
# lr_sched.py:9
if args.lr_schedule == "constant":
    lr = args.lr
```

This directly explains the epoch-40 plateau. Without cosine decay, the optimizer oscillates in a basin rather than settling. Every modern high-performance diffusion recipe (LightningDiT, SiT, REPA) uses cosine decay to min_lr near 0.

The existing plan's Phase 1 ("train to 200 epochs") won't help much if the LR is still constant. The model will plateau at roughly the same loss level.

**Evidence:**
- `sbatch/jit_training.sbatch` — no `--lr_schedule` flag, so defaults to `constant`
- `main_jit.py:76` — `default='constant'`
- `lr_sched.py:4-11` — cosine schedule is implemented but unused

**Fix:** Switch to `--lr_schedule cosine` with `--min_lr 1e-5` (or even 0). Expected to break through the plateau and continue improving to epoch 200+.

---

### Finding 3: Online FID Measured With Only 5k Images

**Confidence: HIGH | Expected FID impact: clarifies the plateau (may shift reported FID by 2-5 points)**

The training sbatch uses:
```bash
NUM_IMAGES="${NUM_IMAGES:-5000}"
```

FID at 5k samples has dramatically higher variance than 50k. Published benchmarks universally use 50k. The "FID ~13" number at 80 epochs may be measured at 5k, which typically overestimates (worse) compared to 50k FID by 2-5 points. The apparent "plateau" around epoch 40 could partially be measurement noise at 5k samples.

**Evidence:**
- `sbatch/jit_training.sbatch:28` — `NUM_IMAGES` defaults to 5000
- `sbatch/jit_eval.sbatch:37` — standalone eval defaults to 50000

**Fix:** Run standalone 50k eval (`jit_eval.sbatch`) on saved checkpoints at epochs 20, 40, 60, 80 to get real FID numbers. If 50k FID is already significantly better than 13, the "plateau" narrative changes.

---

### Finding 4: VAE Latent Extraction Uses Stochastic Sampling

**Confidence: MEDIUM-HIGH | Expected FID impact: 0.5-1.5 points**

```python
# vae.py:226
x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
```

`.sample()` draws from the VAE posterior, adding noise proportional to the posterior variance. Each image gets ONE random draw frozen for all of training. This permanently corrupts every training target with irreducible noise.

The SDXL-VAE posterior variance is typically small but non-zero. Using `.mean()` (posterior mode) gives strictly cleaner targets. The REPA and LightningDiT papers both use posterior mean.

**Evidence:**
- `vae.py:226` — uses `.sample()` instead of `.mean()` or `latent_dist.mode()`
- The existing improvement plan (Phase 4, item 2) correctly identifies this but labels it as a future improvement

**Fix:** Re-extract latents with `.latent_dist.mean().mul_(vae.config.scaling_factor)`. Requires one re-extraction job (~8 hours on 2 GPUs) but no code changes to training.

---

### Finding 5: DINO Features Extracted at Non-Native 256x256 Resolution

**Confidence: MEDIUM | Expected FID impact: 0.3-1.0 points**

The DINOv3 ViT-B/16 is pretrained at 224x224 (14x14 patches). The extraction runs at 256x256:

```bash
# sbatch/dinov3_features.sbatch:37
--image-size 256
```

This produces a 16x16 grid (matching the latent grid), but requires position embedding interpolation from 14x14 to 16x16. timm handles this transparently, but interpolated position embeddings degrade feature quality compared to native resolution.

**Evidence:**
- `sbatch/dinov3_features.sbatch:37` — `--image-size 256`
- `dinov3.py:95-100` — timm creates model with `img_size=args.image_size`, which triggers pos_embed interpolation
- The default in `dinov3.py:244` is `--image-size 224` but the sbatch overrides it

**Fix:** Extract at native 224 (14x14 grid), then bilinearly upsample feature maps to 16x16 before saving. Preserves DINO quality while matching the denoiser's spatial grid. Alternatively, use a 14x14 grid everywhere, requiring patch_size=2 on a 28x28 input or patch_size=1 on 14x14 (infeasible due to sequence length). The simplest approach is extract-then-upsample.

**Alternative fix (lower effort):** Accept 256 resolution but switch to a DINOv2-Large (1024 dim) or a model specifically fine-tuned at 256x256 resolution.

---

### Finding 6: Equal Loss Weight on Latent and DINO Streams

**Confidence: MEDIUM | Expected FID impact: 0.5-2.0 points**

```python
# denoiser.py:78-80
loss_latent = ((v_latent - v_latent_pred) ** 2).mean()
loss_dino = ((v_dino - v_dino_pred) ** 2).mean()
loss = loss_latent + loss_dino
```

Both streams contribute equally to the loss. But the streams differ in dimensionality (4 channels vs 768 channels per token) and in downstream importance:
- The latent stream carries fine spatial detail (texture, edges, color).
- The DINO stream carries semantic structure (object parts, layout).
- For FID, spatial detail matters more than semantic alignment (FID is computed on pixel-space images via InceptionV3, which is sensitive to texture artifacts).

With equal weights, the DINO stream provides 768/4 = 192x more gradient signal per spatial position. Even though `.mean()` normalizes over all elements, the DINO loss has much richer per-token gradients that may dominate the shared transformer's parameter updates, starving the latent stream of learning signal.

**Inference (not directly observed):** This is hard to verify without logging per-stream gradient norms, but the architectural setup makes it plausible. The symptom would be good DINO denoising but weaker latent denoising — which would show as blurry/noisy images despite correct object structure.

**Fix:** Sweep loss weights: `loss = latent_weight * loss_latent + dino_weight * loss_dino`. Try (1.0, 0.5), (1.0, 0.25), (2.0, 1.0). Log both losses separately to W&B to diagnose the balance. The simplest diagnostic: after a training run, generate samples and check whether objects have correct shapes (DINO working) but washed-out textures (latent underperforming), or vice versa.

---

### Finding 7: No Timestep-Dependent Loss Weighting (Min-SNR)

**Confidence: MEDIUM | Expected FID impact: 0.5-1.5 points**

The v-loss formulation implicitly reweights by `1/(1-t)^2` (capped at `1/t_eps^2 = 400`). This massively upweights near-clean timesteps (t near 1) and underweights noisy timesteps (t near 0).

Combined with the timestep distribution `P_mean=-0.8, P_std=0.8` which biases toward lower t (mode at sigmoid(-0.8) ~ 0.31), the training sees mostly noisy samples but weights near-clean samples ~400x more when they appear. This creates high gradient variance.

Min-SNR-gamma (Hang et al., ICCV 2023) clamps the effective weight to `min(SNR, gamma)/SNR` with gamma=5, which dramatically reduces gradient variance and has been shown to speed convergence 3.4x.

**Evidence:**
- `denoiser.py:68-69` — v-target computation with implicit 1/(1-t)^2 weighting
- `denoiser.py:56-57` — lognormal timestep sampling (P_mean=-0.8)
- `main_jit.py:83-84` — P_mean and P_std defaults
- No Min-SNR implementation anywhere in the codebase

**Fix:** Add Min-SNR-gamma weighting. Multiply each sample's loss by `min(SNR(t), gamma) / SNR(t)` where `SNR(t) = t^2 / (1-t)^2` for flow matching. This clamps the effective weight for near-clean timesteps, reducing variance.

---

### Finding 8: Uniform ODE Timestep Schedule at Inference

**Confidence: MEDIUM | Expected FID impact: 0.3-1.0 points**

```python
# denoiser.py:94-95
timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device)
```

A uniform schedule allocates equal step budget to all noise levels. LightningDiT and FLUX demonstrated that "time shifting" — concentrating steps at intermediate noise levels — improves FID by 0.3-1.0 with zero training cost.

The formula from FLUX: `shifted_t = t / (t + (1-t) * shift)` where shift > 1 pushes steps toward the middle. For ImageNet-256 latent diffusion, shift values of 1.5-3.0 are typical.

**Evidence:**
- `denoiser.py:94-95` — linspace timestep schedule
- No time-shifting code exists in the codebase

**Fix:** Replace linspace with a shifted schedule in `generate()`. This is a pure inference change, zero training cost. Can be swept as a hyperparameter.

---

### Finding 9: No Stream-Specific Modality Embeddings

**Confidence: MEDIUM-LOW | Expected FID impact: 0.3-0.8 points**

Latent and DINO tokens share identical sinusoidal `pos_embed` and the same RoPE. The model must infer stream identity purely from content statistics.

```python
# model_jit.py:431-432
latent += self.pos_embed
dino_features += self.pos_embed
```

The attention mechanism (model_jit.py:143-154) applies the same RoPE to both streams:
```python
q_lat = rope(q[:, :, num_prefix:num_prefix + num_patches])
q_dino = rope(q[:, :, num_prefix + num_patches:])
```

Adding learned type embeddings (one for latent, one for DINO) is cheap (~1536 parameters) and lets the model trivially distinguish streams without wasting attention capacity.

**Evidence:**
- `model_jit.py:431-432` — shared pos_embed for both streams
- `model_jit.py:143-148` — shared RoPE for both streams
- No modality embedding exists

**Fix:** Add `self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))` and `self.dino_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))`. Initialize with `std=0.02`. Add to respective tokens before concatenation. ~10 lines of code.

---

### Finding 10: Weight Decay = 0

**Confidence: LOW-MEDIUM | Expected FID impact: 0.2-0.5 points**

```python
# main_jit.py:78
parser.add_argument('--weight_decay', type=float, default=0.0)
```

The sbatch does not override this. Many recent DiT-family models use `weight_decay=0.05`. However, the original SiT and DiT papers also used wd=0 with AdamW. The betas are (0.9, 0.95), which is standard.

With constant LR and no weight decay, there's no regularization pressure. This may not matter much once cosine decay is added (which provides implicit regularization via decreasing LR).

**Fix:** Try `--weight_decay 0.05` in ablation after switching to cosine LR. Low priority.

---

## Verification of Existing Improvement Plan Claims

The existing plan at `notes/JIT_DIFFUSION_IMPROVEMENT_PLAN.md` makes several claims. Here's what holds up against the code:

| Claim | Verdict | Evidence |
|---|---|---|
| "FID ~13 on 50k balanced eval at 80 epochs" | **Uncertain** — online eval uses 5k, not 50k. Real 50k FID may be better. | `sbatch/jit_training.sbatch:28` |
| "effective batch 1024, LR 2e-4" | **Correct** | batch_size=64 * accum_iter=4 * 4 GPUs = 1024; blr=5e-5 * 1024/256 = 2e-4 |
| "Phase 1: train to 200 epochs -> FID 4-7" | **Overoptimistic without LR decay** — constant LR will plateau, not converge further | `lr_sched.py`, `main_jit.py:76` |
| "CFG interval tuning -> 0.4+ FID" | **Correct and underexplored** — only (0.1, 1.0) has been tested | `sbatch/jit_training.sbatch:48-49` |
| "Time shifting at inference" | **Correct recommendation, not implemented** | `denoiser.py:94-95` uses linspace |
| "Test posterior mean latents" | **Correct, high priority** | `vae.py:226` uses `.sample()` |
| "DDT-style wide shallow head" | **Reasonable but premature** — fix fundamentals first | N/A |
| "Stream loss balance sweep" | **Correct, should be done earlier** | `denoiser.py:78-80` equal weights |
| "DINO double-norm" | **Stale — appears to have been fixed already** | `dataset.py:465-466,479-480` has comments "DINO features are normalized once during extraction. Repeating layer norm here silently changes float16 shards." No normalization call at load time. |

The double-norm was listed as a known bug in project memory. The code now has comments explicitly noting NOT to re-normalize. This appears fixed but should be verified by checking if `normalize_dino_feature_map_tokens` is called anywhere in the dataset loading path (it is NOT — confirmed by reading `dataset.py`).

---

## Answering the Specific Questions

### Why might training plateau around epoch 40?

Three factors, ordered by confidence:

1. **Constant learning rate (HIGH confidence).** With LR=2e-4 and no decay, the optimizer reaches a basin around epoch 30-40 and then oscillates rather than converging. This is the most likely single cause. Cosine decay to ~1e-5 would break the plateau.

2. **5k FID measurement noise (HIGH confidence).** FID at 5k samples has std ~1-2 FID points on ImageNet. The "plateau" may partly be measurement noise masking slow improvement. 50k eval would clarify.

3. **v-loss gradient variance from t_eps interaction (MEDIUM confidence).** The 1/(1-t)^2 weighting creates ~400x variance between timesteps near 0 and timesteps near 0.95. This high gradient variance makes it hard for the optimizer to make steady progress once the easy-to-learn parts of the distribution are captured.

### Is this mostly an eval-setting plateau or a real optimization plateau?

**Both, in roughly equal measure.** The 5k FID measurement adds noise (eval artifact), while the constant LR causes genuine optimization stalling (real plateau). Running 50k eval on existing checkpoints would immediately resolve how much is each.

### Are latent and DINO balanced correctly in the current objective?

**Likely not, but hard to confirm without logging.** The equal-weight loss `loss = loss_latent + loss_dino` gives the DINO stream 192x more gradient signal per spatial position (768 vs 4 channels). Whether this actually harms the latent stream depends on how the shared transformer routes gradients. The diagnostic is to log per-stream losses during training and compare their magnitudes and convergence rates.

**Prediction:** loss_dino will converge faster and to a lower value than loss_latent, confirming that the model allocates disproportionate capacity to the easier (already high-SNR) DINO prediction task.

### Is the current x-output + v-loss setup helping or hurting?

**Mixed.** The x-prediction output is correct and enables direct use of the model's prediction at inference. But the v-loss formulation with t_eps clamping creates two issues:
1. Massive implicit reweighting toward near-clean timesteps (may or may not help, depends on the noise schedule)
2. The t_eps clamping causes ODE undershoot at inference (definitely hurts)

The v-loss itself is fine in principle — it's closely related to the velocity field loss used by SiT and flow matching models. The problem is specifically the t_eps interaction during inference.

### Is the likely bottleneck the denoiser, the feature targets, the sampler, the eval protocol, or the decoder?

**Primarily the denoiser training setup (LR schedule, loss weighting), secondarily the sampler (t_eps, uniform schedule), with minor contributions from feature targets (stochastic VAE, non-native DINO) and eval protocol (5k FID).**

The decoder (rFID ~1.2) is a minor contributor. The denoiser's training plateau is the dominant bottleneck. The sampler issues (t_eps clamping, uniform schedule) add 1-3 FID on top.

Breakdown estimate:
| Component | Estimated FID contribution to the ~13 total |
|---|---|
| Denoiser undertrained (80ep + constant LR) | 6-8 |
| Sampler deficiencies (t_eps + uniform schedule) | 1-3 |
| Feature target noise (stochastic VAE + non-native DINO) | 0.5-1.5 |
| Loss imbalance (stream weights + no Min-SNR) | 0.5-2 |
| Decoder degradation | 1-2 |
| Eval protocol (5k samples) | 0-2 (noise, not real FID) |

---

## Prioritized Improvement Plan

### Horizon 1: Immediate (test on existing checkpoints, zero or minimal code changes)

#### 1A. Run 50k FID eval on existing checkpoints
- **Rationale:** Establishes the true baseline. 5k FID is too noisy to make decisions.
- **Expected upside:** Clarifies whether FID is already ~10-11 (better than thought) or truly ~13.
- **Difficulty:** Zero code changes, one sbatch job per checkpoint.
- **Files:** `sbatch/jit_eval.sbatch`
- **Validation:** Compare 5k vs 50k FID on the same checkpoint.

#### 1B. Remove t_eps clamping at inference (use x-pred directly for final step)
- **Rationale:** The last ODE step reaches only 40% of the correction it should. Direct x-prediction sidesteps this.
- **Expected upside:** 0.5-2 FID improvement.
- **Difficulty:** ~5 lines changed in `denoiser.py`.
- **Files:** `JiT/denoiser.py` (generate, _euler_step, _forward_sample)
- **Validation:** Re-run 50k eval on existing checkpoint with and without the fix. Difference = fix impact.

**Implementation sketch:**
```python
# In generate(), replace the final Euler step:
# Instead of:
#   z_latent, z_dino = self._euler_step(z_latent, z_dino, timesteps[-2], timesteps[-1], labels)
# Do:
latent_pred, dino_pred = self._forward_sample_xpred(z_latent, z_dino, timesteps[-2], labels)
z_latent, z_dino = latent_pred, dino_pred
```

Where `_forward_sample_xpred` returns the x-prediction directly (with CFG applied in x-space, not v-space). This requires a new helper but is straightforward.

#### 1C. Sweep CFG and interval on existing checkpoint
- **Rationale:** The current CFG=2.9 with interval (0.1, 1.0) is a single point. The optimal may differ.
- **Expected upside:** 0.3-1.0 FID.
- **Difficulty:** Shell script loop, no code changes.
- **Files:** `sbatch/jit_eval.sbatch`
- **Validation:** Grid search: CFG in {1.5, 2.0, 2.5, 2.9, 3.5, 4.0} x interval in {(0,1), (0.1,0.9), (0.2,0.8), (0.1,1.0)}.

#### 1D. Sweep EMA on existing checkpoint
- **Rationale:** ema_params1 (decay=0.9999) is always used. ema_params2 (decay=0.9996) might be better at 80 epochs.
- **Expected upside:** 0.1-0.5 FID.
- **Difficulty:** Small code change to eval path to select which EMA.
- **Files:** `JiT/engine_jit.py` (evaluate function, switch `ema_params1` to `ema_params2`)
- **Validation:** Compare FID with both EMAs.

#### 1E. Add time shifting to inference schedule
- **Rationale:** Free FID improvement, well-established technique.
- **Expected upside:** 0.3-1.0 FID.
- **Difficulty:** ~5 lines in `denoiser.py`.
- **Files:** `JiT/denoiser.py` (generate function)
- **Validation:** Sweep shift parameter in {1.0, 1.5, 2.0, 3.0} with 50k eval.

**Implementation sketch:**
```python
# In generate(), replace:
#   timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device)
# With:
timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device)
shift = 2.0  # hyperparameter
timesteps = timesteps / (timesteps + (1 - timesteps) * shift)
```

---

### Horizon 2: Near-term (small-medium code changes or data regeneration)

#### 2A. Switch to cosine LR schedule
- **Rationale:** Directly addresses the epoch-40 plateau. Single most important training change.
- **Expected upside:** 2-5 FID improvement over 200 epochs (vs constant LR).
- **Difficulty:** 1-line flag change.
- **Files:** `sbatch/jit_training.sbatch` — add `--lr_schedule cosine --min_lr 1e-5`
- **Validation:** Train 200 epochs with cosine decay, compare loss curve and FID trajectory vs constant LR.

#### 2B. Re-extract latents with posterior mean
- **Rationale:** Removes frozen stochastic noise from every training target.
- **Expected upside:** 0.5-1.5 FID.
- **Difficulty:** 1-line code change + ~8 hour re-extraction job.
- **Files:** `JiT/feature_extraction/vae.py:226` — change `.sample()` to `.mean()`
- **Validation:** Train short run (40 epochs) with mean latents vs sample latents, compare loss curves.

#### 2C. Add Min-SNR-gamma loss weighting
- **Rationale:** Reduces gradient variance from the v-loss's implicit 1/(1-t)^2 weighting.
- **Expected upside:** 0.5-1.5 FID + faster convergence.
- **Difficulty:** ~15 lines in `denoiser.py`.
- **Files:** `JiT/denoiser.py` (forward method)
- **Validation:** Compare training loss variance and FID at epochs 40, 80, 120 with and without Min-SNR.

**Implementation sketch:**
```python
# In forward(), after computing loss_latent and loss_dino:
snr = (t / (1 - t).clamp_min(self.t_eps)) ** 2  # per-sample SNR
gamma = 5.0
snr_weight = torch.clamp(snr, max=gamma) / snr.clamp_min(1e-6)
# Apply per-sample:
loss_latent = (snr_weight * ((v_latent - v_latent_pred) ** 2).mean(dim=[1,2,3])).mean()
loss_dino = (snr_weight * ((v_dino - v_dino_pred) ** 2).mean(dim=[1,2,3])).mean()
```

#### 2D. Add stream loss weighting
- **Rationale:** Allows rebalancing gradient signal between latent and DINO streams.
- **Expected upside:** 0.5-2.0 FID.
- **Difficulty:** ~5 lines, plus CLI args.
- **Files:** `JiT/denoiser.py`, `JiT/main_jit.py`
- **Validation:** Sweep (latent_weight, dino_weight) in {(1,1), (1,0.5), (1,0.25), (2,1)} with separate W&B logging of each stream's loss.

#### 2E. Add modality type embeddings
- **Rationale:** Cheap disambiguation of latent vs DINO tokens.
- **Expected upside:** 0.3-0.8 FID.
- **Difficulty:** ~10 lines in `model_jit.py`.
- **Files:** `JiT/model_jit.py`
- **Validation:** Compare FID at 80 epochs with and without type embeddings (ablation).

#### 2F. Log per-stream losses to W&B
- **Rationale:** Critical diagnostic. Without this, stream balance and convergence rates are invisible.
- **Expected upside:** No direct FID improvement, but enables all stream-related optimizations.
- **Difficulty:** ~10 lines in `engine_jit.py` and `denoiser.py`.
- **Files:** `JiT/denoiser.py` (return per-stream losses), `JiT/engine_jit.py` (log them)
- **Validation:** Inspect W&B plots for loss_latent vs loss_dino convergence rates.

---

### Horizon 3: Research (larger architectural changes)

#### 3A. Re-extract DINO features at native 224 + bilinear upsample to 16x16
- **Rationale:** Preserves native DINO quality while matching spatial grid.
- **Expected upside:** 0.3-1.0 FID.
- **Difficulty:** Medium — modify extraction pipeline, re-extract, verify alignment.
- **Files:** `JiT/feature_extraction/dinov3.py`, `sbatch/dinov3_features.sbatch`
- **Validation:** Compare DINO feature quality (nearest-neighbor retrieval accuracy) at 224-native-upsample vs 256-direct.

#### 3B. DDT-style wide shallow prediction head
- **Rationale:** High-dimensional latent spaces benefit from wider output heads (DDT, RAE evidence).
- **Expected upside:** 1-3 FID.
- **Difficulty:** Medium-high. Add 2-layer MLP with width 2048 before final prediction layers.
- **Files:** `JiT/model_jit.py` (FinalLayer, DinoFinalLayer)
- **Validation:** Compare FID with and without the wide head at 200 epochs.

#### 3C. VF loss / flow matching loss alternatives
- **Rationale:** The current x-pred + v-loss is one of several viable parameterizations. Direct velocity prediction (as in SiT/FM) or VF loss (LightningDiT) may converge faster.
- **Expected upside:** 0.5-2.0 FID (from faster convergence).
- **Difficulty:** Medium. Modify loss computation in `denoiser.py`.
- **Files:** `JiT/denoiser.py`
- **Validation:** Train 200 epochs with VF loss vs current v-loss, compare FID curves.

#### 3D. PixelREPA representation alignment
- **Rationale:** Vanilla REPA hurts JiT (per PixelREPA paper, March 2026), but PixelREPA improves JiT-B/16 from 3.66 to 3.17.
- **Expected upside:** 0.5-1.5 FID + faster convergence.
- **Difficulty:** High. Requires masked transformer adapter + frozen DINO target.
- **Files:** New module + modifications to `JiT/denoiser.py`, `JiT/engine_jit.py`
- **Validation:** Compare FID at 200 epochs with and without PixelREPA.

---

## Top 3 Experiments to Run Next

### Experiment 1: Diagnostic Eval Suite (1 day, zero training)

Run on the existing epoch-80 checkpoint:
```bash
# 50k FID with current settings
NUM_IMAGES=50000 sbatch sbatch/jit_eval.sbatch

# Sweep CFG at 50k
for CFG in 1.5 2.0 2.5 2.9 3.5 4.0; do
  CFG=$CFG NUM_IMAGES=50000 sbatch sbatch/jit_eval.sbatch
done

# Compare EMA1 vs EMA2 (requires small code change)
```

**Purpose:** Establishes the true FID baseline and identifies the optimal inference settings. Everything else builds on this number.

### Experiment 2: Cosine LR + 200 Epochs (the real baseline)

```bash
# Add to jit_training.sbatch:
--lr_schedule cosine --min_lr 1e-5 --epochs 200 --eval_freq 20
```

With the t_eps inference fix and time shifting applied to the eval path.

**Purpose:** Tests whether the plateau is purely an LR issue. If FID keeps dropping past epoch 40 with cosine decay, the constant-LR hypothesis is confirmed and the 200-epoch result becomes the true baseline.

### Experiment 3: Kitchen-sink inference improvements on epoch-80 checkpoint

Apply ALL inference-only fixes to the existing checkpoint:
1. Direct x-prediction at final step (t_eps fix)
2. Time shifting (sweep shift in {1.5, 2.0, 3.0})
3. CFG sweep (grid search)
4. EMA2 if better than EMA1

**Purpose:** Establishes the ceiling for inference-only improvements. The delta between this and the raw eval gives the total sampler-side headroom.

---

## Single Highest-ROI Code Change

**Switch to cosine LR schedule.**

This is a 1-line change (`--lr_schedule cosine --min_lr 1e-5`) that directly addresses the primary cause of the epoch-40 plateau. Every other improvement builds on top of a properly-converging training run. Without this fix, training to 200 or 600 epochs will show diminishing returns because the optimizer is stuck oscillating at a constant LR.

---

## Single Most Important Ablation to Understand the Plateau

**Run 50k FID eval on checkpoints at epochs 20, 40, 60, 80 with both EMA1 and EMA2.**

This single experiment distinguishes between:
- **Eval noise:** If 50k FID shows steady improvement (e.g., 15, 12, 10, 9) where 5k showed a plateau, the problem is measurement noise.
- **Real plateau:** If 50k FID confirms the plateau (e.g., 14, 11, 11, 11), the optimization is genuinely stuck.
- **EMA lag:** If EMA2 (faster tracking) shows improvement where EMA1 doesn't, the model IS improving but the slow EMA masks it.

This ablation costs zero training compute and directly informs which of the other fixes to prioritize.
