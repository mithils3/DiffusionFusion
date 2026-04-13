# Decoder FID Improvement Plan

## Current Gap

| Metric | JiT Decoder | SDXL-VAE | Gap |
|---|---|---|---|
| FID (50k ImageNet val) | 1.218 | 0.690 | 0.528 |
| Recon MSE | 0.0673 | — | — |
| Checkpoint | epoch 11 / 16 | — | — |

## Important Framing

- `JiT/eval/eval_decoder.py` measures decoder-only autoencoding from clean saved latents + clean DINO features. This is NOT end-to-end JiT generation quality.
- Noise augmentation (`noise_tau = 0.2`) is kept during training to preserve downstream robustness. Evaluation feeds clean features, so the noise penalty is real but accepted.
- The decoder receives 49x more input information than SDXL-VAE (200k values vs 4k). The FID gap is not a capacity problem — it is architecture, training length, and loss alignment.
- Goal: get close to SDXL-VAE FID (~0.70-0.80) while preserving noise robustness. NOT optimizing the clean-decoder benchmark at all costs.

## What the Code Is Actually Doing

Architecture (`model.py`):
- 12-layer transformer, hidden_size=1152, 16 heads, 257M parameters.
- 256 learnable query tokens, each producing one 16x16 output patch via `Linear(1152, 768)`.
- Cross-attention context: `[DINO(256 tokens), latent(256 tokens)]` = 512 tokens.
- Both streams receive the SAME shared `pos_embed` — no modality embedding.
- Context tokens are never normalized before cross-attention K,V.
- Dropout applied only to middle 50% of layers.

Loss (`losses.py`, `train.py`):
- Base reconstruction: **L1** (not MSE). MSE is logged but not optimized.
- Total = `L1 + 1.0 * LPIPS(VGG) + disc_weight * adaptive_weight * adversarial_scale * adversarial_loss`.
- L1 computed in DINO-normalized image space (not pixel space).
- LPIPS and discriminator convert to pixel-like space internally.

GAN schedule (`default_config.yaml`):
- Epochs 0-3: L1 + LPIPS only.
- Epochs 4-5: discriminator trains, no adversarial signal to decoder.
- Epochs 6-7: adversarial signal ramps in (warmup over 2 epochs).
- Epochs 8-15: full adversarial strength.
- The epoch 11 checkpoint had only ~3 epochs of full adversarial training.

Noise augmentation (`gan.py:39-56`):
- Same sigma drawn from `|N(0, 0.2)|` applied to both DINO and latent streams.
- DINO features (768-dim, layer-normed to ~unit variance): sigma=0.16 avg is ~16% relative noise.
- SDXL latents (4-dim, arbitrary scale): same absolute sigma has a different relative impact.
- Streams are corrupted asymmetrically by design accident.

DINO normalization:
- `normalize_dino_feature_map_tokens` called at extraction time (`dinov3.py:159`).
- Called AGAIN at batch load time (`dataset.py:513`).
- Second pass is near-identity on already-normalized float16 data. Matches RAE recipe to normalize once, not twice.

Optimizer:
- Adam (not AdamW) with `betas=(0.5, 0.9)`, lr=2e-4, weight_decay=0.
- Cosine schedule with 1 epoch warmup, decay to 2e-5.
- EMA decay 0.9978. EMA model used for evaluation.

## What the RAE Paper Implies

### Transfers directly
- L1 + LPIPS + GAN is the correct loss family. Confirmed.
- Noise augmentation helps generation (gFID 4.81->4.28) at cost of clean rFID (0.49->0.57). ~0.08 rFID penalty in their setup.
- Per-token channel normalization of encoder features. We do this (but twice).
- Frozen DINO discriminator. We don't freeze ours — worth testing.
- Decoder capacity scaling matters (ViT-B 0.58 -> ViT-L 0.50 -> ViT-XL 0.49). But our 257M decoder with 49x input information should already exceed RAE ViT-L.

### Does NOT transfer
- RAE decodes ONE stream. We decode TWO. Modality fusion is strictly harder.
- RAE's decoder inverts a frozen encoder representation. Ours must fuse two encoders not designed for this decoder.
- RAE's noise level was tuned for single-stream. Our two-stream setup needs stream-aware noise.

### Extra important for our two-stream setup
- Modality disambiguation is critical — RAE never needs it.
- Information redundancy between DINO and SDXL latent must be handled by cross-attention specialization.
- Noise augmentation should respect stream-specific signal-to-noise ratios.

## Top 5 Bottlenecks

### 1. Insufficient adversarial training time
The epoch 11 checkpoint had ~3 epochs of full adversarial. VQGAN-style decoders converge over many more steps. Doubling epochs roughly doubles the full-adversarial window from 8 to 24 epochs.

- **Targets:** clean FID + robustness
- **Expected impact:** high
- **Risk:** low
- **Cost:** wall time only (zero code change)
- **Touchpoints:** `sbatch/decoder_training.sbatch:29` (EPOCHS)

### 2. No modality disambiguation in context stream
Both DINO and latent tokens get identical `pos_embed`. Cross-attention must infer stream identity from content statistics alone, wasting capacity.

- **Targets:** clean FID + robustness
- **Expected impact:** high
- **Risk:** low
- **Cost:** low (~15 lines in `model.py`)
- **Touchpoints:** `model.py:34-42,99,142,156`

### 3. Asymmetric noise corruption
Same absolute sigma applied to streams with different signal scales. One stream is systematically over-corrupted relative to the other.

- **Targets:** both
- **Expected impact:** medium
- **Risk:** low
- **Cost:** low (~10 lines in `gan.py`)
- **Touchpoints:** `gan.py:39-56`

### 4. Double DINO normalization
Redundant `normalize_dino_feature_map_tokens` at load time. Near-identity transform on float16 data that deviates from RAE recipe.

- **Targets:** clean FID
- **Expected impact:** low-medium
- **Risk:** very low
- **Cost:** zero (delete one line)
- **Touchpoints:** `dataset.py:513`

### 5. No context normalization before cross-attention K,V
Raw mixed-scale context tokens (DINO ~unit variance, latent ~arbitrary scale) passed as K,V without normalization. Only queries are normalized via `norm1`.

- **Targets:** clean FID
- **Expected impact:** medium
- **Risk:** low
- **Cost:** low (~5 lines in `model.py`)
- **Touchpoints:** `model.py:162`, add `RMSNorm` on context

## Likely No-Ops and Low-Value Distractions

1. **Charbonnier loss instead of L1:** Expected impact ~0.01-0.02 FID. Noise relative to the 0.53 gap.
2. **Scaling to depth=24 before fixing architecture:** Trains a bigger misconfigured model. Not capacity-limited.
3. **Mixed clean+noisy batch training:** Implementation complexity for uncertain gain. Tuning tau gives 90% of the benefit.
4. **Overlapping patch decoding:** Expensive, uncertain benefit. Self-attention between queries already provides inter-patch coherence.
5. **Optimizer beta1 sweep (0.5 vs 0.9):** beta1=0.5 is standard for VQGAN-style training. Low-confidence improvement.
6. **Multi-scale discriminator:** Premature before generator-side fixes land.

## Prioritized Experiment Table

| Priority | Change | Targets | Upside | Risk | Cost | Success if... |
|---|---|---|---|---|---|---|
| 1 | Train 32 epochs | FID + robustness | High | Low | Wall time only | FID < 1.0 at epoch 32 |
| 2 | Remove double DINO norm | Clean FID | Low-Med | Very low | Delete 1 line | Any measurable FID change |
| 3 | Add modality type embeddings | FID + robustness | High | Low | 15 lines | FID improves > 0.05 |
| 4 | Add context RMSNorm | Clean FID | Medium | Low | 5 lines | FID improves > 0.02 |
| 5 | Stream-specific noise tau | Both | Medium | Low | 10 lines | Clean FID improves without hurting noisy-input recon |
| 6 | Freeze disc backbone | Clean FID (texture) | Medium | Medium | Config change | Sharper textures, stable disc_loss |
| 7 | Earlier disc start (2/4) | Clean FID | Medium | Medium | Config change | FID < 1.0 at epoch 16 |
| 8 | Pixel-space L1 | Clean FID | Low-Med | Low | 5 lines | FID improves > 0.03 |
| 9 | Conv refinement head | Clean FID | Med-High | Medium | 50 lines | FID < 0.85 |
| 10 | noise_tau sweep | Both | Medium | Low | Config change | Map clean-FID vs tau Pareto curve |

## Best 3 Runs to Launch First

### Run A: Extended baseline
```bash
EPOCHS=32 EVAL_FREQ=4 WANDB_RUN_NAME="decoder-32ep-baseline" \
sbatch sbatch/decoder_training.sbatch
```
Diagnostic: is training length the bottleneck?
Expected: FID 0.95-1.10 at epoch 32.

### Run B: Architecture fixes + extended training
Code changes:
1. `dataset.py:513` — remove second DINO normalization
2. `model.py` — add `dino_type_embed`, `latent_type_embed` (learned type embeddings)
3. `model.py` — add `ctx_norm = RMSNorm(hidden_size)` on concatenated context

```bash
EPOCHS=32 EVAL_FREQ=4 WANDB_RUN_NAME="decoder-32ep-arch-fixes" \
sbatch sbatch/decoder_training.sbatch
```
Diagnostic: do modality + norm fixes matter?
Expected: FID 0.85-1.00 at epoch 32.

### Run C: Diagnostic ceiling (near-clean noise)
```bash
EPOCHS=16 NOISE_TAU=0.05 EVAL_FREQ=2 WANDB_RUN_NAME="decoder-16ep-tau005-ceiling" \
sbatch sbatch/decoder_training.sbatch
```
Diagnostic: what's the best clean-FID achievable with current architecture?
Expected: FID 0.85-1.05. Gap between this and tau=0.2 run = noise penalty.
**Do NOT ship this decoder for downstream diffusion.**

### Decision matrix

| Run A result | Run C result | Diagnosis | Next step |
|---|---|---|---|
| FID drops a lot (< 1.0) | — | Training length was main issue | Run A recipe + arch fixes at 32-48ep |
| FID plateaus (~1.2) | FID drops (< 0.9) | Noise penalty is dominant | Stream-specific noise, careful tau tuning |
| FID plateaus (~1.2) | Also plateaus (~1.1+) | Architecture is bottleneck | Modality embeddings, context norm, conv head |
| Run B much better than A | — | Architecture fixes are high-value | Stack all arch fixes into production recipe |

## Exact Code Changes for Architecture Fixes (Run B)

### Fix 1: Remove double DINO normalization
File: `JiT/decoder/dataset.py`, line 513

```python
# Before:
dino = normalize_dino_feature_map_tokens(torch.from_numpy(rows["dino"]))

# After:
dino = torch.from_numpy(rows["dino"])
```

Verify: log DINO feature mean/std before and after to confirm features are already normalized.

### Fix 2: Add modality type embeddings
File: `JiT/decoder/model.py`

In `__init__`, after `self.query_pos_embed` (line 40):
```python
self.dino_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
```

In `init_weights`, after `nn.init.normal_(self.query_pos_embed, std=0.02)` (line 99):
```python
nn.init.normal_(self.dino_type_embed, std=0.02)
nn.init.normal_(self.latent_type_embed, std=0.02)
```

In `_prepare_dino_tokens`, change line 142:
```python
return self.dino_embedder(dino) + self.pos_embed + self.dino_type_embed
```

In `forward`, change line 156:
```python
latent_tokens = latent_tokens + self.pos_embed + self.latent_type_embed
```

### Fix 3: Add context normalization
File: `JiT/decoder/model.py`

In `__init__`, after `self.blocks` (line 54):
```python
self.ctx_norm = RMSNorm(hidden_size, eps=1e-6)
```

In `forward`, change line 162:
```python
ctx_tokens = self.ctx_norm(torch.cat([dino, latent_tokens], dim=1))
```

### Fix 4 (future): Stream-specific noise
File: `JiT/decoder/gan.py`, lines 39-56

Replace shared sigma with per-stream scaling:
```python
def apply_noise_augmentation(latent, dino, noise_tau, dino_noise_scale=1.0):
    if noise_tau <= 0.0:
        return latent, dino
    batch_size = latent.shape[0]
    base_sigma = latent.new_empty(batch_size).normal_(mean=0.0, std=noise_tau).abs_()

    latent_sigma = base_sigma.view(batch_size, *([1] * (latent.ndim - 1)))
    dino_sigma = (base_sigma * dino_noise_scale).to(device=dino.device, dtype=dino.dtype)
    dino_sigma = dino_sigma.view(batch_size, *([1] * (dino.ndim - 1)))

    latent = latent + latent_sigma * torch.randn_like(latent)
    dino = dino + dino_sigma * torch.randn_like(dino)
    return latent, dino
```

Sweep `dino_noise_scale` in [0.5, 1.0, 2.0] to find the right ratio.

## What Success Looks Like

| Milestone | FID target | How to get there |
|---|---|---|
| Baseline after full training | < 1.0 | Run A (32 epochs, no code changes) |
| Architecture fixes | < 0.90 | Run B (modality embed + ctx norm + single norm) |
| Optimized recipe | < 0.80 | Best of above + stream-specific noise + frozen disc + earlier disc start |
| Near SDXL-VAE | < 0.75 | Above + conv refinement head or capacity scaling |

Track TWO metrics throughout:
1. Clean decoder reconstruction FID (this benchmark)
2. Noisy-input reconstruction quality (sigma=0.2) to verify robustness is preserved

## Changes NOT to Prioritize Yet

- Capacity scaling (bigger model) — not capacity-limited at 257M with 49x input information.
- Charbonnier loss — ~0.01 FID expected impact.
- Mixed clean+noisy batches — implementation complexity for uncertain gain.
- Overlapping patch decoding — expensive, uncertain benefit.
- Optimizer beta1 sweep — low confidence, risky interaction with GAN stability.
- Multi-scale discriminator — premature before generator fixes land.
