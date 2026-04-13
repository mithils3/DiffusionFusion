# Implementation Guide: Priority Fixes

Exact code changes for each fix, ordered by expected ROI.

---

## Fix 1: t_eps Clamping at Inference (denoiser.py)

The core issue: `(1-t).clamp_min(self.t_eps)` in inference velocity computation causes the last 2-3 ODE steps to under-travel. The model outputs x-prediction natively, so we can use it directly for the final step.

### Option A: Direct x-prediction for final step (recommended)

In `denoiser.py`, add a new method and modify `generate()`:

```python
@torch.no_grad()
def _forward_sample_xpred(self, z_latent, z_dino, t, labels):
    """Return x-predictions with CFG applied in x-space."""
    latent_cond, dino_cond = self.net(z_latent, z_dino, t.flatten(), labels)
    latent_uncond, dino_uncond = self.net(
        z_latent, z_dino, t.flatten(), torch.full_like(labels, self.num_classes))

    low, high = self.cfg_interval
    interval_mask = (t < high) & ((low == 0) | (t > low))
    cfg = torch.where(interval_mask, self.cfg_scale, 1.0)

    latent_pred = latent_uncond + cfg * (latent_cond - latent_uncond)
    dino_pred = dino_uncond + cfg * (dino_cond - dino_uncond)
    return latent_pred, dino_pred
```

Then in `generate()`, replace:
```python
# last step euler
z_latent, z_dino = self._euler_step(
    z_latent, z_dino, timesteps[-2], timesteps[-1], labels)
```

With:
```python
# last step: use x-prediction directly to avoid t_eps undershoot
z_latent, z_dino = self._forward_sample_xpred(
    z_latent, z_dino, timesteps[-2], labels)
```

### Option B: Reduced t_eps for inference only

Add a parameter `inference_t_eps=1e-5` and use it in `_forward_sample`. More conservative but requires threading a flag through.

### Verification

Run 50k eval on the same checkpoint with and without the fix. Expected: FID improvement of 0.5-2.0 points. Also visually compare: fixed samples should have crisper fine details.

---

## Fix 2: Cosine LR Schedule (sbatch change only)

In `sbatch/jit_training.sbatch`, add flags:

```bash
--lr_schedule cosine \
--min_lr 1e-5 \
```

This uses the already-implemented cosine schedule in `lr_sched.py:12-13`:
```python
lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
    (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
```

No code changes needed. The schedule decays from 2e-4 to 1e-5 over epochs 5-200 following a half-cosine.

### Verification

Compare W&B loss curves: constant LR run vs cosine LR run. The cosine run should show continued loss decrease past epoch 40 where the constant run plateaus.

---

## Fix 3: Time Shifting at Inference (denoiser.py)

In `generate()`, after creating the linspace timesteps:

```python
timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device)
# Apply FLUX-style time shifting to concentrate steps at intermediate noise levels
shift = 2.0  # tunable hyperparameter
timesteps = timesteps / (timesteps + (1 - timesteps) * shift)
timesteps = timesteps.view(-1, *([1] * z_latent.ndim)).expand(-1, bsz, -1, -1, -1)
```

Note: the shift should be applied BEFORE the view/expand.

Expose `shift` as a CLI argument:
```python
# main_jit.py
parser.add_argument('--time_shift', default=1.0, type=float,
                    help='FLUX-style time shifting for ODE schedule (1.0 = no shift)')
```

Store in Denoiser and use in generate().

### Verification

Sweep shift in {1.0, 1.5, 2.0, 3.0} on the same checkpoint with 50k eval. Plot FID vs shift.

---

## Fix 4: Per-Stream Loss Logging (diagnostic, no FID impact)

In `denoiser.py`, change `forward()` to return a dict:

```python
def forward(self, latent, dino, labels):
    # ... existing code ...
    loss_latent = ((v_latent - v_latent_pred) ** 2).mean()
    loss_dino = ((v_dino - v_dino_pred) ** 2).mean()
    loss = loss_latent + loss_dino
    return loss, {
        'loss_latent': loss_latent.item(),
        'loss_dino': loss_dino.item(),
    }
```

In `engine_jit.py`, update the training loop:

```python
# Replace:
loss = model(latent, dino, labels)
# With:
loss, loss_components = model(latent, dino, labels)
```

And add to W&B logging:
```python
if wandb_run is not None and optimizer_step % args.log_freq == 0:
    payload.update({
        "train/loss_latent": loss_components.get('loss_latent', 0),
        "train/loss_dino": loss_components.get('loss_dino', 0),
    })
```

### Verification

Inspect W&B: if loss_dino converges much faster and to a much lower value than loss_latent, the stream imbalance hypothesis is confirmed.

---

## Fix 5: Stream Loss Weighting (denoiser.py + main_jit.py)

Add CLI args:
```python
parser.add_argument('--latent_loss_weight', default=1.0, type=float)
parser.add_argument('--dino_loss_weight', default=1.0, type=float)
```

In `Denoiser.__init__`:
```python
self.latent_loss_weight = args.latent_loss_weight
self.dino_loss_weight = args.dino_loss_weight
```

In `forward()`:
```python
loss = self.latent_loss_weight * loss_latent + self.dino_loss_weight * loss_dino
```

### Verification

Train short runs (40 epochs) with weights (1,1), (1,0.5), (1,0.25), (2,1). Compare FID and per-stream losses. The optimal ratio tells you which stream the model was underserving.

---

## Fix 6: Min-SNR-gamma (denoiser.py)

Add CLI arg:
```python
parser.add_argument('--min_snr_gamma', default=0.0, type=float,
                    help='Min-SNR-gamma clamp (0 = disabled, 5 = recommended)')
```

In `Denoiser.forward()`, replace the loss computation:

```python
# Per-sample losses (reduce over spatial/channel dims, keep batch dim)
loss_latent_per_sample = ((v_latent - v_latent_pred) ** 2).mean(dim=list(range(1, v_latent.ndim)))
loss_dino_per_sample = ((v_dino - v_dino_pred) ** 2).mean(dim=list(range(1, v_dino.ndim)))

if self.min_snr_gamma > 0:
    # SNR for flow matching: SNR(t) = t^2 / (1-t)^2
    # But we already compute v-loss which has implicit 1/(1-t)^2 weighting
    # So the effective SNR of the v-loss is proportional to t^2
    # Min-SNR clamp: weight = min(SNR, gamma) / SNR = min(1, gamma/SNR)
    t_flat = t.view(-1)
    snr = (t_flat / (1 - t_flat).clamp_min(self.t_eps)) ** 2
    snr_weight = torch.clamp(snr, max=self.min_snr_gamma) / snr.clamp_min(1e-8)
    loss_latent_per_sample = snr_weight * loss_latent_per_sample
    loss_dino_per_sample = snr_weight * loss_dino_per_sample

loss_latent = loss_latent_per_sample.mean()
loss_dino = loss_dino_per_sample.mean()
loss = loss_latent + loss_dino
```

### Verification

Train 80 epochs with `--min_snr_gamma 5` vs baseline. Compare:
1. Training loss variance (should decrease dramatically)
2. FID at epoch 40 and 80 (should improve, especially at 40)

---

## Fix 7: Posterior Mean Latents (vae.py)

In `JiT/feature_extraction/vae.py:226`, change:
```python
# Before:
x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
# After:
x = vae.encode(x).latent_dist.mean.mul_(vae.config.scaling_factor)
```

Then re-run the extraction job:
```bash
# Use a new output name to keep old latents for comparison
HF_DATASET_NAME=imagenet256_latents_mean sbatch sbatch/vae_features.sbatch
```

Update training to use new latents:
```bash
--latent_dir_name imagenet256_latents_mean
```

### Verification

Compare FID of a 40-epoch run with mean latents vs sample latents. Expected improvement: 0.5-1.5 FID.

---

## Fix 8: Modality Type Embeddings (model_jit.py)

In `JiT.__init__()`, after `self.pos_embed` (line 314-315):
```python
self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
self.dino_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
```

In `initialize_weights()`, add:
```python
nn.init.normal_(self.latent_type_embed, std=0.02)
nn.init.normal_(self.dino_type_embed, std=0.02)
```

In `forward()`, change:
```python
# Before:
latent += self.pos_embed
dino_features += self.pos_embed
# After:
latent += self.pos_embed + self.latent_type_embed
dino_features += self.pos_embed + self.dino_type_embed
```

Note: this breaks checkpoint compatibility. New training runs only.

### Verification

Compare 80-epoch FID with and without type embeddings. Expected: 0.3-0.8 improvement. Should be run together with other training changes (cosine LR, Min-SNR) to amortize training cost.

---

## Priority Execution Order

| # | Fix | Type | Cost | Needs retraining? |
|---|---|---|---|---|
| 1 | 50k FID eval on existing checkpoints | Diagnostic | Zero | No |
| 2 | t_eps inference fix (x-pred final step) | Inference | 5 lines | No |
| 3 | Time shifting | Inference | 5 lines | No |
| 4 | CFG + EMA sweep | Inference | Shell script | No |
| 5 | Cosine LR schedule | Training | 1 sbatch flag | Yes |
| 6 | Per-stream loss logging | Diagnostic | 10 lines | Yes (to see logs) |
| 7 | Posterior mean latents | Data | 1 line + re-extract | Yes |
| 8 | Min-SNR-gamma | Training | 15 lines | Yes |
| 9 | Stream loss weights | Training | 5 lines + sweeps | Yes |
| 10 | Modality type embeddings | Architecture | 10 lines | Yes |

Fixes 1-4 can be done immediately on the current checkpoint. Fix 5 is the first thing to change for the next training run. Fixes 6-10 should be combined into a single well-instrumented training run.
