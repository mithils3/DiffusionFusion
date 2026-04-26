# V-Co-guided improvement plan for JiT dual noising

Date: 2026-04-25

Goal: use V-Co as the main reference for improving the current `JiT-Dual-B/2-4C-896` run, which is around 8 FID after 80 epochs. The useful framing from V-Co is that co-denoising quality depends on four separable choices:

1. Architecture
2. CFG / unconditional branch definition
3. Auxiliary semantic loss
4. Cross-stream calibration

This repo already has a latent+DINO dual noising system, so the plan is not a ground-up rewrite. The priority is to make our existing setup closer to the parts of V-Co that had the largest controlled ablation gains.

## Current repo baseline

- Model: `JiT-Dual-B/2-4C-896`
- Training budget: 80 epochs
- Effective batch size: `128 * accum_iter 2 * 4 GPUs = 1024`
- LR: `blr=5e-5`, expected actual LR `2e-4`
- Sampling/eval: Heun, 50 steps, CFG `2.9`, interval `[0.1, 1.0]`
- Current architecture: separate latent and DINO towers, in-context tokens at block 0, bidirectional cross-fusion at blocks 4 and 8.
- Current noising: latent time `t = sigmoid(N(P_mean=-0.8, P_std=0.8))`; DINO time is shifted by `log(sqrt(dino_dim / latent_dim))`.
- Current loss: equal latent velocity MSE and DINO velocity MSE by default.
- Current CFG: unconditional pass replaces class label with the null class, but still passes the same noised DINO stream through the model.

Important caveat: V-Co is pixel+DINO. This repo is SD-VAE latent+DINO+decoder. So copy the principles, not the exact numbers.

## V-Co match map

### What already matches

- Joint denoising of an image-side stream and DINO features.
- Clean-target prediction converted into velocity loss.
- Separate latent/DINO processing paths.
- Separate DINO time conditioning, which is related to V-Co's SNR schedule-shift view.
- A decoder good enough that the denoiser/sampler is probably the main bottleneck.

### What does not match yet

- V-Co's best model lets streams interact every block through joint attention with separate norm/MLP/QKV. Our model has local towers and periodic cross-attention.
- V-Co defines CFG's unconditional branch structurally by masking semantic-to-pixel attention. Our unconditional branch still lets the latent stream read DINO.
- V-Co's semantic loss weight is small, around `lambda_d = 0.1`. Our default DINO loss weight is `1.0`.
- V-Co calibrates by empirical RMS feature scaling. Our current DINO shift is dimension-based and may not match actual latent/DINO signal magnitudes.
- V-Co's auxiliary gain comes from perceptual/drifting losses on predicted images; our latent+DINO setup needs an adapted version because image decoding during training is expensive.

## Priority 1: structural CFG masking

This is the highest-leverage V-Co transplant.

### Why

CFG uses:

`guidance = pred_cond - pred_uncond`

For this to mean "add class/semantic conditioning", `pred_uncond` must actually be unconditional for the image-side stream.

Right now:

- `latent_cond` sees class label + DINO.
- `latent_uncond` sees null class + DINO.
- The latent stream can read DINO through `latent_from_dino` cross-fusion in both passes.

So `latent_uncond` is still semantically conditioned. The CFG delta mostly captures class-label change, while DINO semantics are partially canceled or muddled. V-Co's result says this is a bad unconditional branch for co-denoising.

V-Co's controlled CFG ablation on ImageNet-256 with JiT-B/16 at 200 epochs:

- Input-dropout unconditional baselines were around `FID 6.69` guided.
- Independent dropout plus semantic-to-pixel structural masking reached `FID 3.59` guided.
- Joint class+DINO dropout plus semantic-to-pixel structural masking reached `FID 3.18` guided.
- Joint dropout with bidirectional masking was worse at `FID 5.66` guided.

So the best structural masking setup improved guided FID by about `3.51` absolute over the input-dropout baseline, roughly a `52%` relative reduction from `6.69` to `3.18`. The one-way mask mattered: blocking only semantic-to-pixel was much better than blocking both directions.

For our repo, the equivalent of V-Co's semantic-to-pixel mask is DINO-to-latent masking:

- During unconditional prediction, disable `latent_from_dino`.
- Keep `dino_from_latent` active.

Keeping latent-to-DINO flow matters because the DINO stream can still stay aligned with the image-side trajectory, while the image-side latent prediction is prevented from reading semantic DINO information.

### Implementation

Add a structural mask path:

- Add `mask_dino_to_latent: bool = False` to `CrossFusionBlock.forward`.
- If true, skip `self.latent_from_dino(...)` and leave `latent = latent_snapshot`.
- Keep `self.dino_from_latent(...)` active.
- Thread the flag through `JiTDualStream.forward`.
- Thread the flag through `Denoiser._net_forward`.
- In `_forward_sample_xpred`, call the unconditional network pass with `mask_dino_to_latent=True`.

Training should match inference:

- When labels are dropped for CFG training, jointly activate the DINO-to-latent mask.
- First implementation can be batch-level dropout for simplicity.
- Better implementation can support per-sample masks by running conditional and masked subsets separately and stitching outputs back.

Retraining note:

- Inference-only masking on an existing checkpoint is worth a quick test because it is cheap.
- But the real V-Co result requires training with the same structural unconditional branch. If the model was only trained with label dropout, the masked unconditional path is out-of-distribution at sampling time.
- Best practical path: implement inference masking first for a sanity sweep, then run either a finetune or a fresh short pilot with joint label/mask dropout.

### Pilot experiment

Train a short run with:

- Structural unconditional masking enabled.
- Joint class/mask dropout probability `0.1`.
- Same schedule and loss weights as current baseline.

Evaluate:

- 10K FID for fast signal.
- Then 50K FID for the best setting.
- CFG sweep: `1.5, 2.0, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5, 5.0`.
- Interval min sweep: `0.05, 0.1, 0.15, 0.2`.

Expected result: cleaner CFG direction, better guided FID, and less dependence on very high CFG.

## Priority 2: DINO loss weight sweep

V-Co found the semantic stream helps most when it is secondary to the image/pixel stream. Their best `lambda_d` region is around `0.01` to `0.1`, with `0.1` used in the final recipe.

Our current default is:

- `latent_loss_weight = 1.0`
- `dino_loss_weight = 1.0`

Because each loss is averaged independently, this gives the DINO task equal aggregate weight, not a small auxiliary role.

### Sweep

Keep `latent_loss_weight=1.0`, test:

- `dino_loss_weight=0.05`
- `dino_loss_weight=0.1`
- `dino_loss_weight=0.25`
- `dino_loss_weight=0.5`
- `dino_loss_weight=1.0`

Run these after structural CFG masking is in place, because the best loss balance may change once CFG is fixed.

Expected result: lower DINO weights may improve image latent quality and guided FID, even if DINO reconstruction metrics worsen slightly.

## Priority 3: empirical RMS calibration

V-Co's calibration result is very relevant. They show that pixels and DINO features need matched signal magnitude, otherwise the same timestep creates mismatched SNR.

Current repo uses a dimension-ratio DINO time shift:

`alpha = sqrt(dino_dim / latent_dim)`

This is principled for high-dimensional representation latents, but it does not verify actual signal RMS. In this repo:

- DINO features are per-token layer-normalized during extraction.
- SD-VAE latents are not clearly dataset-normalized beyond VAE scaling.
- The decoder may expect a particular latent scale.

So we should measure before guessing.

### Measurement

Add a stats pass over training shards:

- SD-VAE latent global RMS.
- SD-VAE latent per-channel mean/std/RMS.
- DINO global RMS after current feature preprocessing.
- DINO per-channel/token RMS summary.

Compute:

`alpha_rms = rms_latent / rms_dino`

### Experiments

Compare:

- Current dimension-based DINO time shift.
- No DINO time shift, but DINO features scaled by `alpha_rms`.
- DINO time shift derived from empirical RMS instead of dimension ratio.
- Combined latent+DINO effective shift only if the above two do not help.

Expected result: better schedule calibration and less branch imbalance. This is probably lower immediate impact than CFG masking, but it derisks every longer run.

## Priority 4: architecture closer to V-Co

V-Co's architectural lesson is not "just add another stream"; it is:

- Keep feature-specific computation.
- Let streams interact flexibly.
- Avoid forcing everything through a mostly shared backbone.

Our model keeps feature-specific computation, but interaction is sparse: cross-fusion happens periodically rather than every block.

### Low-risk architecture pilots

Expose model variants:

- `cross_start=0, cross_every=2`
- `cross_start=0, cross_every=1`
- optionally unidirectional ablation where DINO-to-latent is available only in conditional paths and latent-to-DINO is always available.

Memory permitting, `cross_every=1` is the closest current-code approximation to V-Co's every-block interaction.

### Larger architecture change

If sparse cross-fusion remains a bottleneck, implement a V-Co-style block:

- Separate latent norm/MLP/QKV.
- Separate DINO norm/MLP/QKV.
- Joint attention over both streams every block.
- Attention mask support for DINO-to-latent during unconditional passes.

This is the clean architectural target, but it is a bigger implementation than structural CFG masking.

Expected result: better semantic transfer across the whole denoising path. This probably requires a full retrain to judge.

## Priority 5: auxiliary semantic objective

V-Co found:

- REPA on hidden states gives little once co-denoising already exists.
- Perceptual DINO loss on predicted images helps more.
- Perceptual + drifting hybrid helps best.

For this repo, direct image-space perceptual loss requires decoding predicted latents during training, which is expensive and may complicate gradients through the decoder.

### Practical staged version

Start simple:

- Do not add plain REPA as the first bet.
- First try a DINO-space consistency loss on the predicted DINO clean output, if it is not redundant with the v-loss.
- Then try a batch-level same-class repulsion term on predicted DINO features, adapted from V-Co's drifting loss.

Hybrid sketch:

- Positive field: pull predicted DINO clean feature toward the target DINO feature.
- Negative field: repel predicted DINO clean feature from same-class generated/predicted neighbors in the batch.
- Gate: use similarity to the target DINO feature, so repulsion dominates when far and attraction dominates when close.

This is cheaper than image decoding and follows the spirit of V-Co, but it is not identical to their image perceptual loss.

Expected result: possible diversity/precision improvement after CFG and calibration are fixed. Do this later because bad CFG can hide the effect of auxiliary losses.

## Evaluation discipline

Do not compare stale 10K FID against paper-style 50K FID.

For each serious candidate:

- Run quick 10K FID for direction.
- Confirm best candidates at 50K FID.
- Sweep CFG and CFG interval.
- Track both EMA settings if checkpoints support it.
- Record exact checkpoint, epoch, CFG, interval, steps, EMA, and sample count.

Default sweep:

- CFG: `1.5, 2.0, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5, 5.0`
- Interval min: `0.05, 0.10, 0.15, 0.20`
- Interval max: `1.0`
- Steps: Heun 50 first; test 75/100 only for the top two configurations.

## Ordered implementation queue

1. Implement structural DINO-to-latent masking for unconditional CFG.
2. Add joint label/mask dropout during training.
3. Run short pilot and CFG sweep against the current baseline.
4. Sweep `dino_loss_weight` around V-Co's `lambda_d` regime.
5. Add latent/DINO RMS stats and test empirical calibration.
6. Test denser cross-fusion variants.
7. Only after the above, adapt V-Co's perceptual-drifting hybrid loss in DINO space.

## Best bet

The fastest path down from 8 FID is probably not more parameters or plain REPA. It is:

1. Make CFG structurally unconditional by masking DINO-to-latent in the unconditional branch.
2. Train with the same joint mask/dropout behavior used at inference.
3. Reduce the DINO loss weight toward V-Co's semantic auxiliary regime.
4. Calibrate latent and DINO signal magnitudes empirically.

This is the closest clean extension of V-Co's controlled findings to our current JiT dual noising code.
