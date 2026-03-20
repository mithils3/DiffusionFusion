# Decoder Blur Analysis And Improvement Plan

Date: 2026-03-19

This note combines:

- what the current decoder in this repo is doing,
- what the target-vs-decoder sample pair shows visually,
- and what recent primary-source research suggests for improving sharpness.

## Short Answer

The blur is mostly a high-frequency reconstruction problem, not a semantics problem.

From the sample pair you shared, the decoder preserves:

- scene layout,
- object placement,
- coarse color and lighting,
- and overall depth structure.

What it loses is:

- crisp object boundaries,
- thin structures like the string and chime edges,
- small contrast transitions,
- and the local sharpness of background highlights and texture.

That pattern is exactly what you expect when:

1. the decoder gets the low-frequency structure right,
2. the output head is spatially coarse,
3. the adversarial/perceptual pressure is not strong enough or not local enough,
4. and the model has no explicit mechanism dedicated to recovering high-frequency detail.

## What The Current Decoder Is

The current decoder is a transformer that fuses:

- SDXL-VAE latents: `(B, 4, 32, 32)`
- DINO features: `(B, 768, 16, 16)`

and reconstructs a `256 x 256` RGB image.

Important implementation details from the repo:

- `JiT/decoder/model.py` uses `latent_patch_size=2`, so the VAE latent is reduced from `32x32` to `16x16` tokens.
- The decoder uses exactly `256` learnable query tokens and each query predicts one `16x16` RGB patch through a single linear head.
- There is no convolutional refinement stage after the transformer output.
- The default model size is already large: about `257.5M` trainable parameters.
- The training loss is `L1 + LPIPS + GAN`, not MSE as some of the documentation still implies.
- The discriminator is a DINOv3 ViT-S/16 feature discriminator with a shallow conv head.
- Noise augmentation is applied to both latent streams during decoder training.

## What The Visual Comparison Suggests

Looking at the target versus decoder output:

- The decoder understands the scene correctly. This is not a "wrong content" failure.
- Blur is strongest on thin, high-contrast, and edge-heavy structures.
- Background bokeh and local microcontrast are softened.
- The output looks like a slightly averaged version of the target rather than a broken or artifact-heavy reconstruction.

That usually means the model is learning a good coarse inverse mapping, but the training signal for local sharpness is still too weak or too coarse.

## Why The Current Decoder Is Blurry

## 1. The output head is very coarse

Right now, one query token predicts one full `16x16` RGB patch via a single linear projection.

Why this matters:

- a `16x16` patch is a lot of pixels to predict from one token,
- there is no overlapping prediction,
- there is no local convolutional cleanup pass,
- and the model has to get both structure and texture right in one shot.

This is efficient, but it is not the sharpest possible design. It tends to favor smoothness over crisp micro-detail.

This is the single most likely architectural reason for the softness you are seeing.

## 2. The discriminator is likely too semantic and too coarse for subtle blur

Your discriminator uses DINOv3 ViT-S/16 features and then predicts real/fake from that feature map.

That is a smart discriminator for semantic realism, but it is not necessarily the best one for punishing subtle blur because:

- ViT-S/16 already works at a `16x16` patch granularity,
- DINO features are intentionally somewhat invariant to tiny pixel-level changes,
- and a blur that preserves scene semantics can still look "acceptable" in feature space.

So the discriminator can learn "this is the right object in the right place" without strongly penalizing "the edge is too soft."

## 3. The actual training schedule leaves little time for adversarial sharpening

The default config is already short:

- `16` epochs in `JiT/decoder/default_config.yaml`
- discriminator updates start at epoch `4`
- adversarial gradients to the decoder start at epoch `6`
- adversarial warmup lasts `2` epochs

But the sbatch job currently overrides this to only `12` epochs in `sbatch/decoder_training.sbatch`.

So in the run you are actually launching, the decoder only gets:

- epochs `0-3`: no discriminator updates,
- epochs `4-5`: discriminator trains, but generator does not get GAN gradients,
- epochs `6-7`: adversarial signal ramps up,
- epochs `8-11`: only four fully active adversarial epochs.

That is very little sharpening time for a 257M-parameter image decoder.

## 4. There is no explicit high-frequency reconstruction loss

The current loss is:

- L1 reconstruction,
- LPIPS,
- adversarial loss.

That is a solid baseline, but there is no explicit loss on:

- image gradients,
- Laplacian bands,
- wavelet high-frequency bands,
- or Fourier/high-pass components.

If the model can get a good L1/LPIPS score with slightly softened edges, it often will.

## 5. Noise augmentation helps diffusion robustness but can slightly hurt pure reconstruction sharpness

The decoder injects Gaussian noise into both latent streams during training.

That is useful if the decoder will later consume imperfect denoised latents from the generative model. But it can also make clean reconstructions a bit softer if the noise level is too high or always-on.

This does not mean the idea is wrong. It means `noise_tau` needs to be treated as a tradeoff:

- more robustness to noisy generative latents,
- versus less perfectly sharp clean reconstructions.

## What Recent Research Says

## 1. Perceptual + adversarial supervision is still the standard answer to blur

VQGAN showed very clearly that strong compression models need perceptual and adversarial losses to avoid overly smooth reconstructions. The core message is still right: pure pixel losses wash out local realism, while patch-based adversarial supervision restores it.

LDM reused this idea in its autoencoder stage and explicitly argued for mild compression because stronger compression leaves too much detail recovery to later stages.

Takeaway for this repo:

- you are already on the right track conceptually,
- but the current sharpening signal probably is not strong enough and not local enough.

## 2. Recent autoencoder work says decoder design matters as much as encoder design

LiteVAE is directly relevant here. It reports that:

- decoder architecture changes improve reconstruction quality,
- high-frequency losses based on Gaussian filtering and wavelets consistently improve reconstruction metrics,
- and a UNet-based discriminator outperforms PatchGAN and StyleGAN discriminators in rFID while being more stable.

This is highly aligned with what your sample pair shows: the missing information is mostly in the high-frequency band.

## 3. Recent RAE work says bigger decoders help, but decoder robustness also matters

The 2025 RAE paper is especially relevant because your decoder is conceptually RAE-like.

Two takeaways matter most here:

- larger decoders improve reconstruction quality,
- and noise-augmented decoder training is important when the decoder will later see diffusion-generated latents instead of clean encoder outputs.

That supports two conclusions:

- decoder capacity can help,
- but in your repo, capacity is probably not the first bottleneck because the model is already large,
- and noise augmentation should be tuned for the exact clean-reconstruction-versus-generation tradeoff you want.

## 4. Two-stage refinement remains a strong option

Both DALL-E 2 and SDXL use an explicit refinement idea:

- DALL-E 2 uses a diffusion decoder conditioned on image latents,
- SDXL adds a refiner to improve visual fidelity after base generation.

That is a heavier solution, but it is still one of the strongest options if your goal is maximum sharpness rather than decoder simplicity.

## Prioritized Improvements

## Priority 1: Fix the training recipe before changing the whole architecture

These are the highest-return changes for this repo.

### A. Train longer

Do this first.

Recommendation:

- stop overriding the run to `12` epochs,
- go back to at least `16`,
- and realistically try `20-24` if compute allows.

Reason:

- with the current schedule, the decoder only gets a few fully active adversarial epochs.

Expected effect:

- sharper local detail,
- stronger contrast on thin structures,
- better LPIPS/rFID.

### B. Add explicit high-frequency loss

Add one of:

- Laplacian pyramid L1 loss,
- gradient loss on Sobel/Scharr edges,
- Gaussian high-pass loss,
- or wavelet-band reconstruction loss on LH/HL/HH components.

If you want the most research-aligned choice, use a wavelet or Gaussian-filter high-frequency loss.

Why:

- your current losses do not explicitly tell the model that recovering edge energy matters.

Expected effect:

- better sharpness on strings, feathers, chime edges, foliage boundaries, and bokeh contours.

### C. Tune `noise_tau` instead of treating it as fixed

Try a sweep like:

- `0.0`
- `0.05`
- `0.1`
- `0.2`

Also consider a mixed strategy:

- half the batches clean,
- half noise-augmented.

Or a schedule:

- start near `0`,
- increase later in training once clean reconstruction is already learned.

Why:

- if your current comparison is on clean reconstructions, always-on noise can cost sharpness.
- if your true use case is diffusion outputs, some noise is still useful.

### D. Add a sharper discriminator

Best next step:

- keep the current DINO discriminator for semantic realism,
- add a second pixel-space discriminator for local detail.

Strong options:

- UNet discriminator,
- or a multi-scale PatchGAN.

Why:

- the current discriminator is likely too coarse and too invariant to subtle blur.

Expected effect:

- more pressure on local edges and texture,
- less "soft-but-correct" output.

## Priority 2: Improve the decoder head

### E. Add a convolutional refinement tail after the transformer

This is the architectural change I would try first.

Idea:

1. keep the transformer as the coarse fusion module,
2. reconstruct a coarse image or feature map,
3. run a shallow residual conv stack or small UNet refinement head on top.

Why:

- transformers are good at global coordination,
- convolutions are good at local detail cleanup.

This is the clearest way to add a locality bias without throwing away the current decoder.

Expected effect:

- sharper edges,
- less averaging inside each `16x16` predicted patch,
- more visually crisp outputs.

### F. Decouple output query resolution from input token resolution

Right now, `256` context-aligned queries produce a `16x16` output patch grid.

A stronger design is:

- use a finer output query grid, e.g. `32x32`,
- and predict smaller patches, e.g. `8x8`,
- while still attending to the same latent context.

This requires code changes because query count is currently tied to the latent token count.

Why:

- it reduces how much image content each query must synthesize at once,
- which usually helps local sharpness and boundary quality.

If that is too expensive, the conv refinement tail above is the cheaper compromise.

## Priority 3: Improve fusion and decoder supervision

### G. Add modality/type embeddings and separate positional embeddings

At minimum, add:

- `dino_type_embed`
- `vae_type_embed`

Preferably also use:

- `dino_pos_embed`
- `latent_pos_embed`

instead of sharing the same `pos_embed`.

Why:

- DINO carries semantic structure,
- VAE carries local appearance/detail,
- and the decoder should not have to infer the source stream only from token statistics.

This is more of a quality and robustness improvement than a silver bullet for blur, but it is worth doing.

### H. Revisit adaptive adversarial weighting

The current code uses an adaptive GAN weight based on last-layer gradient norms.

That can be helpful, but in practice it can also make the adversarial term too timid exactly when the model most needs sharpening pressure.

I would run an ablation:

- adaptive weight on,
- adaptive weight off with a tuned constant.

Given the current softness, this is worth testing.

## Priority 4: Bigger changes if you want maximum fidelity

### I. Use a stronger decoder only after fixing the objective

Recent RAE work supports larger decoders, but your current decoder is already large.

So I would not make "just scale the transformer" the first move.

Order matters:

1. fix losses and discriminator,
2. add local refinement,
3. then scale the decoder if it is still detail-limited.

### J. Add a second-stage refiner

If absolute image quality matters more than simplicity:

- coarse transformer decoder first,
- then a refiner model conditioned on the coarse reconstruction and/or latents.

This is the most expensive option, but it is still the strongest path if you want the last bit of crispness.

## What I Would Do First In This Repo

If I had to choose the first five experiments, I would run these in order:

1. Train the existing model for `20-24` epochs instead of `12`.
2. Sweep `noise_tau` over `0.0, 0.05, 0.1, 0.2`.
3. Add a high-frequency loss term.
4. Add a pixel-space UNet or multi-scale PatchGAN discriminator alongside the DINO discriminator.
5. Add a shallow conv refinement head after the transformer output.

That sequence is the best balance of:

- implementation effort,
- scientific clarity,
- and probability of visibly reducing blur.

## Concrete Repo-Level Suggestions

### Training

- Remove the `--epochs 12` override in `sbatch/decoder_training.sbatch`.
- Log LPIPS during eval, not just during training.
- Add sharpness-sensitive metrics to eval: LPIPS and optionally DISTS or edge reconstruction error.
- Save fixed target/reconstruction crop grids every eval so sharpness regressions are obvious.

### Losses

- Keep L1 as the base reconstruction term.
- Add one explicit high-frequency term.
- Ablate adaptive GAN weighting.

### Discriminator

- Keep the DINO discriminator for semantic realism.
- Add a second discriminator specialized for pixel-level sharpness.
- If you only try one replacement, a UNet discriminator is the most research-backed choice from the papers I reviewed.

### Architecture

- Add modality embeddings for DINO vs VAE streams.
- Add a conv refinement tail.
- Later, decouple query count from latent token count so output patches can be smaller than `16x16`.

## My Read On The Biggest Root Causes

If I had to rank the likely causes of the blur in your current setup:

1. coarse `16x16` linear patch output head with no local refinement,
2. too little fully active adversarial training in the current `12`-epoch sbatch run,
3. discriminator not being sharpness-sensitive enough,
4. no explicit high-frequency loss,
5. `noise_tau` trading clean sharpness for robustness,
6. weaker-than-ideal DINO/VAE fusion signaling.

## Sources

Primary sources I used for the research scan:

- Esser et al., "Taming Transformers for High-Resolution Image Synthesis" (CVPR 2021 / arXiv 2012.09841): https://arxiv.org/abs/2012.09841
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022 / arXiv 2112.10752): https://arxiv.org/abs/2112.10752
- Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2, arXiv 2204.06125): https://arxiv.org/abs/2204.06125
- Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (arXiv 2307.01952): https://arxiv.org/abs/2307.01952
- Sadat et al., "LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models" (NeurIPS 2024 / arXiv 2405.14477): https://arxiv.org/abs/2405.14477
- Zheng et al., "Diffusion Transformers with Representation Autoencoders" (arXiv 2510.11690): https://arxiv.org/abs/2510.11690

## Notes On Interpretation

Two points are important:

- Some recommendations above are direct conclusions from the papers.
- Some are engineering inferences from your current code and the visual failure mode.

The strongest paper-backed recommendations for your exact problem are:

- use stronger sharpening supervision,
- consider high-frequency losses,
- improve discriminator design,
- and do not underestimate decoder architecture choices.
