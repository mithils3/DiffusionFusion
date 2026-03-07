# Unified Image Decoder

## Purpose

The decoder takes the two denoised latent streams from the DiT and produces a single RGB image. It is a transformer-based image decoder in the style of RAE, but instead of inverting one representation, it fuses two.

```
Inputs:
  vae_latent:   (B, 4, 32, 32)    ← denoised VAE stream from DiT
  dino_latent:  (B, 768, 16, 16)  ← denoised DINO stream from DiT

Output:
  image:        (B, 3, 256, 256)  ← reconstructed RGB image
```

---

## Tokenization

Both inputs are converted into 256-token sequences and projected to the decoder's hidden dimension D.

**DINO.** Already a 16x16 grid of 768-dim vectors. Flatten spatially to get `(B, 256, 768)`. Linear projection from 768 to D. Each token encodes semantic information about one spatial region: object identity, part structure, scene context.

**VAE.** A 32x32 grid of 4-dim vectors. Patchify with `patch_size=2`: group every 2x2 spatial block into one token, giving `(B, 256, 16)` where `16 = 4 x 2 x 2`. Linear projection from 16 to D. Each token encodes pixel-level detail for the same spatial region as the corresponding DINO token: edges, color gradients, texture.

After projection, both are `(B, 256, D)`. They are concatenated along the token axis to form a single context sequence of `(B, 512, D)`. Each stream gets its own learned positional embeddings added before concatenation so the decoder can distinguish DINO tokens from VAE tokens and knows where each sits in the spatial grid.

The spatial alignment matters: DINO token `i` and VAE token `i` refer to the same 16x16 pixel region of the original image. After concatenation, the context sequence has 512 tokens where positions 0 through 255 are DINO and 256 through 511 are VAE, both in raster order over the same 16x16 grid.

---

## Learnable Decoder Queries

The decoder does not directly process the 512 context tokens through self-attention. Instead it uses 256 learnable query tokens, one per output patch position on the 16x16 grid. These queries are randomly initialized parameters of shape `(1, 256, D)` that are broadcast across the batch and receive their own positional embeddings.

The queries act as the decoder's working memory. Each query is responsible for producing one patch of the output image. They start as learned priors over what typical image patches look like, and are iteratively refined through the transformer blocks by attending to the context.

This is the same pattern used by RAE's ViT decoder and by DETR's object queries. It decouples the output resolution from the input token count and gives the model a clean separation between "what I'm building" (queries) and "what I'm building it from" (context).

---

## Transformer Blocks

The decoder consists of N transformer blocks applied sequentially. Each block has three stages:

**Self-attention.** All 256 queries attend to each other. This enforces spatial coherence in the output: neighboring patches agree on lighting, edges continue smoothly across patch boundaries, global structure is consistent. Without self-attention, each query would independently decode its patch with no awareness of its neighbors, producing visible seams and inconsistencies.

**Cross-attention.** All 256 queries attend to all 512 context tokens (the concatenated DINO + VAE sequences). Q comes from the 256 decoder queries, K and V come from the 512 context tokens. Each query produces a 512-length attention weight vector over the context, determining how much it pulls from each DINO and VAE token. The output remains 256 tokens since Q determines the output sequence length.

A query at position (i, j) can pull semantic context from nearby DINO tokens ("this region is part of a dog's face") and pixel detail from nearby VAE tokens ("the exact color gradient here goes from brown to white"). Because it is attention, each query can also pull from distant context tokens when needed, for example at object boundaries where multiple DINO tokens are relevant.

The cross-attention does not distinguish between DINO and VAE tokens explicitly. They sit at different positions in the context sequence (0-255 vs 256-511) with different positional embeddings, and the attention weights learn which tokens are useful for which queries. In practice, attention maps will show that queries pull heavily from both streams, using DINO for coarse structure decisions and VAE for fine detail.

**Feed-forward network.** A two-layer MLP with GELU activation applied independently to each query token. This is where the model integrates and transforms the information gathered from self-attention and cross-attention into a representation closer to the final pixel values. Each FFN layer has an expansion factor of 4 (hidden dimension is 4D).

All three stages use pre-LayerNorm and residual connections.

After N blocks of this refinement, each query token holds a D-dimensional representation that encodes everything needed to reconstruct its corresponding image patch.

---

## Pixel Output Head

Each of the 256 refined query tokens is mapped to raw pixel values by a single linear layer:

```
Linear(D, 3 x 16 x 16) = Linear(D, 768)
```

This produces 768 values per token: the RGB values for a 16x16 pixel patch. The 256 output patches are then rearranged into the full image:

```
(B, 256, 768) → rearrange → (B, 3, 256, 256)
```

The rearrangement places each token's 16x16 patch at the corresponding position in the 16x16 grid: `16 grid positions x 16 pixels per position = 256 pixels` along each spatial axis.

No convolutions, no learned upsampling. The transformer's linear head predicts all pixel values directly, and the spatial structure comes purely from the grid arrangement. This follows RAE's design exactly.

---

## Architecture Diagram

```
VAE (B,4,32,32)              DINO (B,768,16,16)
      │                             │
 patchify 2x2                   flatten
      │                             │
 (B, 256, 16)               (B, 256, 768)
      │                             │
 Linear(16, D)               Linear(768, D)
 + pos embed                  + pos embed
      │                             │
      └──────────┬──────────────────┘
           concatenate
                 │
          (B, 512, D)
           = context
                 │
                 │      ┌──────────────────┐
                 │      │  256 learnable    │
                 │      │  query tokens     │
                 │      │  (B, 256, D)      │
                 │      │  + pos embed      │
                 │      └────────┬─────────┘
                 │               │
                 │               ▼
                 │         ×N blocks:
                 │          ┌─────────────────┐
                 │          │ self-attn        │
                 │          │ (queries ↔ queries)
                 │          │                 │
                 └────────► │ cross-attn      │
                            │ (queries ← context)
                            │                 │
                            │ FFN             │
                            └────────┬────────┘
                                     │
                               (B, 256, D)
                                     │
                              Linear(D, 768)
                                     │
                              unpatchify
                              16x16 grid
                              16x16 patches
                                     │
                              (B, 3, 256, 256)
```

---

## Initialization

**Query tokens.** `N(0, 0.02)`. Small random normal to break symmetry without destabilizing early cross-attention. This is the ViT standard used by both RAE and DETR.

**Positional embeddings.** `N(0, 0.02)` for all three sets (query, DINO context, VAE context). Learned, not sinusoidal. The grid is only 16x16 so there are only 256 positions to learn.

**Input projections.** Xavier uniform (PyTorch default for `nn.Linear`) for both the VAE projection `Linear(16, D)` and DINO projection `Linear(768, D)`. These are just input embeddings so the default is fine.

**Attention layers.** Xavier uniform for Q, K, V, and output projection weight matrices. Zeros for biases. The output projection in each attention block is additionally scaled by `1 / sqrt(2N)` where N is the number of blocks. This is the GPT-2 residual scaling trick: each block adds a residual, and without scaling the variance of the residual stream grows linearly with depth. The scaling keeps it stable.

**FFN layers.** Xavier uniform for both linear layers, zeros for biases. The second linear layer (the one that feeds back into the residual stream) is scaled by `1 / sqrt(2N)` for the same reason as the attention output projection.

**LayerNorm.** Weight to ones, bias to zeros.

**Pixel output head.** `N(0, 0.02 / sqrt(2N))` or simply `N(0, 0.01)`. This is the most important initialization to get right. Small init means the model starts by predicting near-zero (near-mean) pixel values rather than random large values. If the output head has large weights at init, the first few training steps produce wild pixel predictions and the loss is huge, which can destabilize training or waste early gradient steps. The model should start by predicting a gray image and gradually learn structure.

---

## Sizing

The decoder should be lighter than the DiT. The DiT does the hard work of denoising and cross-stream fusion over many layers. The decoder's job is reconstruction from already-clean, already-fused representations. RAE uses up to 28 layers for its decoder, but RAE's decoder must invert a frozen encoder that was never trained for reconstruction. Our DiT is trained end-to-end, so its outputs are more decoder-friendly.

Recommended starting point: `D = 1152`, `N = 12 to 16` blocks, 16 attention heads. This puts the decoder at roughly 200 to 300M parameters. If that proves excessive, try `D = 768`, `N = 12` for a ~100M decoder. Scale up only if reconstruction quality is the bottleneck.

---

## Decoder Noise Augmentation

During training, the decoder receives clean latents from the frozen encoders (not from the DiT). At inference time, it receives the DiT's denoised outputs, which are only approximately clean. This train-inference gap can cause artifacts.

The fix is to add small Gaussian noise to the clean latents during decoder training, before tokenization. This teaches the decoder to handle the slightly noisy inputs it will see at inference. The noise level should be small (a hyperparameter to tune), just enough to close the gap without degrading reconstruction quality on clean inputs. RAE uses this same technique.

---

## Training

The decoder is trained in a separate stage from the DiT. Freeze both encoders (DINO and VAE), encode training images into both latent spaces, add noise augmentation (noise_tau = 0.8), and train the decoder to reconstruct the original image from both latent representations. Once trained and frozen, the DiT is trained in the latent space with its own diffusion loss. At inference, noise_tau is set to 0 and the DiT denoises latents and the frozen decoder maps them to an image.

---

## Discriminator

The discriminator is not a standard PatchGAN. It is built on a pretrained DINO ViT-Small with patch size 8, following RAE.

**Backbone.** A pretrained DINO ViT-S/8 processes the image (real or reconstructed) at 224x224. With patch size 8, this produces a 28x28 grid of 384-dim feature tokens. This is a different DINO model than the encoder: the encoder uses DINOv2-Base with patch 16 giving a 16x16 grid at 768-dim. The discriminator's ViT-S/8 sees the image at much higher spatial resolution (28x28 vs 16x16), which lets it catch fine-grained artifacts that the encoder's representation would not capture.

**Convolutional head.** On top of the DINO feature map `(B, 384, 28, 28)`, a convolutional projection with kernel size 9 slides over the spatial grid and produces a real/fake logit at each position. Batch normalization and spectral normalization are applied. Spectral normalization constrains the Lipschitz constant of the conv layers, preventing the discriminator from producing extreme gradients and stabilizing GAN training.

**Output.** A spatial map of real/fake logits. Each position in the output map corresponds to a region of the image and predicts whether that region looks real or reconstructed. This is patch-based in that it outputs per-region predictions, but the patches are defined in DINO feature space, not raw pixel space. Each prediction is informed by a 9x9 region of DINO features, which covers a much larger receptive field in pixel space since each DINO token already sees its 8x8 patch plus global context from self-attention.

**Why DINO features instead of raw pixels.** A raw convolutional discriminator has to learn visual features from scratch. By starting from DINO features, the discriminator already operates in a perceptually meaningful space that captures texture, edges, object parts, and semantic structure. It can immediately focus on the perceptually meaningful differences between real and reconstructed images rather than spending capacity learning basic visual features.

```
Image (real or reconstructed, 224×224)
        │
        ▼
  DINO ViT-S/8 backbone
        │
        ▼
  (B, 384, 28, 28) feature map
        │
        ▼
  Conv head (kernel 9×9)
  + batch norm
  + spectral norm
        │
        ▼
  (B, 1, H', W') spatial logits
  each position = real/fake prediction
```

---

## Loss

Three loss components are used, following RAE.

**MSE (L2 reconstruction).** Pixel-wise mean squared error between the predicted and target images. Provides the base reconstruction signal and ensures the output is globally correct in terms of color and brightness. Tends to produce slightly blurry outputs when used alone because it penalizes all pixel errors equally, encouraging the model to hedge toward the mean when uncertain about fine details.

**LPIPS (perceptual loss, weight = 1.0).** Measures perceptual distance using features extracted from a pretrained network (typically VGG or AlexNet). Two images that look similar to a human will have low LPIPS even if their pixel values differ slightly. Compensates for MSE's tendency toward blur by rewarding outputs that match the target's texture and edge structure even if individual pixels are slightly off.

**Adversarial loss (weight = 0.75).** The discriminator's spatial logits provide a per-region training signal to the decoder.

Discriminator loss uses hinge formulation:

```
L_disc = mean(max(0, 1 - D(real))) + mean(max(0, 1 + D(reconstructed)))
```

The mean is over all spatial positions in the discriminator's output map. Once a prediction is past the margin (above +1 for real, below -1 for fake), the discriminator stops getting gradient for that position. This prevents the discriminator from pushing predictions to infinity and makes training more stable than vanilla GAN loss.

Generator (decoder) loss uses vanilla non-saturating formulation:

```
L_gen = -mean(D(reconstructed))
```

The decoder wants all spatial positions to output high values (real). The mean over spatial positions means the decoder gets a gradient signal from every region of the image. If the discriminator thinks the top-left looks fake but the bottom-right looks fine, the gradient is stronger for fixing the top-left.

Combined decoder loss:

```
L_decoder = L_mse + 1.0 * L_lpips + 0.75 * L_gen
```

The adversarial weight is capped by `max_d_weight = 10000.0` if adaptive weighting is used (where the adversarial weight is automatically scaled based on the ratio of gradients from the reconstruction and adversarial losses, following VQGAN).

The discriminator is updated once per decoder update. The decoder's adversarial gradient flows through the reconstruction back to the decoder weights, but the discriminator's weights are frozen during this step. They alternate: one discriminator step, one decoder step.

---

## Training Schedule

Training runs for 16 epochs with a global batch size of 512. The losses are phased in over the course of training:

**Epochs 0-5: MSE + LPIPS only.** The decoder learns a reasonable reconstruction without any adversarial pressure. This is critical because in the early epochs the decoder outputs near-random images. A discriminator could trivially distinguish these from real images, providing gradients that are too strong and uninformative.

**Epochs 6-7: Discriminator warmup.** The discriminator begins training (seeing real vs reconstructed images and updating its own weights) but its loss does not flow back to the decoder. This gives the discriminator two epochs to learn what reconstruction artifacts look like before its gradients start affecting the decoder.

**Epochs 8-15: Full MSE + LPIPS + adversarial.** The adversarial loss kicks in. The discriminator already knows which regions look fake, so its signal to the decoder is meaningful from the start. The decoder sharpens fine details, reduces artifacts, and produces outputs that are perceptually closer to real images.

```
Epoch:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
MSE:    ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
LPIPS:  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
D train:                     ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
D→Dec:                             ■  ■  ■  ■  ■  ■  ■  ■
```

---

## Optimizer and Schedule

Both the decoder and discriminator use AdamW with `lr = 2e-4`, `betas = (0.9, 0.95)`, `weight_decay = 0.0`.

Both use a cosine learning rate schedule with 1 epoch of linear warmup from zero, decaying from `2e-4` to `2e-5` over the full 16 epochs.

EMA is applied to the decoder with a decay of 0.9978. The EMA weights are used at inference.

---

## Discriminator Augmentation

Data augmentation is applied to the discriminator inputs with probability 1.0 (always on). Cutout is set to 0.0 (disabled). The specific augmentations are not documented in the config but typically include horizontal flips, color jitter, and similar geometric/photometric transforms. Augmentation prevents the discriminator from overfitting to the training set, which would cause it to stop providing useful gradients to the decoder.

---

## Full Training Config (Reference)

```yaml
training:
  epochs: 16
  ema_decay: 0.9978
  global_batch_size: 512
  optimizer:
    lr: 2.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  scheduler:
    type: cosine
    warmup_epochs: 1
    decay_end_epoch: 16
    base_lr: 2.0e-4
    final_lr: 2.0e-5
    warmup_from_zero: true

gan:
  disc:
    arch:
      dino_ckpt_path: dino_vit_small_patch8_224.pth
      ks: 9
      norm_type: bn
      using_spec_norm: true
      recipe: S_8
    optimizer:
      lr: 2.0e-4
      betas: [0.9, 0.95]
      weight_decay: 0.0
    augment:
      prob: 1.0
      cutout: 0.0
  loss:
    disc_loss: hinge
    gen_loss: vanilla
    disc_weight: 0.75
    perceptual_weight: 1.0
    disc_start: 8          # adversarial loss → decoder from epoch 8
    disc_upd_start: 6      # discriminator trains from epoch 6
    lpips_start: 0          # perceptual loss from epoch 0
    max_d_weight: 10000.0
    disc_updates: 1         # one D step per G step
```