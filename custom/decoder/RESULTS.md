# Decoder Small — Reconstruction Results

## Our Model

| Metric | Value |
|---|---|
| Parameters | 114.68M |
| GFLOPs | 65.58 |
| Output Resolution | 256x256 |
| Encoder | DINOv2 (frozen) |
| rFID (ImageNet-1K val, 50K) | 0.387 |
| Reconstruction MSE | 0.0805 |

## Comparison with RAE Decoders (Zheng et al., 2025)

All RAE results use DINOv2-B encoder with ViT decoders on ImageNet 256x256.

| Decoder | rFID | GFLOPs |
|---|---|---|
| **Ours (Small)** | **0.387** | **65.58** |
| RAE ViT-B | 0.58 | 22.2 |
| RAE ViT-L | 0.50 | 78.1 |
| RAE ViT-XL | 0.49 | 106.7 |
| SD-VAE (full) | 0.62 | 310.4 |


Decoder achieves the best rFID (0.387) at a compute cost between ViT-B and ViT-L.

## Comparison with Other Decoders/Tokenizers (ImageNet 256x256)

Collected from recent literature. All rFID measured on ImageNet-1K validation set.


| Model | Params | Tokens | rFID | Source |
|---|---|---|---|---|
| **Ours (Small, DINOv2)** | **114.68M** | **512** | **0.387** | - |
| VA-VAE | 70M | 256 | 0.28 | Yao et al., 2025 |
| ViTok S-L/16 | 426.8M | 256 | 0.46 | Hansen-Estruch et al., 2025 |
| MAETok | 176M | 128 | 0.48 | Chen et al., 2025 |
| RAE ViT-XL (DINOv2-B) | - | 256 | 0.49 | Zheng et al., 2025 |
| ViTok S-B/16 | 129M | 256 | 0.50 | Hansen-Estruch et al., 2025 |
| SoftVQ | 176M | 64 | 0.61 | Chen et al., 2024 |
| SD-VAE | 84M | 1024 | 0.62 | Rombach et al., 2022 |
| SDXL-VAE | - | 1024 | 0.67-0.68 | Podell et al., 2023 |
| TexTok | 176M | 256 | 0.69 | - |
| DC-AE (f32c32) | 323M | 64 | 0.77 | Chen et al., 2025 |
| SD-VAE (f8) | 59.3M | 1024 | 0.78 | Rombach et al., 2022 |
| Cosmos-CI | - | 256 | 2.02 | NVIDIA, 2024 |

### VAEs with Equivariance Regularization (EQ-VAE)

| Model | rFID | rFID + EQ | Source |
|---|---|---|---|
| SD3-VAE | 0.20 | 0.19 | Kouzelis et al., 2025 |
| SDXL-VAE | 0.67 | 0.65 | Kouzelis et al., 2025 |
| SD-VAE | 0.90 | 0.82 | Kouzelis et al., 2025 |

### Higher Compression Ratios (ImageNet 256x256)

| Model | Latent Shape | rFID | Source |
|---|---|---|---|
| DC-AE (f32c32) | 8x8x32 | 0.69 | Chen et al., 2025 |
| SD-VAE (f32c32) | 8x8x32 | 2.64 | Chen et al., 2025 |
| DC-AE (f64c128) | 4x4x128 | 0.81 | Chen et al., 2025 |
| SD-VAE (f64c128) | 4x4x128 | 26.65 | Chen et al., 2025 |

## Comparison Across Encoders (RAE, ViT-XL decoder)

| Encoder | rFID | Linear Probe Acc |
|---|---|---|
| DINOv2-B | 0.49 | 84.5% |
| SigLIP2-B | 0.53 | 79.1% |
| MAE-B | 0.16 | 68.0% |
| SD-VAE | 0.62 | 8.0% |

## Generation Results (ImageNet 256x256, no guidance)

| Method | Epochs | gFID |
|---|---|---|
| DiTDH-XL (RAE DINOv2-B) | 800 | 1.51 |
| DiTDH-XL (RAE DINOv2-B) | 80 | 2.16 |
| REPA-E | 800 | 1.70 |
| DDT | 400 | 6.27 |
| REPA | 800 | 5.78 |
| VA-VAE | 800 | 2.17 |
| SiT-XL | 1400 | 8.61 |
| DiT-XL | 1400 | 9.62 |

## Config

| Setting | Value |
|---|---|
| Split | validation |
| Batch Size | 100 |
| Checkpoint | `custom_decoder_small_gan_lower_noise/checkpoint-last.pth` |
| Checkpoint Key | `model_ema` |
| Checkpoint Epoch | 13 |
| FID Batch Size | 256 |
| FID Dims | 2048 |
