# Decoder Small — Reconstruction Results

## Model

| Metric | Value |
|---|---|
| Parameters | 114.68M |
| GFLOPs | 65.58 |
| Output Resolution | 256x256 |

## Reconstruction Evaluation (ImageNet-1K Validation)

| Metric | Value |
|---|---|
| FID | 0.387 |
| Reconstruction MSE | 0.0805 |
| Num Images | 50,000 |

## Config

| Setting | Value |
|---|---|
| Split | validation |
| Batch Size | 100 |
| Checkpoint | `jit_decoder_small_gan_lower_noise/checkpoint-last.pth` |
| Checkpoint Key | `model_ema` |
| Checkpoint Epoch | 13 |
| FID Batch Size | 256 |
| FID Dims | 2048 |
