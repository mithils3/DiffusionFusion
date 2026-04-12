## Custom Dual-Stream Diffusion Stack

This package contains the repo-owned `custom` training and evaluation stack used in DiffusionFusion.

### Runtime Surface

- Training entrypoint: `custom/main_custom.py`
- Decoder training entrypoint: `custom/main_decoder.py`
- Decoder eval helpers: `custom/eval/eval_decoder.py`
- Latent-only VAE eval helpers: `custom/eval/vae_eval.py`
- Built-in FID stats: `custom/fid_stats/custom_in256_stats.npz`, `custom/fid_stats/custom_in512_stats.npz`

### Supported Denoiser IDs

- `CustomDiT-B/2-4C`
- `CustomDiT-B/4-4C`
- `CustomDiT-L/2-4C`
- `CustomDiT-L/4-4C`

Legacy `JiT-*` checkpoints and runtime IDs are intentionally unsupported after the LightningDiT-based backbone migration.

### Architecture Summary

- Dual-stream denoiser over latent and DINO features
- LightningDiT-style transformer blocks with RMSNorm, SwiGLU, RoPE, and standard AdaLN
- Separate latent and DINO type embeddings with per-stream spatial RoPE
- Unchanged outer workflow: latent shards + DINO shards, same denoiser/sampler API, same decoder/eval flow, same checkpoint key layout

### Historical Citations

- Upstream JiT paper/repo: https://arxiv.org/abs/2511.13720 and https://github.com/LTH14/JiT
- Upstream LightningDiT repo: https://github.com/hustvl/LightningDiT
