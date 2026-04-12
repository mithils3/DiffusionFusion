https://github.com/openai/guided-diffusion/tree/main/evaluations - Link to ADM evaluation suite for DiT; useful reference for the custom stack, though the upstream JiT authors use torch-fidelity.
set -g mouse on add this to my tmux config 
![alt text](benchmark_results.png)

https://huggingface.co/stabilityai/sdxl-vae - VAE to use for custom latent evaluation
## Metrics to keep in mind 

### FID (Fréchet Inception Distance)
FID measures how close generated images are to real images by comparing their feature distributions extracted from an Inception network. It models both as Gaussians and computes the Fréchet distance between their means and covariances—lower is better.

### IS (Inception Score)
IS evaluates image quality and diversity using a pretrained Inception classifier. Good models produce images that lead to confident predictions (low entropy per image) while covering many classes overall (high entropy across images)—higher is better.


## TODOs
- Extract the autoencoder from DiT based models for usage
- Design the architecture for a latent-based custom model, using the autoencoder from DiT. Run a short training on this model to make sure it can be used in combination with the earlier pixel-based experiments.
- Design the bridge between the earlier pixel-space JiT experiments and the latent-based custom model, probably using bidirectional cross-attention between the two models.

