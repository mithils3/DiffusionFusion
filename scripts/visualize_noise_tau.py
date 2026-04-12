"""Visualize the effect of noise_tau on latent/feature corruption.

Run:
    python scripts/visualize_noise_tau.py [--image PATH] [--tau 0.2] [--samples 8]

Opens a matplotlib window showing:
  - Row 1: The original image repeated
  - Row 2: Images reconstructed after adding noise_tau-level corruption
           to a dummy "latent" (the image pixels themselves, for visualization)
  - Row 3: The per-pixel absolute difference (amplified 5x)
  - A histogram of sampled per-batch sigma values

This gives intuition for how much corruption noise_tau=0.2 introduces.
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def sample_noise_sigmas(batch_size: int, noise_tau: float) -> torch.Tensor:
    """Same sampling as custom/decoder/gan.py apply_noise_augmentation."""
    return torch.empty(batch_size).normal_(mean=0.0, std=noise_tau).abs_()


def corrupt_tensor(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.view(x.shape[0], *([1] * (x.ndim - 1)))
    return x + sigma * torch.randn_like(x)


def load_image_as_tensor(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize noise_tau corruption")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image. If omitted, uses a synthetic gradient image.")
    parser.add_argument("--tau", type=float, default=0.2, help="noise_tau value")
    parser.add_argument("--samples", type=int, default=8, help="Number of corrupted samples to show")
    args = parser.parse_args()

    noise_tau = args.tau
    n_samples = args.samples

    if args.image:
        original = load_image_as_tensor(args.image)
    else:
        # Synthetic test image with fine detail
        x = torch.linspace(0, 1, 256).unsqueeze(0).repeat(256, 1)
        y = torch.linspace(0, 1, 256).unsqueeze(1).repeat(1, 256)
        r = (torch.sin(x * 30) * 0.5 + 0.5)
        g = (torch.sin(y * 30) * 0.5 + 0.5)
        b = (torch.sin((x + y) * 20) * 0.5 + 0.5)
        original = torch.stack([r, g, b], dim=0).unsqueeze(0)

    # Repeat for batch
    batch = original.repeat(n_samples, 1, 1, 1)

    # Sample sigmas the same way the training code does
    sigmas = sample_noise_sigmas(n_samples, noise_tau)
    corrupted = corrupt_tensor(batch, sigmas)

    # Also sample a large batch of sigmas to show the distribution
    many_sigmas = sample_noise_sigmas(10000, noise_tau)
    half_normal_mean = noise_tau * math.sqrt(2.0 / math.pi)

    # --- Plot ---
    fig = plt.figure(figsize=(2.5 * n_samples + 3, 10))
    gs = fig.add_gridspec(3, n_samples + 1, width_ratios=[1] * n_samples + [1.2],
                          hspace=0.3, wspace=0.15)

    for i in range(n_samples):
        # Row 0: original
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tensor_to_numpy_image(original))
        ax.set_title(f"original", fontsize=8)
        ax.axis("off")

        # Row 1: corrupted
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(tensor_to_numpy_image(corrupted[i:i+1]))
        ax.set_title(f"sigma={sigmas[i]:.3f}", fontsize=8)
        ax.axis("off")

        # Row 2: difference (amplified)
        diff = (corrupted[i:i+1] - original).abs() * 5.0
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(tensor_to_numpy_image(diff))
        ax.set_title(f"|diff| x5", fontsize=8)
        ax.axis("off")

    # Sigma distribution histogram (right column)
    ax_hist = fig.add_subplot(gs[:, -1])
    ax_hist.hist(many_sigmas.numpy(), bins=60, color="#2ecc71", edgecolor="white",
                 density=True, alpha=0.85)
    ax_hist.axvline(half_normal_mean, color="red", linestyle="--", linewidth=1.5,
                    label=f"mean={half_normal_mean:.3f}")
    ax_hist.axvline(noise_tau, color="orange", linestyle=":", linewidth=1.5,
                    label=f"tau={noise_tau}")
    ax_hist.set_xlabel("per-sample sigma", fontsize=9)
    ax_hist.set_ylabel("density", fontsize=9)
    ax_hist.set_title(f"sigma ~ |N(0, {noise_tau})|", fontsize=10)
    ax_hist.legend(fontsize=8)

    fig.suptitle(
        f"noise_tau = {noise_tau}   |   mean corruption = {half_normal_mean:.1%} of feature std",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("noise_tau_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved noise_tau_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
