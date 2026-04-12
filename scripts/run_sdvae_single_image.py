#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.transforms import functional as TF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and reconstruct a single image with a Stable Diffusion VAE."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where reconstruction artifacts will be written.",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sdxl-vae",
        help="Diffusers VAE model id or local path. Defaults to the custom repo baseline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device to use. Defaults to "cuda" when available, otherwise "cpu".',
    )
    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Sample from the encoder posterior instead of using its mean for a deterministic reconstruction.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model weights from the local Hugging Face cache.",
    )
    return parser.parse_args()


def center_crop_multiple_of_8(image: Image.Image) -> Image.Image:
    width, height = image.size
    cropped_width = width - (width % 8)
    cropped_height = height - (height % 8)

    if cropped_width == 0 or cropped_height == 0:
        raise ValueError(
            f"Input image is too small for VAE encoding after enforcing /8 dimensions: {width}x{height}"
        )

    if (cropped_width, cropped_height) == (width, height):
        return image

    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    return image.crop((left, top, left + cropped_width, top + cropped_height))


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(-1.0, 1.0).add(1.0).div(2.0).cpu()
    return TF.to_pil_image(tensor)


def save_comparison(original: Image.Image, reconstruction: Image.Image, output_path: Path) -> None:
    canvas = Image.new("RGB", (original.width + reconstruction.width, max(original.height, reconstruction.height)))
    canvas.paste(original, (0, 0))
    canvas.paste(reconstruction, (original.width, 0))
    canvas.save(output_path, format="PNG")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    image = Image.open(input_path).convert("RGB")
    processed_image = center_crop_multiple_of_8(image)

    image_tensor = TF.to_tensor(processed_image).unsqueeze(0).to(device)
    image_tensor = image_tensor.mul(2.0).sub(1.0)

    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        local_files_only=args.local_files_only,
    ).to(device)
    vae.eval()
    vae.enable_slicing()

    posterior = vae.encode(image_tensor).latent_dist
    latents = posterior.sample() if args.sample_posterior else posterior.mean
    latents = latents * vae.config.scaling_factor
    reconstruction = vae.decode(latents / vae.config.scaling_factor).sample[0]

    processed_path = output_dir / "input_processed.png"
    reconstruction_path = output_dir / "reconstruction.png"
    comparison_path = output_dir / "comparison.png"
    latents_path = output_dir / "latents.pt"

    processed_image.save(processed_path, format="PNG")
    tensor_to_image(reconstruction).save(reconstruction_path, format="PNG")
    save_comparison(processed_image, tensor_to_image(reconstruction), comparison_path)
    torch.save(latents.cpu(), latents_path)

    print(f"input={input_path}")
    print(f"processed_input={processed_path}")
    print(f"reconstruction={reconstruction_path}")
    print(f"comparison={comparison_path}")
    print(f"latents={latents_path}")
    print(f"device={device}")
    print(f"vae_model={args.vae_model}")
    print(f"sample_posterior={args.sample_posterior}")
    print(f"input_size={image.size[0]}x{image.size[1]}")
    print(f"processed_size={processed_image.size[0]}x{processed_image.size[1]}")
    print(f"latent_shape={tuple(latents.shape)}")


if __name__ == "__main__":
    main()
