import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def sample_sigma(tau: float, rng: np.random.Generator) -> float:
    if tau <= 0.0:
        return 0.0
    return float(abs(rng.normal(loc=0.0, scale=tau)))


def apply_pixel_space_analogue(
    image_array: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    sigma = sample_sigma(tau, rng)
    if sigma == 0.0:
        return image_array.copy(), sigma
    noise = rng.normal(loc=0.0, scale=sigma, size=image_array.shape).astype(np.float32)
    noisy = np.clip(image_array + noise, 0.0, 1.0)
    return noisy, sigma


def make_panel(image_array: np.ndarray, label: str, panel_size: int, font: ImageFont.ImageFont) -> Image.Image:
    image_uint8 = (np.clip(image_array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = Image.fromarray(image_uint8)
    image = ImageOps.fit(image, (panel_size, panel_size), method=Image.Resampling.LANCZOS)

    label_height = 44
    panel = Image.new("RGB", (panel_size, panel_size + label_height), color=(255, 255, 255))
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, panel_size, panel_size, panel_size + label_height), fill=(245, 245, 245))
    draw.multiline_text((8, panel_size + 6), label, fill=(20, 20, 20), font=font, spacing=2)
    return panel


def build_grid(
    source_image: Image.Image,
    taus: list[float],
    focus_tau: float,
    focus_draws: int,
    seed: int,
    panel_size: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    font = ImageFont.load_default()
    image_array = np.asarray(source_image.convert("RGB"), dtype=np.float32) / 255.0

    top_panels: list[Image.Image] = []
    original = make_panel(image_array, "original", panel_size, font)
    top_panels.append(original)
    for tau in taus:
        noisy, sigma = apply_pixel_space_analogue(image_array, tau, rng)
        label = f"tau={tau:.2f}\nsigma={sigma:.3f}"
        top_panels.append(make_panel(noisy, label, panel_size, font))

    bottom_panels: list[Image.Image] = []
    for draw_idx in range(focus_draws):
        noisy, sigma = apply_pixel_space_analogue(image_array, focus_tau, rng)
        label = f"tau={focus_tau:.2f} draw {draw_idx + 1}\nsigma={sigma:.3f}"
        bottom_panels.append(make_panel(noisy, label, panel_size, font))

    cols = max(len(top_panels), len(bottom_panels))
    spacer = 12
    row_gap = 24
    caption_height = 56
    width = cols * panel_size + (cols - 1) * spacer
    height = caption_height + 2 * (panel_size + 44) + row_gap
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    caption = (
        "Decoder noise visualization (pixel-space analogue)\n"
        "Training noise is added in latent space; this shows comparable per-sample sigma sampling."
    )
    draw.multiline_text((8, 8), caption, fill=(10, 10, 10), font=font, spacing=3)

    y_top = caption_height
    for idx, panel in enumerate(top_panels):
        x = idx * (panel_size + spacer)
        canvas.paste(panel, (x, y_top))

    y_bottom = caption_height + panel_size + 44 + row_gap
    for idx, panel in enumerate(bottom_panels):
        x = idx * (panel_size + spacer)
        canvas.paste(panel, (x, y_bottom))

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize stochastic decoder noise.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an input image.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the visualization.")
    parser.add_argument(
        "--taus",
        type=float,
        nargs="*",
        default=[0.2, 0.4, 0.8],
        help="Tau values to visualize in the top row.",
    )
    parser.add_argument("--focus-tau", type=float, default=0.4, help="Tau to repeat in the bottom row.")
    parser.add_argument("--focus-draws", type=int, default=4, help="Number of repeated stochastic draws.")
    parser.add_argument("--panel-size", type=int, default=256, help="Per-panel output size.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.input)
    grid = build_grid(
        source_image=image,
        taus=list(args.taus),
        focus_tau=float(args.focus_tau),
        focus_draws=int(args.focus_draws),
        seed=int(args.seed),
        panel_size=int(args.panel_size),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
