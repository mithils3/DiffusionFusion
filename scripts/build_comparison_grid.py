#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stitched comparison image from matching target and reconstruction PNGs."
        )
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory containing target/reference images.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory containing reconstructed/result images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the stitched comparison image. Defaults to <target-dir>/../comparison_grid.png.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Optional number of image columns per row. Defaults to all matched images.",
    )
    parser.add_argument(
        "--image-gap",
        type=int,
        default=8,
        help="Gap in pixels between neighboring images.",
    )
    parser.add_argument(
        "--row-gap",
        type=int,
        default=14,
        help="Gap in pixels between the target row and reconstruction row.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Outer padding in pixels around the stitched image.",
    )
    parser.add_argument(
        "--label-width",
        type=int,
        default=110,
        help="Reserved width for the row labels.",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="target",
        help="Label for the top row.",
    )
    parser.add_argument(
        "--result-label",
        type=str,
        default="reconstruction",
        help="Label for the bottom row.",
    )
    return parser.parse_args()


def discover_pairs(target_dir: Path, result_dir: Path) -> list[tuple[Path, Path]]:
    target_files = {path.name: path for path in sorted(target_dir.glob("*.png"))}
    result_files = {path.name: path for path in sorted(result_dir.glob("*.png"))}
    shared_names = sorted(set(target_files) & set(result_files))
    return [(target_files[name], result_files[name]) for name in shared_names]


def build_grid(
    pairs: list[tuple[Path, Path]],
    *,
    columns: int | None,
    image_gap: int,
    row_gap: int,
    padding: int,
    label_width: int,
    target_label: str,
    result_label: str,
) -> Image.Image:
    font = ImageFont.load_default()
    target_images = [Image.open(target_path).convert("RGB") for target_path, _ in pairs]
    result_images = [Image.open(result_path).convert("RGB") for _, result_path in pairs]

    image_width = target_images[0].width
    image_height = target_images[0].height
    if any(image.size != (image_width, image_height) for image in target_images + result_images):
        raise ValueError("All target and reconstruction images must have the same size.")

    columns = len(pairs) if columns is None else max(1, min(columns, len(pairs)))
    rows_per_section = (len(pairs) + columns - 1) // columns
    grid_width = columns * image_width + max(columns - 1, 0) * image_gap
    section_height = rows_per_section * image_height + max(rows_per_section - 1, 0) * image_gap
    canvas_width = label_width + grid_width + 2 * padding
    canvas_height = 2 * section_height + row_gap + 2 * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    target_label_box = draw.textbbox((0, 0), target_label, font=font)
    result_label_box = draw.textbbox((0, 0), result_label, font=font)

    target_label_y = padding + max(0, (section_height - (target_label_box[3] - target_label_box[1])) // 2)
    result_section_y = padding + section_height + row_gap
    result_label_y = result_section_y + max(
        0,
        (section_height - (result_label_box[3] - result_label_box[1])) // 2,
    )
    draw.text((padding, target_label_y), target_label, fill=(0, 0, 0), font=font)
    draw.text((padding, result_label_y), result_label, fill=(0, 0, 0), font=font)

    image_origin_x = padding + label_width
    for index, image in enumerate(target_images):
        row = index // columns
        column = index % columns
        x = image_origin_x + column * (image_width + image_gap)
        y = padding + row * (image_height + image_gap)
        canvas.paste(image, (x, y))

    for index, image in enumerate(result_images):
        row = index // columns
        column = index % columns
        x = image_origin_x + column * (image_width + image_gap)
        y = result_section_y + row * (image_height + image_gap)
        canvas.paste(image, (x, y))

    return canvas


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    pairs = discover_pairs(target_dir, result_dir)
    if not pairs:
        raise ValueError(
            f"No matching PNG filenames found between {target_dir} and {result_dir}."
        )

    output_path = args.output
    if output_path is None:
        output_path = target_dir.parent / "comparison_grid.png"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = build_grid(
        pairs,
        columns=args.columns,
        image_gap=args.image_gap,
        row_gap=args.row_gap,
        padding=args.padding,
        label_width=args.label_width,
        target_label=args.target_label,
        result_label=args.result_label,
    )
    grid.save(output_path, format="PNG")

    print(f"pairs={len(pairs)}")
    print(f"target_dir={target_dir}")
    print(f"result_dir={result_dir}")
    print(f"output={output_path}")
    resolved_columns = len(pairs) if args.columns is None else max(1, min(args.columns, len(pairs)))
    print(f"columns={resolved_columns}")


if __name__ == "__main__":
    main()
