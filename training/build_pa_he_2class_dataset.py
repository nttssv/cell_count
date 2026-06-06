"""Build a two-class YOLO segmentation dataset from the PA H&E annotation repo.

The source data repo stays separate from this training repo. This script reads:

- <data-repo>/yolo_seg_dataset for nucleus and clear-cell-boundary polygons
- <data-repo>/auxiliary_masks/*_gt_uncertain_ignore.png for uncertain boundary
  regions, which are merged into the boundary class by default

The generated runtime dataset is written under the training repo outputs folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "nucleus", 1: "cell_boundary"}
SOURCE_CLASS_NAMES = {
    0: "nucleus",
    1: "clear_cell_boundary",
    2: "compact_cell_boundary",
    3: "stroma",
}


def parse_tile_multiplier(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TILE_ID=MULTIPLIER")
    tile_id, raw_multiplier = value.split("=", 1)
    tile_id = tile_id.strip()
    if not tile_id:
        raise argparse.ArgumentTypeError("Tile id is empty")
    try:
        multiplier = int(raw_multiplier)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid multiplier: {raw_multiplier}") from exc
    if multiplier < 1:
        raise argparse.ArgumentTypeError("Multiplier must be >= 1")
    return tile_id, multiplier


def read_source_label_lines(label_path: Path) -> tuple[list[str], Counter[str]]:
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        old_cls = int(float(parts[0]))
        if old_cls == 0:
            lines.append(" ".join(["0", *parts[1:]]))
            counts["nucleus"] += 1
        elif old_cls == 1:
            lines.append(" ".join(["1", *parts[1:]]))
            counts["clear_cell_boundary"] += 1
            counts["cell_boundary"] += 1
        elif old_cls in SOURCE_CLASS_NAMES:
            counts[f"skipped_{SOURCE_CLASS_NAMES[old_cls]}"] += 1
    return lines, counts


def uncertain_mask_to_yolo_lines(
    mask_path: Path,
    width: int,
    height: int,
    *,
    min_area: float,
    epsilon_fraction: float,
) -> tuple[list[str], int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], 0

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(1.0, epsilon_fraction * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        coords: list[str] = []
        for x, y in approx:
            nx = min(max(float(x) / width, 0.0), 1.0)
            ny = min(max(float(y) / height, 0.0), 1.0)
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")
        lines.append("1 " + " ".join(coords))

    return lines, len(lines)


def write_data_yaml(output_dir: Path) -> Path:
    data_yaml = output_dir / "data.yaml"
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    data_yaml.write_text(
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return data_yaml


def build_dataset(args: argparse.Namespace) -> dict:
    data_repo = args.data_repo.expanduser().resolve()
    source_yolo = data_repo / "yolo_seg_dataset"
    auxiliary_masks = data_repo / "auxiliary_masks"
    output_dir = args.output.expanduser().resolve()

    if not source_yolo.exists():
        raise FileNotFoundError(f"Missing source YOLO dataset: {source_yolo}")
    if args.include_uncertain_as_boundary and not auxiliary_masks.exists():
        raise FileNotFoundError(f"Missing auxiliary mask folder: {auxiliary_masks}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    oversample = dict(args.oversample_tile or [])
    rows: list[dict[str, object]] = []
    aggregate: Counter[str] = Counter()

    for split in ["train", "val"]:
        out_image_dir = output_dir / "images" / split
        out_label_dir = output_dir / "labels" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        source_image_dir = source_yolo / "images" / split
        source_label_dir = source_yolo / "labels" / split
        if not source_image_dir.exists():
            raise FileNotFoundError(f"Missing image split: {source_image_dir}")

        for image_path in sorted(source_image_dir.glob("*.png")):
            tile_id = image_path.stem
            label_path = source_label_dir / f"{tile_id}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {tile_id}: {label_path}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]

            lines, counts = read_source_label_lines(label_path)
            uncertain_count = 0
            if args.include_uncertain_as_boundary:
                uncertain_path = auxiliary_masks / f"{tile_id}_gt_uncertain_ignore.png"
                uncertain_lines, uncertain_count = uncertain_mask_to_yolo_lines(
                    uncertain_path,
                    width,
                    height,
                    min_area=args.min_uncertain_area,
                    epsilon_fraction=args.contour_epsilon_fraction,
                )
                lines.extend(uncertain_lines)
                counts["uncertain_cell_boundary"] += uncertain_count
                counts["cell_boundary"] += uncertain_count

            copies = oversample.get(tile_id, 1) if split == "train" else 1
            for copy_idx in range(copies):
                suffix = "" if copy_idx == 0 else f"_os{copy_idx + 1}"
                out_stem = f"{tile_id}{suffix}"
                shutil.copy2(image_path, out_image_dir / f"{out_stem}.png")
                (out_label_dir / f"{out_stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
                row = {
                    "tile_id": tile_id,
                    "out_tile_id": out_stem,
                    "copy_id": copy_idx + 1,
                    "split": split,
                    "width": width,
                    "height": height,
                    "nucleus": counts["nucleus"],
                    "clear_cell_boundary": counts["clear_cell_boundary"],
                    "uncertain_cell_boundary": uncertain_count,
                    "cell_boundary": counts["cell_boundary"],
                    "skipped_compact_cell_boundary": counts["skipped_compact_cell_boundary"],
                    "skipped_stroma": counts["skipped_stroma"],
                    "total": counts["nucleus"] + counts["cell_boundary"],
                }
                rows.append(row)
                aggregate.update(
                    {
                        "images": 1,
                        "nucleus": row["nucleus"],
                        "clear_cell_boundary": row["clear_cell_boundary"],
                        "uncertain_cell_boundary": row["uncertain_cell_boundary"],
                        "cell_boundary": row["cell_boundary"],
                        "skipped_compact_cell_boundary": row["skipped_compact_cell_boundary"],
                        "skipped_stroma": row["skipped_stroma"],
                    }
                )

    data_yaml = write_data_yaml(output_dir)
    summary = {
        "data_repo": str(data_repo),
        "source_yolo_dataset": str(source_yolo),
        "auxiliary_masks": str(auxiliary_masks),
        "output_dataset": str(output_dir),
        "data_yaml": str(data_yaml),
        "classes": CLASS_NAMES,
        "source_classes": SOURCE_CLASS_NAMES,
        "include_uncertain_as_boundary": args.include_uncertain_as_boundary,
        "min_uncertain_area": args.min_uncertain_area,
        "contour_epsilon_fraction": args.contour_epsilon_fraction,
        "oversample_tile": oversample,
        "aggregate": dict(aggregate),
        "rows": rows,
    }

    import csv

    with (output_dir / "label_counts_2class_boundary_plus_uncertain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "conversion_summary_2class_boundary_plus_uncertain.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo", type=Path, required=True, help="Path to cloned nttssv/training_pa_he_annotation repo")
    parser.add_argument("--output", type=Path, required=True, help="Output runtime YOLO dataset directory")
    parser.add_argument("--include-uncertain-as-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-uncertain-area", type=float, default=20.0)
    parser.add_argument("--contour-epsilon-fraction", type=float, default=0.002)
    parser.add_argument(
        "--oversample-tile",
        action="append",
        type=parse_tile_multiplier,
        default=[],
        help="Optional train-tile oversampling in TILE_ID=MULTIPLIER format",
    )
    return parser.parse_args()


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
