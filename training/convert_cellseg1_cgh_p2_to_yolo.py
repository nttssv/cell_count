"""Convert the CGH P2 CellSeg1 QuPath export into a YOLO segmentation dataset.

The source export keeps trainable cell boundaries as instance masks plus CSV
metadata, and keeps nuclei/stroma in auxiliary masks. This converter writes a
YOLO-seg dataset with these positive classes:

0 nucleus
1 clear_cell_boundary
2 compact_cell_boundary
3 stroma

Uncertain and edge/invalid masks are intentionally not converted into positive
YOLO labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "nucleus",
    1: "clear_cell_boundary",
    2: "compact_cell_boundary",
    3: "stroma",
}

BOUNDARY_CLASS_TO_ID = {
    "GT Clear cell boundary": 1,
    "GT Compact cell boundary": 2,
}

DEFAULT_SOURCE = Path("/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2")
DEFAULT_OUTPUT = DEFAULT_SOURCE / "yolo_seg_dataset"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def tile_number(tile_id: str) -> int:
    return int(tile_id.rsplit("_", 1)[1])


def choose_split(tile_ids: list[str], val_every: int) -> dict[str, str]:
    split = {}
    for tile_id in tile_ids:
        split[tile_id] = "val" if tile_number(tile_id) % val_every == 0 else "train"
    if all(value == "train" for value in split.values()) and tile_ids:
        split[tile_ids[-1]] = "val"
    return split


def load_instance_class_map(source_dir: Path, tile_id: str) -> dict[int, int]:
    rows = read_csv_rows(source_dir / f"{tile_id}_instances.csv")
    mapping: dict[int, int] = {}
    for row in rows:
        instance_label = int(row["instance_label"])
        boundary_class = row["boundary_class"]
        if boundary_class not in BOUNDARY_CLASS_TO_ID:
            raise ValueError(f"Unexpected boundary class {boundary_class!r} in {tile_id}")
        mapping[instance_label] = BOUNDARY_CLASS_TO_ID[boundary_class]
    return mapping


def contours_to_yolo_lines(
    mask: np.ndarray,
    class_id: int,
    width: int,
    height: int,
    min_area: float,
    epsilon_px: float,
) -> list[str]:
    binary = (mask > 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, closed=True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        coords: list[str] = []
        for x, y in points:
            xn = min(max(float(x) / width, 0.0), 1.0)
            yn = min(max(float(y) / height, 0.0), 1.0)
            coords.extend([f"{xn:.6f}", f"{yn:.6f}"])
        lines.append(" ".join([str(class_id)] + coords))
    return lines


def instance_mask_to_yolo_lines(
    mask_path: Path,
    instance_to_class: dict[int, int],
    width: int,
    height: int,
    min_area: float,
    epsilon_px: float,
) -> list[str]:
    mask = np.array(Image.open(mask_path))
    lines: list[str] = []
    for label in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        class_id = instance_to_class[label]
        lines.extend(
            contours_to_yolo_lines(
                mask=(mask == label),
                class_id=class_id,
                width=width,
                height=height,
                min_area=min_area,
                epsilon_px=epsilon_px,
            )
        )
    return lines


def binary_or_instance_mask_to_yolo_lines(
    mask_path: Path,
    class_id: int,
    width: int,
    height: int,
    min_area: float,
    epsilon_px: float,
) -> list[str]:
    mask = np.array(Image.open(mask_path))
    lines: list[str] = []
    labels = [int(value) for value in np.unique(mask) if int(value) != 0]
    if labels and max(labels) > 1 and len(labels) > 1:
        for label in sorted(labels):
            lines.extend(
                contours_to_yolo_lines(
                    mask=(mask == label),
                    class_id=class_id,
                    width=width,
                    height=height,
                    min_area=min_area,
                    epsilon_px=epsilon_px,
                )
            )
    else:
        lines.extend(
            contours_to_yolo_lines(
                mask=(mask > 0),
                class_id=class_id,
                width=width,
                height=height,
                min_area=min_area,
                epsilon_px=epsilon_px,
            )
        )
    return lines


def write_yaml(dataset_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    (dataset_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )


def convert_dataset(args: argparse.Namespace) -> dict:
    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    manifest_path = source_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows = read_csv_rows(manifest_path)
    tile_ids = [row["tile_id"] for row in rows]
    split_by_tile = choose_split(tile_ids, args.val_every)

    reset_dir(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tile_summaries = []

    for row in rows:
        tile_id = row["tile_id"]
        split = split_by_tile[tile_id]
        image_src = source_dir / row["image_file"]
        image_dst = output_dir / "images" / split / image_src.name
        label_dst = output_dir / "labels" / split / f"{tile_id}.txt"
        width = int(row["width"])
        height = int(row["height"])
        shutil.copy2(image_src, image_dst)

        instance_to_class = load_instance_class_map(source_dir, tile_id)
        lines: list[str] = []
        lines.extend(
            binary_or_instance_mask_to_yolo_lines(
                source_dir / "auxiliary_masks" / f"{tile_id}_gt_nucleus_instances.png",
                class_id=0,
                width=width,
                height=height,
                min_area=args.min_nucleus_area,
                epsilon_px=args.nucleus_epsilon,
            )
        )
        lines.extend(
            instance_mask_to_yolo_lines(
                source_dir / row["mask_file"],
                instance_to_class=instance_to_class,
                width=width,
                height=height,
                min_area=args.min_boundary_area,
                epsilon_px=args.boundary_epsilon,
            )
        )
        lines.extend(
            binary_or_instance_mask_to_yolo_lines(
                source_dir / "auxiliary_masks" / f"{tile_id}_gt_stroma.png",
                class_id=3,
                width=width,
                height=height,
                min_area=args.min_stroma_area,
                epsilon_px=args.stroma_epsilon,
            )
        )

        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        per_tile_counts = Counter(CLASS_NAMES[int(line.split()[0])] for line in lines)
        class_counts.update(per_tile_counts)
        split_counts[split].update(per_tile_counts)
        tile_summaries.append(
            {
                "tile_id": tile_id,
                "split": split,
                "image": str(image_dst),
                "label": str(label_dst),
                "labels": sum(per_tile_counts.values()),
                "class_counts": dict(per_tile_counts),
            }
        )

    write_yaml(output_dir)

    summary = {
        "source": str(source_dir),
        "output": str(output_dir),
        "data_yaml": str(output_dir / "data.yaml"),
        "classes": CLASS_NAMES,
        "tile_count": len(rows),
        "splits": dict(Counter(split_by_tile.values())),
        "class_counts": dict(class_counts),
        "split_class_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "tiles": tile_summaries,
        "notes": [
            "Uncertain cell boundaries are excluded from positive YOLO labels.",
            "Edge/invalid boundaries are excluded from positive YOLO labels.",
            "Stroma is converted from exported binary components after positive-boundary subtraction.",
        ],
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "label_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_id", "split", *CLASS_NAMES.values(), "total"])
        writer.writeheader()
        for item in tile_summaries:
            counts = item["class_counts"]
            writer.writerow(
                {
                    "tile_id": item["tile_id"],
                    "split": item["split"],
                    **{name: counts.get(name, 0) for name in CLASS_NAMES.values()},
                    "total": item["labels"],
                }
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-every", type=int, default=5, help="Put every Nth tile into validation.")
    parser.add_argument("--min-nucleus-area", type=float, default=12.0)
    parser.add_argument("--min-boundary-area", type=float, default=24.0)
    parser.add_argument("--min-stroma-area", type=float, default=64.0)
    parser.add_argument("--nucleus-epsilon", type=float, default=0.6)
    parser.add_argument("--boundary-epsilon", type=float, default=1.2)
    parser.add_argument("--stroma-epsilon", type=float, default=1.5)
    args = parser.parse_args()
    summary = convert_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
