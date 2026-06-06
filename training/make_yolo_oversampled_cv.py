"""Build oversampled train/val and 5-fold YOLO datasets from converted CGH P2 labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = {
    0: "nucleus",
    1: "clear_cell_boundary",
    2: "compact_cell_boundary",
    3: "stroma",
}

DEFAULT_SOURCE = Path("/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/yolo_seg_dataset")
DEFAULT_OUTPUT = Path("/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/yolo_seg_dataset_cv_oversampled")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def tile_number(tile_id: str) -> int:
    return int(tile_id.rsplit("_", 1)[1])


def class_counts(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_id = int(stripped.split()[0])
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def collect_tiles(source: Path) -> dict[str, dict]:
    tiles: dict[str, dict] = {}
    for split in ("train", "val"):
        label_dir = source / "labels" / split
        image_dir = source / "images" / split
        for label_path in sorted(path for path in label_dir.glob("*.txt") if not path.name.startswith("._")):
            tile_id = label_path.stem
            image_path = next((image_dir / f"{tile_id}{ext}" for ext in (".png", ".jpg", ".jpeg") if (image_dir / f"{tile_id}{ext}").exists()), None)
            if image_path is None:
                raise FileNotFoundError(f"Missing image for {label_path}")
            counts = class_counts(label_path)
            tiles[tile_id] = {
                "tile_id": tile_id,
                "source_split": split,
                "image": image_path,
                "label": label_path,
                "counts": counts,
            }
    return dict(sorted(tiles.items(), key=lambda item: tile_number(item[0])))


def repeat_count(counts: Counter[str], max_repeats: int) -> int:
    compact = counts.get("compact_cell_boundary", 0)
    stroma = counts.get("stroma", 0)
    repeats = 1
    if compact > 0:
        repeats += 2
    if compact >= 6:
        repeats += 1
    if compact >= 12:
        repeats += 1
    if stroma >= 5:
        repeats += 1
    return min(repeats, max_repeats)


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


def copy_tile(dataset_dir: Path, tile: dict, split: str, copy_index: int) -> dict:
    suffix = "" if copy_index == 0 else f"_copy{copy_index:02d}"
    out_id = f"{tile['tile_id']}{suffix}"
    image_dst = dataset_dir / "images" / split / f"{out_id}{tile['image'].suffix}"
    label_dst = dataset_dir / "labels" / split / f"{out_id}.txt"
    shutil.copy2(tile["image"], image_dst)
    shutil.copy2(tile["label"], label_dst)
    return {
        "tile_id": tile["tile_id"],
        "output_id": out_id,
        "split": split,
        "copy_index": copy_index,
        **{name: tile["counts"].get(name, 0) for name in CLASS_NAMES.values()},
        "total": sum(tile["counts"].values()),
    }


def build_dataset(dataset_dir: Path, tiles: dict[str, dict], val_tile_ids: set[str], max_repeats: int) -> list[dict]:
    reset_dir(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = []
    for tile_id, tile in tiles.items():
        if tile_id in val_tile_ids:
            rows.append(copy_tile(dataset_dir, tile, "val", 0))
            continue
        repeats = repeat_count(tile["counts"], max_repeats)
        for idx in range(repeats):
            rows.append(copy_tile(dataset_dir, tile, "train", idx))

    write_yaml(dataset_dir)
    with (dataset_dir / "oversampling_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["tile_id", "output_id", "split", "copy_index", *CLASS_NAMES.values(), "total"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def split_summary(rows: list[dict]) -> dict:
    by_split = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row["split"] == split]
        counts = Counter()
        for row in split_rows:
            for name in CLASS_NAMES.values():
                counts[name] += int(row[name])
        by_split[split] = {
            "images": len(split_rows),
            "labels": sum(counts.values()),
            "class_counts": dict(counts),
            "unique_tiles": len({row["tile_id"] for row in split_rows}),
        }
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=5)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    tiles = collect_tiles(source)
    reset_dir(output)

    all_summaries = {
        "source": str(source),
        "output": str(output),
        "tile_count": len(tiles),
        "max_repeats": args.max_repeats,
        "datasets": {},
    }

    source_val = {tile_id for tile_id, tile in tiles.items() if tile["source_split"] == "val"}
    rows = build_dataset(output / "trainval_oversampled", tiles, source_val, args.max_repeats)
    all_summaries["datasets"]["trainval_oversampled"] = {
        "path": str(output / "trainval_oversampled"),
        "val_tiles": sorted(source_val, key=tile_number),
        "summary": split_summary(rows),
    }

    ordered_ids = list(tiles)
    for fold in range(args.folds):
        val_ids = {tile_id for tile_id in ordered_ids if (tile_number(tile_id) - 1) % args.folds == fold}
        dataset_name = f"fold_{fold}"
        fold_rows = build_dataset(output / dataset_name, tiles, val_ids, args.max_repeats)
        all_summaries["datasets"][dataset_name] = {
            "path": str(output / dataset_name),
            "val_tiles": sorted(val_ids, key=tile_number),
            "summary": split_summary(fold_rows),
        }

    (output / "cv_oversampling_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
