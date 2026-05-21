"""Biologically constrained active-learning loop for adrenal H&E segmentation.

This is a pathology-aware self-training driver. It does not reuse pseudo-labels
unless they pass morphology, topology, confidence, and parenchyme-context QC.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage import io, measure

from cellpose_count_cells import (
    extract_he_channels,
    filter_and_relabel_mask,
    preprocess_for_cytoplasm,
    preprocess_for_nuclei,
    refine_with_watershed,
)
from morphology_qc import (
    BiologicalQCConfig,
    BiologicalQCResult,
    IMAGE_SUFFIXES,
    aggregate_qc_results,
    export_qc_outputs,
    load_rgb_image,
    morphology_metrics,
    relabel_consecutive,
)


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Runtime settings for the active-learning loop."""

    project_root: Path = Path(".")
    sample_dir: Path = Path("sample")
    corrections_dir: Path = Path("corrections")
    output_dir: Path = Path("output")
    dataset_dir: Path = Path("output/active_learning_datasets")
    runs_dir: Path = Path("runs")
    max_rounds: int = 10
    min_improvement: float = 0.01
    convergence_patience: int = 2
    yolo_base_model: str = "yolov8n-seg.pt"
    yolo_conf_threshold: float = 0.35
    cell_diameter: float = 50.0
    nuclei_diameter: float = 25.0
    flow_threshold: float = 0.35
    cellprob_threshold: float = -1.5
    epochs: int = 25
    imgsz: int = 640
    batch: int = 4
    validation_fraction: float = 0.20
    seed: int = 13
    dry_run: bool = False
    skip_training: bool = False
    cellpose_gpu: bool = False
    class_names: tuple[str, str] = ("nucleus", "parenchyme")


@dataclass
class MaskBundle:
    """One image and the current labels proposed for it."""

    image_path: Path
    nuclei_mask: np.ndarray
    parenchyme_mask: np.ndarray
    confidence_values: list[float]
    confidence_map: np.ndarray | None
    source: str
    manual_correction_applied: bool = False


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def resolve_config_paths(config: ActiveLearningConfig) -> ActiveLearningConfig:
    root = config.project_root.expanduser().resolve()
    return ActiveLearningConfig(
        **{
            **asdict(config),
            "project_root": root,
            "sample_dir": (root / config.sample_dir).resolve() if not config.sample_dir.is_absolute() else config.sample_dir.resolve(),
            "corrections_dir": (root / config.corrections_dir).resolve() if not config.corrections_dir.is_absolute() else config.corrections_dir.resolve(),
            "output_dir": (root / config.output_dir).resolve() if not config.output_dir.is_absolute() else config.output_dir.resolve(),
            "dataset_dir": (root / config.dataset_dir).resolve() if not config.dataset_dir.is_absolute() else config.dataset_dir.resolve(),
            "runs_dir": (root / config.runs_dir).resolve() if not config.runs_dir.is_absolute() else config.runs_dir.resolve(),
        }
    )


def find_images(sample_dir: str | Path) -> list[Path]:
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory does not exist: {sample_dir}")
    images = sorted(path for path in sample_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No image files found in {sample_dir}")
    return images


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def save_masks(round_dir: Path, bundle: MaskBundle) -> dict[str, Path]:
    mask_dir = round_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = bundle.image_path.stem
    nuclei_path = mask_dir / f"{stem}_nuclei_mask.npy"
    parenchyme_path = mask_dir / f"{stem}_parenchyme_mask.npy"
    np.save(nuclei_path, bundle.nuclei_mask.astype(np.int32))
    np.save(parenchyme_path, bundle.parenchyme_mask.astype(np.int32))
    return {"nuclei": nuclei_path, "parenchyme": parenchyme_path}


def connected_parenchyme_instances(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.max() <= 1:
        return measure.label(mask > 0).astype(np.int32)
    return relabel_consecutive(mask)


def yolo_segmentation_lines(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    class_id: int,
    min_points: int = 3,
    max_points: int = 120,
) -> list[str]:
    """Convert a labeled mask to YOLOv8 segmentation polygon lines."""

    height, width = image_shape[:2]
    lines: list[str] = []
    label_mask = np.asarray(mask, dtype=np.int32)
    for region in measure.regionprops(label_mask):
        row0, col0, row1, col1 = region.bbox
        object_mask = label_mask[row0:row1, col0:col1] == region.label
        contours = measure.find_contours(object_mask.astype(np.uint8), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) < min_points:
            continue
        if len(contour) > max_points:
            indices = np.linspace(0, len(contour) - 1, max_points, dtype=int)
            contour = contour[indices]
        rows = np.clip(contour[:, 0] + row0, 0, height - 1) / height
        cols = np.clip(contour[:, 1] + col0, 0, width - 1) / width
        coords = np.column_stack([cols, rows]).reshape(-1)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))
    return lines


def write_multiclass_yolo_label(
    label_path: str | Path,
    nuclei_mask: np.ndarray,
    parenchyme_mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> int:
    """Write YOLO segmentation labels for nucleus=0 and parenchyme=1."""

    lines: list[str] = []
    lines.extend(yolo_segmentation_lines(relabel_consecutive(nuclei_mask), image_shape, class_id=0))
    lines.extend(yolo_segmentation_lines(connected_parenchyme_instances(parenchyme_mask), image_shape, class_id=1))
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def split_train_val(image_paths: list[Path], validation_fraction: float, seed: int) -> tuple[set[Path], set[Path]]:
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    if len(paths) <= 1:
        return set(paths), set(paths)
    order = np.arange(len(paths))
    rng.shuffle(order)
    val_count = max(1, int(round(len(paths) * validation_fraction)))
    val_indices = set(order[:val_count].tolist())
    train = {path for idx, path in enumerate(paths) if idx not in val_indices}
    val = {path for idx, path in enumerate(paths) if idx in val_indices}
    if not train:
        train = set(paths)
    return train, val


def build_yolo_dataset(
    bundles: list[MaskBundle],
    dataset_root: Path,
    round_number: int,
    config: ActiveLearningConfig,
) -> tuple[Path, int]:
    """Create a YOLOv8 segmentation dataset for one active-learning round."""

    dataset_dir = dataset_root / f"round_{round_number}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_paths, val_paths = split_train_val([bundle.image_path for bundle in bundles], config.validation_fraction, config.seed + round_number)
    label_count = 0
    for bundle in bundles:
        split = "val" if bundle.image_path in val_paths else "train"
        image_dst = dataset_dir / "images" / split / bundle.image_path.name
        shutil.copy2(bundle.image_path, image_dst)
        image_rgb = load_rgb_image(bundle.image_path)
        label_count += write_multiclass_yolo_label(
            dataset_dir / "labels" / split / f"{bundle.image_path.stem}.txt",
            bundle.nuclei_mask,
            bundle.parenchyme_mask,
            image_rgb.shape,
        )

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {config.class_names[0]}",
                f"  1: {config.class_names[1]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path, label_count


def run_cellpose_mask(
    image: np.ndarray,
    model_type: str,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    *,
    gpu: bool,
) -> np.ndarray:
    """Run Cellpose with explicit GPU control for reproducible bootstrapping."""

    try:
        from cellpose import models
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("cellpose is required for round-0 bootstrapping. Install with `pip install cellpose`.") from exc

    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    masks, _, _ = model.eval(
        image,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return np.asarray(masks, dtype=np.int32)


def run_initial_cellpose_round(image_paths: list[Path], config: ActiveLearningConfig) -> list[MaskBundle]:
    """Round 0: bootstrap labels from Cellpose plus project-specific filters."""

    bundles: list[MaskBundle] = []
    cellpose_output = config.output_dir / "cellpose_round_0"
    cellpose_output.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        log(f"Round 0 Cellpose bootstrap: {image_path.name}")
        if config.dry_run:
            image_rgb = load_rgb_image(image_path)
            empty = np.zeros(image_rgb.shape[:2], dtype=np.int32)
            bundles.append(MaskBundle(image_path, empty, empty, [1.0], None, source="dry_run_cellpose"))
            continue

        image_rgb = load_rgb_image(image_path)
        h_channel, e_channel = extract_he_channels(image_rgb)
        nuclei_input = preprocess_for_nuclei(h_channel)
        cytoplasm_input = preprocess_for_cytoplasm(e_channel)

        raw_nuclei = run_cellpose_mask(
            nuclei_input,
            "nuclei",
            config.nuclei_diameter,
            config.flow_threshold,
            config.cellprob_threshold,
            gpu=config.cellpose_gpu,
        )
        nuclei_mask = filter_and_relabel_mask(
            raw_nuclei,
            18,
            2500,
            0.35,
            0.99,
        )

        raw_cells = run_cellpose_mask(
            cytoplasm_input,
            "cyto3",
            config.cell_diameter,
            config.flow_threshold,
            config.cellprob_threshold,
            gpu=config.cellpose_gpu,
        )
        refined_cells = refine_with_watershed(raw_cells, nuclei_mask)
        parenchyme_mask = filter_and_relabel_mask(
            refined_cells,
            100,
            30000,
            0.35,
            0.99,
        )
        np.save(cellpose_output / f"{image_path.stem}_nuclei_mask.npy", nuclei_mask.astype(np.int32))
        np.save(cellpose_output / f"{image_path.stem}_parenchyme_mask.npy", parenchyme_mask.astype(np.int32))
        bundles.append(
            MaskBundle(
                image_path=image_path,
                nuclei_mask=nuclei_mask,
                parenchyme_mask=parenchyme_mask,
                confidence_values=[1.0],
                confidence_map=None,
                source="cellpose_round_0",
            )
        )
    return bundles


def import_ultralytics_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "ultralytics is required for training/inference. Install with `pip install ultralytics`."
        ) from exc
    return YOLO


def train_yolo_model(
    dataset_yaml: Path,
    round_number: int,
    previous_weights: Path | None,
    config: ActiveLearningConfig,
) -> tuple[Path | None, dict[str, float]]:
    if config.dry_run or config.skip_training:
        log("Skipping YOLO training because dry-run or skip-training is enabled.")
        return previous_weights, {"mean_iou": float("nan"), "training_loss": float("nan")}

    YOLO = import_ultralytics_yolo()
    base_model = str(previous_weights) if previous_weights and previous_weights.exists() else config.yolo_base_model
    run_name = f"cell_segmenter_round_{round_number}"
    log(f"Training YOLOv8n-seg round {round_number}: {run_name}")
    model = YOLO(base_model)
    model.train(
        data=str(dataset_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        project=str(config.runs_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    run_dir = config.runs_dir / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"YOLO training finished but best weights were not found: {best}")
    return best, read_yolo_training_metrics(run_dir)


def read_yolo_training_metrics(run_dir: Path) -> dict[str, float]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"mean_iou": float("nan"), "training_loss": float("nan")}
    df = pd.read_csv(results_csv)
    if df.empty:
        return {"mean_iou": float("nan"), "training_loss": float("nan")}
    last = df.iloc[-1]
    loss_columns = [column for column in df.columns if "train/" in column and "loss" in column]
    metric_columns = [column for column in df.columns if "mAP50-95" in column or "mAP50(B)" in column or "mAP50(M)" in column]
    training_loss = float(pd.to_numeric(last[loss_columns], errors="coerce").sum()) if loss_columns else float("nan")
    mean_iou_proxy = float(pd.to_numeric(last[metric_columns], errors="coerce").mean()) if metric_columns else float("nan")
    return {"mean_iou": mean_iou_proxy, "training_loss": training_loss}


def yolo_label_file_to_masks(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read YOLO segmentation txt corrections into nucleus and parenchyme masks."""

    nuclei = np.zeros(image_shape, dtype=np.int32)
    parenchyme = np.zeros(image_shape, dtype=np.int32)
    if not label_path.exists():
        return nuclei, parenchyme

    next_nucleus = 1
    next_parenchyme = 1
    height, width = image_shape
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            continue
        points = np.column_stack(
            [
                np.clip(coords[:, 0] * width, 0, width - 1),
                np.clip(coords[:, 1] * height, 0, height - 1),
            ]
        ).astype(np.int32)
        if points.shape[0] < 3:
            continue
        polygon = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(polygon, [points], 1)
        if class_id == 0:
            nuclei[polygon > 0] = next_nucleus
            next_nucleus += 1
        elif class_id == 1:
            parenchyme[polygon > 0] = next_parenchyme
            next_parenchyme += 1

    return relabel_consecutive(nuclei), connected_parenchyme_instances(parenchyme)


def load_manual_corrections(image_path: Path, corrections_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Load manual corrections from corrections/ if present.

    Supported files:
    - corrections/<stem>.txt or corrections/<stem>_labels.txt as YOLO segmentation labels
    - corrections/<stem>_nuclei_mask.npy
    - corrections/<stem>_parenchyme_mask.npy
    """

    if not corrections_dir.exists():
        return None, None, False

    image_rgb = load_rgb_image(image_path)
    image_shape = image_rgb.shape[:2]
    nuclei_mask: np.ndarray | None = None
    parenchyme_mask: np.ndarray | None = None
    applied = False

    for label_path in (corrections_dir / f"{image_path.stem}.txt", corrections_dir / f"{image_path.stem}_labels.txt"):
        if label_path.exists():
            nuclei_from_txt, parenchyme_from_txt = yolo_label_file_to_masks(label_path, image_shape)
            if nuclei_from_txt.max() > 0:
                nuclei_mask = nuclei_from_txt
            if parenchyme_from_txt.max() > 0:
                parenchyme_mask = parenchyme_from_txt
            applied = True
            break

    nuclei_npy = corrections_dir / f"{image_path.stem}_nuclei_mask.npy"
    if nuclei_npy.exists():
        nuclei_mask = relabel_consecutive(np.load(nuclei_npy))
        applied = True
    parenchyme_npy = corrections_dir / f"{image_path.stem}_parenchyme_mask.npy"
    if parenchyme_npy.exists():
        parenchyme_mask = connected_parenchyme_instances(np.load(parenchyme_npy))
        applied = True
    return nuclei_mask, parenchyme_mask, applied


def apply_manual_corrections(bundle: MaskBundle, corrections_dir: Path) -> MaskBundle:
    nuclei, parenchyme, applied = load_manual_corrections(bundle.image_path, corrections_dir)
    if not applied:
        return bundle
    return MaskBundle(
        image_path=bundle.image_path,
        nuclei_mask=bundle.nuclei_mask if nuclei is None else nuclei,
        parenchyme_mask=bundle.parenchyme_mask if parenchyme is None else parenchyme,
        confidence_values=[1.0, *bundle.confidence_values],
        confidence_map=bundle.confidence_map,
        source=f"{bundle.source}+manual_correction",
        manual_correction_applied=True,
    )


def run_yolo_inference(
    image_paths: list[Path],
    weights_path: Path,
    config: ActiveLearningConfig,
) -> list[MaskBundle]:
    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError("Cannot run YOLO inference without trained weights.")
    if config.dry_run:
        bundles: list[MaskBundle] = []
        for image_path in image_paths:
            image_rgb = load_rgb_image(image_path)
            empty = np.zeros(image_rgb.shape[:2], dtype=np.int32)
            bundles.append(MaskBundle(image_path, empty, empty, [0.0], None, source="dry_run_yolo"))
        return bundles

    YOLO = import_ultralytics_yolo()
    model = YOLO(str(weights_path))
    log(f"Running YOLO inference with {weights_path}")
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=config.yolo_conf_threshold,
        imgsz=config.imgsz,
        verbose=False,
        stream=False,
    )

    bundles: list[MaskBundle] = []
    for result in results:
        image_path = Path(result.path)
        image_rgb = load_rgb_image(image_path)
        height, width = image_rgb.shape[:2]
        nuclei = np.zeros((height, width), dtype=np.int32)
        parenchyme = np.zeros((height, width), dtype=np.int32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        confidence_values: list[float] = []
        next_nucleus = 1
        next_parenchyme = 1

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy().astype(float)
            for mask, class_id, confidence in zip(masks, classes, confidences, strict=False):
                resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) >= 0.50
                if not np.any(resized):
                    continue
                confidence_values.append(float(confidence))
                confidence_map[resized] = np.maximum(confidence_map[resized], float(confidence))
                if class_id == 0:
                    nuclei[resized] = next_nucleus
                    next_nucleus += 1
                elif class_id == 1:
                    parenchyme[resized] = next_parenchyme
                    next_parenchyme += 1

        bundles.append(
            MaskBundle(
                image_path=image_path,
                nuclei_mask=relabel_consecutive(nuclei),
                parenchyme_mask=connected_parenchyme_instances(parenchyme),
                confidence_values=confidence_values or [0.0],
                confidence_map=confidence_map,
                source=f"yolo_round_weights:{weights_path}",
            )
        )
    return bundles


def evaluate_bundles(
    bundles: list[MaskBundle],
    round_number: int,
    config: ActiveLearningConfig,
    qc_config: BiologicalQCConfig,
    previous_parenchyme_fraction: float | None,
    previous_parenchyme_masks: dict[str, np.ndarray] | None,
) -> tuple[list[MaskBundle], list[BiologicalQCResult], dict[str, Any]]:
    """QC each pseudo-label bundle and keep only safe labels for training."""

    qc_dir = config.output_dir / f"qc_round_{round_number}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[MaskBundle] = []
    qc_results: list[BiologicalQCResult] = []
    queue_rows: list[dict[str, Any]] = []

    for bundle in bundles:
        corrected = apply_manual_corrections(bundle, config.corrections_dir)
        image_rgb = load_rgb_image(corrected.image_path)
        previous_mask = None if previous_parenchyme_masks is None else previous_parenchyme_masks.get(corrected.image_path.stem)
        result = morphology_metrics(
            image_rgb,
            corrected.nuclei_mask,
            corrected.parenchyme_mask,
            confidence_values=corrected.confidence_values,
            previous_parenchyme_fraction=previous_parenchyme_fraction,
            config=qc_config,
        )
        qc_results.append(result)
        save_masks(config.output_dir / f"round_{round_number}", corrected)
        overlay_paths = export_qc_outputs(
            image_rgb,
            corrected.nuclei_mask,
            corrected.parenchyme_mask,
            result,
            qc_dir,
            stem=corrected.image_path.stem,
            confidence_map=corrected.confidence_map,
            previous_parenchyme_mask=previous_mask,
        )

        status = "accepted" if result.accepted_for_training or corrected.manual_correction_applied else "rejected"
        if result.rejection_reasons and not corrected.manual_correction_applied:
            status = "uncertain" if result.metrics["biological_score"] >= qc_config.min_biological_score * 0.75 else "rejected"
        if status == "accepted":
            accepted.append(corrected)

        queue_rows.append(
            {
                "round_number": round_number,
                "image_path": str(corrected.image_path),
                "status": status,
                "source": corrected.source,
                "manual_correction_applied": corrected.manual_correction_applied,
                "rejection_reasons": ";".join(result.rejection_reasons),
                "biological_score": result.metrics["biological_score"],
                "fragmentation_score": result.metrics["fragmentation_score"],
                "nuclei_outside_parenchyme": result.metrics["nuclei_outside_parenchyme"],
                "parenchyme_fraction": result.metrics["parenchyme_fraction"],
                "confidence_mean": result.metrics["confidence_mean"],
                "qc_panel": str(overlay_paths["morphology_qc_panel"]),
                "uncertain_overlay": str(overlay_paths["uncertain_region_overlay"]),
                "rejected_nuclei_overlay": str(overlay_paths["rejected_nuclei_overlay"]),
            }
        )

    queue = pd.DataFrame(queue_rows)
    queue.to_csv(qc_dir / "manual_review_queue.csv", index=False)
    write_json(qc_dir / "manual_review_queue.json", queue_rows)
    summary = {
        "accepted_images": int((queue["status"] == "accepted").sum()) if not queue.empty else 0,
        "uncertain_images": int((queue["status"] == "uncertain").sum()) if not queue.empty else 0,
        "rejected_images": int((queue["status"] == "rejected").sum()) if not queue.empty else 0,
    }
    return accepted, qc_results, summary


def pseudo_label_counts(bundles: list[MaskBundle]) -> int:
    count = 0
    for bundle in bundles:
        count += int(np.max(bundle.nuclei_mask))
        count += int(np.max(connected_parenchyme_instances(bundle.parenchyme_mask)))
    return count


def append_training_log(config: ActiveLearningConfig, row: dict[str, Any]) -> Path:
    log_path = config.output_dir / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(log_path, index=False)
    return log_path


def copy_best_model(weights_path: Path | None, config: ActiveLearningConfig) -> Path | None:
    if weights_path is None or not weights_path.exists():
        return None
    target = config.runs_dir / "cell_segmenter" / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, target)
    return target


def round_log_row(
    round_number: int,
    aggregate: dict[str, float | int],
    training_metrics: dict[str, float],
    accepted_count: int,
    rejected_count: int,
    review_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round_number": round_number,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "nuclei_count": int(aggregate.get("nuclei_count", 0)),
        "nuclei_density": float(aggregate.get("nuclei_density", 0.0)),
        "fragmentation_score": float(aggregate.get("fragmentation_score", 0.0)),
        "hole_fraction_mean": float(aggregate.get("hole_fraction_mean", 0.0)),
        "parenchyme_fraction": float(aggregate.get("parenchyme_fraction", 0.0)),
        "nuclei_outside_parenchyme": int(aggregate.get("nuclei_outside_parenchyme", 0)),
        "mean_iou": training_metrics.get("mean_iou", float("nan")),
        "training_loss": training_metrics.get("training_loss", float("nan")),
        "accepted_pseudo_labels": int(accepted_count),
        "rejected_pseudo_labels": int(rejected_count),
        "confidence_mean": float(aggregate.get("confidence_mean", 0.0)),
        "biological_score": float(aggregate.get("biological_score", 0.0)),
        "accepted_images": int(review_summary.get("accepted_images", 0)),
        "uncertain_images": int(review_summary.get("uncertain_images", 0)),
        "rejected_images": int(review_summary.get("rejected_images", 0)),
    }


def should_stop(history: list[dict[str, Any]], min_improvement: float, patience: int) -> bool:
    if len(history) < patience + 1:
        return False
    recent = history[-(patience + 1) :]
    improvements: list[float] = []
    for previous, current in zip(recent, recent[1:], strict=False):
        improvements.append(float(current["biological_score"]) - float(previous["biological_score"]))
    return all(improvement < min_improvement for improvement in improvements)


def run_active_learning_loop(config: ActiveLearningConfig) -> dict[str, Any]:
    config = resolve_config_paths(config)
    qc_config = BiologicalQCConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(config.sample_dir)
    log(f"Found {len(image_paths)} sample image(s).")
    write_json(config.output_dir / "active_learning_config.json", {"active_learning": asdict(config), "biological_qc": asdict(qc_config)})

    previous_weights: Path | None = None
    best_weights: Path | None = None
    best_score = -np.inf
    previous_parenchyme_fraction: float | None = None
    previous_parenchyme_masks: dict[str, np.ndarray] | None = None
    history: list[dict[str, Any]] = []

    for round_number in range(config.max_rounds):
        log(f"Starting active-learning round {round_number}")
        if round_number == 0:
            proposed_bundles = run_initial_cellpose_round(image_paths, config)
        else:
            if previous_weights is None:
                log("No previous YOLO weights available; stopping before inference.")
                break
            proposed_bundles = run_yolo_inference(image_paths, previous_weights, config)

        accepted_bundles, qc_results, review_summary = evaluate_bundles(
            proposed_bundles,
            round_number,
            config,
            qc_config,
            previous_parenchyme_fraction,
            previous_parenchyme_masks,
        )
        aggregate = aggregate_qc_results(qc_results)
        accepted_count = pseudo_label_counts(accepted_bundles)
        rejected_count = max(pseudo_label_counts(proposed_bundles) - accepted_count, 0)

        if not accepted_bundles:
            log("No pseudo-labels passed biological QC. Exported review queue; stopping to avoid confirmation bias.")
            training_metrics = {"mean_iou": float("nan"), "training_loss": float("nan")}
            row = round_log_row(round_number, aggregate, training_metrics, accepted_count, rejected_count, review_summary)
            append_training_log(config, row)
            history.append(row)
            break

        dataset_yaml, yolo_label_count = build_yolo_dataset(accepted_bundles, config.dataset_dir, round_number, config)
        log(f"Round {round_number}: accepted {accepted_count} pseudo-label instances, wrote {yolo_label_count} YOLO polygons.")
        current_weights, training_metrics = train_yolo_model(dataset_yaml, round_number, previous_weights, config)
        previous_weights = current_weights

        row = round_log_row(round_number, aggregate, training_metrics, accepted_count, rejected_count, review_summary)
        append_training_log(config, row)
        history.append(row)
        write_json(config.output_dir / "active_learning_history.json", history)

        biological_score = float(row["biological_score"])
        if current_weights is not None and biological_score > best_score:
            best_score = biological_score
            best_weights = current_weights
            copied = copy_best_model(best_weights, config)
            if copied:
                log(f"New best biological model copied to {copied}")

        previous_parenchyme_fraction = float(aggregate.get("parenchyme_fraction", 0.0))
        previous_parenchyme_masks = {
            bundle.image_path.stem: qc_result.parenchyme_mask
            for bundle, qc_result in zip(proposed_bundles, qc_results, strict=False)
        }

        if round_number > 0 and should_stop(history, config.min_improvement, config.convergence_patience):
            log(
                f"Stopping: biological score improvement < {config.min_improvement:.3f} "
                f"for {config.convergence_patience} rounds."
            )
            break

    summary = {
        "rounds_completed": len(history),
        "best_biological_score": best_score if np.isfinite(best_score) else None,
        "best_weights": str(best_weights) if best_weights else None,
        "deployed_best_model": str(config.runs_dir / "cell_segmenter" / "weights" / "best.pt"),
        "training_log": str(config.output_dir / "training_log.csv"),
    }
    write_json(config.output_dir / "active_learning_summary.json", summary)
    log(f"Active-learning loop finished: {summary}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Biologically constrained adrenal H&E active-learning loop.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--sample-dir", type=Path, default=Path("sample"))
    parser.add_argument("--corrections-dir", type=Path, default=Path("corrections"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/active_learning_datasets"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--convergence-patience", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--cell-diameter", type=float, default=50.0)
    parser.add_argument("--nuclei-diameter", type=float, default=25.0)
    parser.add_argument("--flow-threshold", type=float, default=0.35)
    parser.add_argument("--cellprob-threshold", type=float, default=-1.5)
    parser.add_argument("--yolo-base-model", default="yolov8n-seg.pt")
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.35)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cellpose-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create empty labels and skip expensive Cellpose/YOLO work.")
    parser.add_argument("--skip-training", action="store_true", help="Run labeling/QC/dataset export but do not train YOLO.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ActiveLearningConfig(
        project_root=args.project_root,
        sample_dir=args.sample_dir,
        corrections_dir=args.corrections_dir,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        runs_dir=args.runs_dir,
        max_rounds=args.max_rounds,
        min_improvement=args.min_improvement,
        convergence_patience=args.convergence_patience,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cell_diameter=args.cell_diameter,
        nuclei_diameter=args.nuclei_diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        yolo_base_model=args.yolo_base_model,
        yolo_conf_threshold=args.yolo_conf_threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        cellpose_gpu=args.cellpose_gpu,
        dry_run=args.dry_run,
        skip_training=args.skip_training,
    )
    try:
        run_active_learning_loop(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
