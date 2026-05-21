"""Biology-aware morphology QC for adrenal H&E active learning.

The checks here intentionally optimize for biologically plausible morphometry,
not mask smoothness or IoU alone. They are conservative about pseudo-label reuse:
ambiguous or morphologically implausible labels should go to manual review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import color, io, measure, morphology, segmentation


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class BiologicalQCConfig:
    """Thresholds for adrenal clear-cell pathology QC."""

    tissue_min_saturation: float = 0.025
    tissue_max_brightness: float = 0.97
    parenchyme_probability_threshold: float = 0.28
    parenchyme_clear_min_brightness: float = 0.60
    parenchyme_clear_max_saturation: float = 0.50
    parenchyme_clear_max_eosin: float = 0.62
    stroma_min_eosin: float = 0.62
    stroma_min_saturation: float = 0.18
    nucleus_min_area_px: int = 18
    nucleus_max_area_px: int = 2500
    nucleus_min_solidity: float = 0.35
    nucleus_max_hole_fraction: float = 0.22
    nucleus_max_boundary_roughness: float = 4.5
    nucleus_close_pair_distance_px: float = 8.0
    parenchyme_gate_distance_px: float = 12.0
    min_nuclei_count: int = 1
    min_parenchyme_fraction: float = 0.03
    max_fragmentation_score: float = 0.35
    max_hole_fraction_mean: float = 0.20
    max_nuclei_outside_parenchyme_fraction: float = 0.20
    max_isolated_nuclei_fraction: float = 0.30
    min_cytoplasm_continuity_score: float = 0.20
    min_topology_consistency_score: float = 0.45
    min_biological_score: float = 0.55
    min_confidence_mean: float = 0.65
    min_confidence_low_quantile: float = 0.45
    parenchyme_recall_drop_tolerance: float = 0.15
    disconnected_region_warning_count: int = 20


@dataclass
class BiologicalQCResult:
    """Structured QC output for one image or round aggregate."""

    metrics: dict[str, float | int | str | bool]
    nucleus_table: pd.DataFrame
    rejected_nuclei_mask: np.ndarray
    uncertain_mask: np.ndarray
    parenchyme_probability: np.ndarray
    parenchyme_mask: np.ndarray
    stroma_mask: np.ndarray
    rejection_reasons: list[str]
    accepted_for_training: bool


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    low = float(np.nanmin(image_float))
    high = float(np.nanmax(image_float))
    if high <= low:
        return np.zeros(image_float.shape, dtype=np.uint8)
    return np.clip((image_float - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return normalize_to_uint8(image)


def robust_unit(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low = float(np.percentile(image, low_pct))
    high = float(np.percentile(image, high_pct))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def inverse_score(value: np.ndarray, low_good: float, high_bad: float) -> np.ndarray:
    if high_bad <= low_good:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(1.0 - ((value - low_good) / (high_bad - low_good)), 0.0, 1.0)


def he_feature_maps(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgb = image_rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    hsv = color.rgb2hsv(rgb)
    hed = color.separate_stains((rgb * 255).astype(np.uint8), color.hed_from_rgb)
    return {
        "rgb": rgb,
        "hematoxylin": robust_unit(hed[:, :, 0]),
        "eosin": robust_unit(hed[:, :, 1]),
        "brightness": rgb.mean(axis=2),
        "saturation": hsv[:, :, 1],
    }


def clean_binary(mask: np.ndarray, *, min_area: int = 0, closing_px: int = 0, hole_area_px: int = 0) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if closing_px > 0:
        out = morphology.closing(out, morphology.disk(closing_px))
    if hole_area_px > 0:
        try:
            out = morphology.remove_small_holes(out, area_threshold=hole_area_px)
        except TypeError:
            out = morphology.remove_small_holes(out, max_size=hole_area_px)
    if min_area > 0:
        out = morphology.remove_small_objects(out, min_size=min_area)
    return np.asarray(out, dtype=bool)


def relabel_consecutive(label_mask: np.ndarray) -> np.ndarray:
    label_mask = np.asarray(label_mask, dtype=np.int32)
    output = np.zeros_like(label_mask, dtype=np.int32)
    next_label = 1
    for region in measure.regionprops(label_mask):
        output[label_mask == region.label] = next_label
        next_label += 1
    return output


def tissue_mask_from_features(features: dict[str, np.ndarray], config: BiologicalQCConfig) -> np.ndarray:
    mask = (features["saturation"] >= config.tissue_min_saturation) | (
        features["brightness"] <= config.tissue_max_brightness
    )
    return clean_binary(mask, min_area=128, closing_px=2, hole_area_px=512)


def clear_cell_probability(features: dict[str, np.ndarray], config: BiologicalQCConfig) -> np.ndarray:
    brightness = np.clip((features["brightness"] - config.parenchyme_clear_min_brightness) / 0.32, 0.0, 1.0)
    low_saturation = inverse_score(features["saturation"], config.parenchyme_clear_max_saturation, 0.90)
    low_eosin = inverse_score(features["eosin"], config.parenchyme_clear_max_eosin, 0.95)
    low_h = inverse_score(features["hematoxylin"], 0.45, 0.90)
    return np.clip(0.38 * brightness + 0.27 * low_saturation + 0.22 * low_eosin + 0.13 * low_h, 0.0, 1.0)


def estimate_parenchyme_context(
    image_rgb: np.ndarray,
    nuclei_mask: np.ndarray | None,
    predicted_parenchyme_mask: np.ndarray | None,
    config: BiologicalQCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a high-recall parenchyme context for gating nuclei and pseudo-labels."""

    features = he_feature_maps(image_rgb)
    tissue = tissue_mask_from_features(features, config)
    clear_score = clear_cell_probability(features, config)
    nuclei_binary = np.zeros(tissue.shape, dtype=bool) if nuclei_mask is None else np.asarray(nuclei_mask) > 0
    nuclei_context = robust_unit(ndimage.gaussian_filter(nuclei_binary.astype(np.float32), sigma=18), high_pct=99.5)

    stroma_mask = (
        tissue
        & (features["eosin"] >= config.stroma_min_eosin)
        & (features["saturation"] >= config.stroma_min_saturation)
        & (clear_score < 0.35)
    )
    stroma_mask = clean_binary(stroma_mask, min_area=96, closing_px=1)
    predicted = np.zeros(tissue.shape, dtype=np.float32)
    if predicted_parenchyme_mask is not None:
        predicted = (np.asarray(predicted_parenchyme_mask) > 0).astype(np.float32)

    continuity = robust_unit(ndimage.gaussian_filter((tissue & ~stroma_mask).astype(np.float32), sigma=10))
    probability = np.clip(
        0.36 * predicted
        + 0.26 * clear_score
        + 0.18 * nuclei_context
        + 0.12 * continuity
        + 0.08 * (1.0 - stroma_mask.astype(np.float32)),
        0.0,
        1.0,
    )
    probability[~tissue] = 0.0
    parenchyme_mask = (
        tissue
        & ~stroma_mask
        & ((probability >= config.parenchyme_probability_threshold) | (clear_score >= 0.62) | (nuclei_context >= 0.12))
    )
    parenchyme_mask = clean_binary(parenchyme_mask, min_area=128, closing_px=5, hole_area_px=2048)
    uncertain_mask = tissue & ~parenchyme_mask & ~stroma_mask & (probability >= config.parenchyme_probability_threshold * 0.7)
    return probability.astype(np.float32), parenchyme_mask, stroma_mask, uncertain_mask


def _hole_fraction(object_pixels: np.ndarray) -> float:
    filled = ndimage.binary_fill_holes(object_pixels)
    filled_area = int(np.count_nonzero(filled))
    if filled_area == 0:
        return 0.0
    return float((filled_area - int(np.count_nonzero(object_pixels))) / filled_area)


def _region_circularity(region) -> float:
    if region.perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * region.area / (region.perimeter**2))


def _boundary_roughness(region) -> float:
    if region.area <= 0:
        return 0.0
    ideal_perimeter = 2.0 * np.sqrt(np.pi * region.area)
    if ideal_perimeter <= 0:
        return 0.0
    return float(region.perimeter / ideal_perimeter)


def measure_nuclei_qc(
    nuclei_mask: np.ndarray,
    parenchyme_mask: np.ndarray,
    stroma_mask: np.ndarray,
    config: BiologicalQCConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    nuclei_mask = relabel_consecutive(nuclei_mask)
    parenchyme_gate = morphology.dilation(parenchyme_mask, morphology.disk(int(config.parenchyme_gate_distance_px)))
    rejected = np.zeros_like(nuclei_mask, dtype=np.int32)
    rows: list[dict[str, Any]] = []

    close_labels: set[int] = set()
    regions = list(measure.regionprops(nuclei_mask))
    centroids = np.array([region.centroid for region in regions], dtype=float)
    if len(centroids) > 1:
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(r=config.nucleus_close_pair_distance_px)
        for left, right in pairs:
            close_labels.add(int(regions[left].label))
            close_labels.add(int(regions[right].label))

    for region in regions:
        pixels = nuclei_mask == region.label
        hole_fraction = _hole_fraction(pixels)
        boundary_roughness = _boundary_roughness(region)
        parenchyme_overlap = float(np.mean(parenchyme_gate[pixels])) if region.area else 0.0
        stroma_overlap = float(np.mean(stroma_mask[pixels])) if region.area else 0.0
        circularity = _region_circularity(region)

        reasons: list[str] = []
        if region.area < config.nucleus_min_area_px:
            reasons.append("tiny_fragment")
        if region.area > config.nucleus_max_area_px:
            reasons.append("implausibly_large")
        if region.solidity < config.nucleus_min_solidity:
            reasons.append("low_solidity")
        if hole_fraction > config.nucleus_max_hole_fraction:
            reasons.append("porous_or_ring_like")
        if boundary_roughness > config.nucleus_max_boundary_roughness:
            reasons.append("rough_boundary")
        if parenchyme_overlap <= 0.0:
            reasons.append("outside_parenchyme")
        if stroma_overlap > 0.50:
            reasons.append("inside_stroma")
        if int(region.label) in close_labels and region.area < config.nucleus_min_area_px * 3:
            reasons.append("close_small_fragment")

        if reasons:
            rejected[pixels] = int(region.label)

        rows.append(
            {
                "nucleus_id": int(region.label),
                "area": float(region.area),
                "perimeter": float(region.perimeter),
                "solidity": float(region.solidity),
                "eccentricity": float(region.eccentricity),
                "circularity": circularity,
                "hole_fraction": hole_fraction,
                "boundary_roughness": boundary_roughness,
                "centroid_x": float(region.centroid[1]),
                "centroid_y": float(region.centroid[0]),
                "parenchyme_overlap": parenchyme_overlap,
                "stroma_overlap": stroma_overlap,
                "is_close_fragment": bool(int(region.label) in close_labels),
                "qc_status": "rejected" if reasons else "accepted",
                "qc_reasons": ";".join(reasons),
            }
        )

    return pd.DataFrame(rows), relabel_consecutive(rejected)


def morphology_metrics(
    image_rgb: np.ndarray,
    nuclei_mask: np.ndarray,
    parenchyme_mask: np.ndarray | None,
    *,
    confidence_values: list[float] | np.ndarray | None = None,
    previous_parenchyme_fraction: float | None = None,
    config: BiologicalQCConfig | None = None,
) -> BiologicalQCResult:
    """Compute image-level biological QC and reject unsafe pseudo-labels."""

    config = config or BiologicalQCConfig()
    nuclei_mask = relabel_consecutive(nuclei_mask)
    probability, inferred_parenchyme, stroma_mask, uncertain_mask = estimate_parenchyme_context(
        image_rgb,
        nuclei_mask,
        parenchyme_mask,
        config,
    )
    nucleus_table, rejected_nuclei_mask = measure_nuclei_qc(nuclei_mask, inferred_parenchyme, stroma_mask, config)

    tissue = tissue_mask_from_features(he_feature_maps(image_rgb), config)
    tissue_area = max(int(np.count_nonzero(tissue)), 1)
    parenchyme_area = int(np.count_nonzero(inferred_parenchyme))
    stroma_area = int(np.count_nonzero(stroma_mask))
    parenchyme_fraction = float(parenchyme_area / tissue_area)
    stroma_fraction = float(stroma_area / tissue_area)

    parenchyme_components = measure.label(inferred_parenchyme)
    component_props = list(measure.regionprops(parenchyme_components))
    disconnected_count = len(component_props)
    largest_component_area = max((region.area for region in component_props), default=0.0)
    continuity_score = float(largest_component_area / max(parenchyme_area, 1))
    vacuolation_score = float(np.mean(clear_cell_probability(he_feature_maps(image_rgb), config)[inferred_parenchyme])) if parenchyme_area else 0.0

    nuclei_count = int(nuclei_mask.max())
    nuclei_density = float(nuclei_count / (tissue_area / 1_000_000.0))
    if nucleus_table.empty:
        hole_fraction_mean = 0.0
        outside_count = 0
        inside_stroma_count = 0
        isolated_count = 0
        fragmentation_score = 0.0
    else:
        hole_fraction_mean = float(nucleus_table["hole_fraction"].mean())
        outside_count = int((nucleus_table["parenchyme_overlap"] <= 0.0).sum())
        inside_stroma_count = int((nucleus_table["stroma_overlap"] > 0.50).sum())
        isolated_count = int((nucleus_table["parenchyme_overlap"] < 0.25).sum())
        small_fraction = float((nucleus_table["area"] < config.nucleus_min_area_px * 2).mean())
        porous_fraction = float((nucleus_table["hole_fraction"] > config.nucleus_max_hole_fraction).mean())
        low_solidity_fraction = float((nucleus_table["solidity"] < config.nucleus_min_solidity).mean())
        close_fragment_fraction = float(nucleus_table["is_close_fragment"].mean())
        fragmentation_score = float(
            np.clip(0.35 * small_fraction + 0.25 * porous_fraction + 0.20 * low_solidity_fraction + 0.20 * close_fragment_fraction, 0.0, 1.0)
        )

    outside_fraction = float(outside_count / max(nuclei_count, 1))
    isolated_fraction = float(isolated_count / max(nuclei_count, 1))
    topology_consistency = float(
        np.clip(
            0.35 * continuity_score
            + 0.25 * (1.0 - fragmentation_score)
            + 0.20 * (1.0 - outside_fraction)
            + 0.20 * (1.0 - isolated_fraction),
            0.0,
            1.0,
        )
    )
    nc_consistency = float(np.clip(1.0 - outside_fraction - 0.5 * isolated_fraction, 0.0, 1.0))

    confidence = np.asarray([] if confidence_values is None else confidence_values, dtype=np.float32)
    confidence_mean = float(np.mean(confidence)) if confidence.size else 1.0
    confidence_low_quantile = float(np.quantile(confidence, 0.10)) if confidence.size else 1.0

    parenchyme_recall_drop = 0.0
    if previous_parenchyme_fraction is not None:
        parenchyme_recall_drop = max(0.0, float(previous_parenchyme_fraction - parenchyme_fraction))

    biological_score = float(
        np.clip(
            0.24 * (1.0 - fragmentation_score)
            + 0.20 * topology_consistency
            + 0.18 * nc_consistency
            + 0.14 * continuity_score
            + 0.12 * min(parenchyme_fraction / max(config.min_parenchyme_fraction, 1e-6), 1.0)
            + 0.12 * confidence_mean,
            0.0,
            1.0,
        )
    )

    metrics: dict[str, float | int | str | bool] = {
        "nuclei_count": nuclei_count,
        "nuclei_density": nuclei_density,
        "nuclei_area_mean": float(nucleus_table["area"].mean()) if not nucleus_table.empty else 0.0,
        "nuclei_area_std": float(nucleus_table["area"].std(ddof=0)) if not nucleus_table.empty else 0.0,
        "nuclei_solidity_mean": float(nucleus_table["solidity"].mean()) if not nucleus_table.empty else 0.0,
        "hole_fraction_mean": hole_fraction_mean,
        "fragmentation_score": fragmentation_score,
        "nuclei_outside_parenchyme": outside_count,
        "nuclei_inside_stroma": inside_stroma_count,
        "isolated_nuclei_count": isolated_count,
        "parenchyme_fraction": parenchyme_fraction,
        "stroma_fraction": stroma_fraction,
        "cytoplasm_continuity_score": continuity_score,
        "disconnected_region_count": disconnected_count,
        "vacuolation_score": vacuolation_score,
        "nuclei_to_cytoplasm_consistency": nc_consistency,
        "neighborhood_consistency": float(1.0 - isolated_fraction),
        "topology_consistency": topology_consistency,
        "confidence_mean": confidence_mean,
        "confidence_low_quantile": confidence_low_quantile,
        "mean_iou": np.nan,
        "training_loss": np.nan,
        "parenchyme_recall_drop": parenchyme_recall_drop,
        "biological_score": biological_score,
    }

    reasons: list[str] = []
    if nuclei_count < config.min_nuclei_count:
        reasons.append("no_plausible_nuclei")
    if fragmentation_score > config.max_fragmentation_score:
        reasons.append("excessive_nuclei_fragmentation")
    if hole_fraction_mean > config.max_hole_fraction_mean:
        reasons.append("excessive_nuclear_holes")
    if outside_fraction > config.max_nuclei_outside_parenchyme_fraction:
        reasons.append("nuclei_outside_plausible_parenchyme")
    if isolated_fraction > config.max_isolated_nuclei_fraction:
        reasons.append("isolated_nuclei")
    if parenchyme_fraction < config.min_parenchyme_fraction:
        reasons.append("low_parenchyme_recall")
    if continuity_score < config.min_cytoplasm_continuity_score:
        reasons.append("poor_cytoplasm_continuity")
    if topology_consistency < config.min_topology_consistency_score:
        reasons.append("implausible_topology")
    if confidence_mean < config.min_confidence_mean:
        reasons.append("low_yolo_confidence")
    if confidence_low_quantile < config.min_confidence_low_quantile:
        reasons.append("weak_low_quantile_confidence")
    if parenchyme_recall_drop > config.parenchyme_recall_drop_tolerance:
        reasons.append("parenchyme_recall_decreased")
    if biological_score < config.min_biological_score:
        reasons.append("low_biological_score")

    return BiologicalQCResult(
        metrics=metrics,
        nucleus_table=nucleus_table,
        rejected_nuclei_mask=rejected_nuclei_mask,
        uncertain_mask=uncertain_mask,
        parenchyme_probability=probability,
        parenchyme_mask=inferred_parenchyme,
        stroma_mask=stroma_mask,
        rejection_reasons=reasons,
        accepted_for_training=len(reasons) == 0,
    )


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    alpha: float = 0.35,
    boundary: bool = False,
) -> np.ndarray:
    base = normalize_to_uint8(image_rgb).astype(np.float32)
    mask_bool = np.asarray(mask) > 0
    if boundary:
        mask_bool = segmentation.find_boundaries(mask_bool, mode="outer")
    color_arr = np.asarray(color_rgb, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def confidence_heatmap(image_rgb: np.ndarray, confidence_map: np.ndarray | None) -> np.ndarray:
    base = normalize_to_uint8(image_rgb)
    if confidence_map is None or not np.any(confidence_map):
        return base
    heat = normalize_to_uint8(confidence_map)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(0.55 * base.astype(np.float32) + 0.45 * heat.astype(np.float32), 0, 255).astype(np.uint8)


def write_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, normalize_to_uint8(image), check_contrast=False)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def morphology_qc_panel(image_rgb: np.ndarray, result: BiologicalQCResult) -> np.ndarray:
    width = max(900, image_rgb.shape[1])
    height = 520
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    title = "Adrenal H&E Biological QC"
    cv2.putText(panel, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 28, 40), 2, cv2.LINE_AA)
    status = "ACCEPTED" if result.accepted_for_training else "REVIEW / REJECT"
    status_color = (20, 130, 60) if result.accepted_for_training else (200, 65, 45)
    cv2.putText(panel, status, (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

    rows = [
        ("Biological score", result.metrics["biological_score"]),
        ("Nuclei count", result.metrics["nuclei_count"]),
        ("Nuclei density", result.metrics["nuclei_density"]),
        ("Fragmentation score", result.metrics["fragmentation_score"]),
        ("Hole fraction mean", result.metrics["hole_fraction_mean"]),
        ("Nuclei outside parenchyme", result.metrics["nuclei_outside_parenchyme"]),
        ("Parenchyme fraction", result.metrics["parenchyme_fraction"]),
        ("Stroma fraction", result.metrics["stroma_fraction"]),
        ("Cytoplasm continuity", result.metrics["cytoplasm_continuity_score"]),
        ("Disconnected regions", result.metrics["disconnected_region_count"]),
        ("N/C consistency", result.metrics["nuclei_to_cytoplasm_consistency"]),
        ("Confidence mean", result.metrics["confidence_mean"]),
    ]
    y = 132
    for label, value in rows:
        text_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        cv2.putText(panel, f"{label}: {text_value}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 45, 60), 1, cv2.LINE_AA)
        y += 29

    reasons = result.rejection_reasons or ["none"]
    cv2.putText(panel, "Rejection reasons:", (430, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (35, 45, 60), 2, cv2.LINE_AA)
    y = 166
    for reason in reasons[:10]:
        cv2.putText(panel, f"- {reason}", (430, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65, 72, 84), 1, cv2.LINE_AA)
        y += 27
    return panel


def export_qc_outputs(
    image_rgb: np.ndarray,
    nuclei_mask: np.ndarray,
    parenchyme_mask: np.ndarray | None,
    result: BiologicalQCResult,
    output_dir: str | Path,
    *,
    stem: str,
    confidence_map: np.ndarray | None = None,
    previous_parenchyme_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write pathology-review overlays for one image."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["raw_image"] = write_image(output_dir / f"{stem}_raw.png", image_rgb)
    paths["nuclei_overlay"] = write_image(
        output_dir / f"{stem}_nuclei_overlay.png",
        overlay_mask(image_rgb, nuclei_mask, (0, 110, 255), alpha=0.65, boundary=True),
    )
    paths["parenchyme_overlay"] = write_image(
        output_dir / f"{stem}_parenchyme_overlay.png",
        overlay_mask(image_rgb, result.parenchyme_mask, (0, 210, 90), alpha=0.30),
    )
    paths["rejected_nuclei_overlay"] = write_image(
        output_dir / f"{stem}_rejected_nuclei_overlay.png",
        overlay_mask(image_rgb, result.rejected_nuclei_mask, (255, 30, 30), alpha=0.75, boundary=True),
    )
    paths["uncertain_region_overlay"] = write_image(
        output_dir / f"{stem}_uncertain_region_overlay.png",
        overlay_mask(image_rgb, result.uncertain_mask, (255, 170, 0), alpha=0.38),
    )
    paths["morphology_qc_panel"] = write_image(output_dir / f"{stem}_morphology_qc_panel.png", morphology_qc_panel(image_rgb, result))
    paths["confidence_heatmap"] = write_image(output_dir / f"{stem}_confidence_heatmap.png", confidence_heatmap(image_rgb, confidence_map))
    paths["nuclei_fragmentation_map"] = write_image(
        output_dir / f"{stem}_nuclei_fragmentation_map.png",
        overlay_mask(image_rgb, result.rejected_nuclei_mask, (255, 0, 180), alpha=0.80, boundary=True),
    )
    paths["parenchyme_probability_map"] = write_image(
        output_dir / f"{stem}_parenchyme_probability_map.png",
        cv2.cvtColor(cv2.applyColorMap(normalize_to_uint8(result.parenchyme_probability), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB),
    )
    if previous_parenchyme_mask is not None:
        before = overlay_mask(image_rgb, previous_parenchyme_mask, (0, 120, 255), alpha=0.28)
        after = overlay_mask(image_rgb, result.parenchyme_mask, (0, 210, 90), alpha=0.28)
        paths["before_after_comparison"] = write_image(output_dir / f"{stem}_before_after_comparison.png", np.concatenate([before, after], axis=1))

    result.nucleus_table.to_csv(output_dir / f"{stem}_nucleus_qc.csv", index=False)
    write_json(
        output_dir / f"{stem}_qc_metrics.json",
        {
            "metrics": result.metrics,
            "accepted_for_training": result.accepted_for_training,
            "rejection_reasons": result.rejection_reasons,
            "config": asdict(BiologicalQCConfig()),
        },
    )
    return paths


def aggregate_qc_results(results: list[BiologicalQCResult]) -> dict[str, float | int]:
    if not results:
        return {
            "nuclei_count": 0,
            "nuclei_density": 0.0,
            "fragmentation_score": 1.0,
            "hole_fraction_mean": 1.0,
            "parenchyme_fraction": 0.0,
            "nuclei_outside_parenchyme": 0,
            "confidence_mean": 0.0,
            "biological_score": 0.0,
        }

    metric_rows = [result.metrics for result in results]
    df = pd.DataFrame(metric_rows)
    aggregate: dict[str, float | int] = {}
    summed = {"nuclei_count", "nuclei_outside_parenchyme", "nuclei_inside_stroma", "isolated_nuclei_count", "disconnected_region_count"}
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if column in summed:
            aggregate[column] = int(numeric.sum())
        else:
            aggregate[column] = float(numeric.mean())
    return aggregate
