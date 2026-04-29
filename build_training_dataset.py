# build_training_dataset.py
"""
Converts Cellpose masks + manual corrections into a YOLOv8 SEGMENTATION dataset.

Green annotations (Missed Cells) → new polygon labels from freedraw/circle.
Red annotations (False Positives) → remove overlapping predicted objects.
Remaining Cellpose detections → contour polygons extracted from mask.

Label format per line:  class x1 y1 x2 y2 x3 y3 ...  (normalized 0-1)
"""

import json
import math
import shutil
import random
from pathlib import Path
import numpy as np
from skimage import io, measure

# =====================================================================
# CONFIGURATION
# =====================================================================
IMAGE_DIR = Path("sample")
MASK_DIR = Path("output")
CORRECTIONS_DIR = Path("corrections")
DATASET_DIR = Path("datasets")
TRAIN_RATIO = 0.8
CELL_CLASS_ID = 0
MIN_POLYGON_POINTS = 3
CIRCLE_APPROX_POINTS = 16  # points for circle approximation from point annotations


def load_corrections(json_path):
    """Load manual corrections JSON exported from the Streamlit canvas."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("annotations", [])


def clip_coord(v):
    """Clip a normalized coordinate to [0, 1]."""
    return max(0.0, min(1.0, v))


def point_to_circle_polygon(cx, cy, img_w, img_h, radius_px=40, n_points=CIRCLE_APPROX_POINTS):
    """
    Convert a point annotation into a circular polygon (normalized coords).
    Returns list of (x_norm, y_norm) tuples.
    """
    polygon = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        px = cx + radius_px * math.cos(angle)
        py = cy + radius_px * math.sin(angle)
        polygon.append((clip_coord(px / img_w), clip_coord(py / img_h)))
    return polygon


def freedraw_to_polygon(orig_points, img_w, img_h):
    """
    Convert freedraw orig_points to a normalized polygon.
    Subsamples if too many points (>100) for YOLO efficiency.
    Returns list of (x_norm, y_norm) tuples, or None if invalid.
    """
    if len(orig_points) < MIN_POLYGON_POINTS:
        return None

    points = [(p["x"], p["y"]) for p in orig_points]

    # Subsample if too dense
    if len(points) > 100:
        step = len(points) / 100
        indices = [int(i * step) for i in range(100)]
        points = [points[i] for i in indices]

    polygon = [(clip_coord(x / img_w), clip_coord(y / img_h)) for x, y in points]
    return polygon


def mask_region_to_polygon(binary_mask, img_w, img_h, max_points=80):
    """
    Extract the largest contour from a binary mask region and return
    as a normalized polygon. Returns None if contour is too small.
    """
    contours = measure.find_contours(binary_mask, 0.5)
    if not contours:
        return None

    # Take the longest contour
    contour = max(contours, key=len)
    if len(contour) < MIN_POLYGON_POINTS:
        return None

    # Subsample if needed
    if len(contour) > max_points:
        step = len(contour) / max_points
        indices = [int(i * step) for i in range(max_points)]
        contour = contour[indices]

    # contour is (row, col) = (y, x) format from skimage
    polygon = [(clip_coord(float(c[1]) / img_w), clip_coord(float(c[0]) / img_h)) for c in contour]
    return polygon


def polygon_to_yolo_line(class_id, polygon):
    """
    Format a polygon as a YOLO segmentation label line:
    class x1 y1 x2 y2 x3 y3 ...
    """
    parts = [str(class_id)]
    for x, y in polygon:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)


def get_fp_centers_normalized(fp_annotations, img_w, img_h):
    """Get normalized center coordinates of FP annotations."""
    centers = []
    for annot in fp_annotations:
        if annot["type"] == "point":
            centers.append((annot["orig_x"] / img_w, annot["orig_y"] / img_h))
        elif annot["type"] == "freedraw":
            pts = annot["orig_points"]
            cx = np.mean([p["x"] for p in pts]) / img_w
            cy = np.mean([p["y"] for p in pts]) / img_h
            centers.append((cx, cy))
    return centers


def region_overlaps_fp(prop, fp_centers, img_w, img_h):
    """
    Check if a mask region's bounding box contains any FP center.
    """
    r0, c0, r1, c1 = prop.bbox
    # Normalize bbox
    nx0, ny0 = c0 / img_w, r0 / img_h
    nx1, ny1 = c1 / img_w, r1 / img_h

    for fx, fy in fp_centers:
        if nx0 <= fx <= nx1 and ny0 <= fy <= ny1:
            return True
    return False


def build_dataset():
    """Main dataset builder — segmentation version."""
    print("=" * 60)
    print("Building YOLOv8 SEGMENTATION Dataset")
    print("=" * 60)

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Find images
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]:
        image_files.extend(IMAGE_DIR.glob(ext))

    if not image_files:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        return

    # Load corrections
    correction_files = list(CORRECTIONS_DIR.glob("*.json")) if CORRECTIONS_DIR.exists() else []
    corrections_map = {}
    for cf in correction_files:
        corrections_map[cf.stem] = load_corrections(cf)

    all_samples = []

    for img_path in image_files:
        img = io.imread(img_path)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        img_h, img_w = img.shape[:2]
        stem = img_path.stem

        label_lines = []

        # --- 1. Extract polygons from Cellpose mask ---
        mask_npy = MASK_DIR / f"cell_mask_{stem}.npy"
        mask = None
        if mask_npy.exists():
            mask = np.load(mask_npy)

        # --- 2. Get corrections ---
        annots = corrections_map.get(stem, corrections_map.get("manual_corrections", []))
        fp_annots = [a for a in annots if a.get("label") == "False Positive"]
        missed_annots = [a for a in annots if a.get("label") == "Missed Cell"]
        fp_centers = get_fp_centers_normalized(fp_annots, img_w, img_h)

        # --- 3. Convert Cellpose mask regions to polygons (skip FP-overlapping ones) ---
        mask_polygon_count = 0
        if mask is not None:
            props = measure.regionprops(mask)
            for prop in props:
                if region_overlaps_fp(prop, fp_centers, img_w, img_h):
                    continue  # Skip — this was marked as FP

                # Extract binary mask for this region
                region_mask = (mask == prop.label).astype(np.uint8)
                polygon = mask_region_to_polygon(region_mask, img_w, img_h)
                if polygon:
                    label_lines.append(polygon_to_yolo_line(CELL_CLASS_ID, polygon))
                    mask_polygon_count += 1

        # --- 4. Convert missed cell annotations to polygons ---
        missed_count = 0
        for annot in missed_annots:
            if annot["type"] == "point":
                polygon = point_to_circle_polygon(
                    annot["orig_x"], annot["orig_y"], img_w, img_h
                )
                label_lines.append(polygon_to_yolo_line(CELL_CLASS_ID, polygon))
                missed_count += 1
            elif annot["type"] == "freedraw":
                polygon = freedraw_to_polygon(annot["orig_points"], img_w, img_h)
                if polygon:
                    label_lines.append(polygon_to_yolo_line(CELL_CLASS_ID, polygon))
                    missed_count += 1

        all_samples.append((img_path, label_lines))
        print(f"  {stem}: {len(label_lines)} polygons "
              f"({mask_polygon_count} from mask, {missed_count} added, {len(fp_annots)} removed)")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx] if split_idx > 0 else all_samples
    val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]

    if len(all_samples) == 1:
        train_samples = all_samples
        val_samples = all_samples

    # Write files
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_path, label_lines in samples:
            shutil.copy2(img_path, DATASET_DIR / "images" / split / img_path.name)
            label_file = DATASET_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines) + "\n" if label_lines else "")

    # Write data.yaml
    yaml_content = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
  0: cell
"""
    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    print(f"\nSegmentation dataset built: {len(train_samples)} train, {len(val_samples)} val")
    print(f"Output: {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    build_dataset()
