# build_training_dataset.py
"""
Converts Cellpose masks + manual corrections into a YOLOv8 detection dataset.

Green annotations (Missed Cells) → new bounding box labels.
Red annotations (False Positives) → remove overlapping predicted objects.
Remaining Cellpose detections → kept as positive labels.
"""

import json
import shutil
import random
from pathlib import Path
import numpy as np
from skimage import io, measure

# =====================================================================
# CONFIGURATION
# =====================================================================
IMAGE_DIR = Path("sample")
MASK_DIR = Path("output")           # contains cell masks from pipeline
CORRECTIONS_DIR = Path("corrections")  # manual_corrections.json files
DATASET_DIR = Path("datasets")
TRAIN_RATIO = 0.8
CELL_CLASS_ID = 0  # single class


def load_corrections(json_path):
    """Load manual corrections JSON exported from the Streamlit canvas."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("annotations", [])


def annotation_to_bbox(annot, img_w, img_h):
    """
    Convert a single annotation (point or freedraw) to a bounding box
    in YOLO normalized format: (x_center, y_center, width, height).
    Coordinates in the JSON are already in original image space.
    """
    if annot["type"] == "point":
        cx, cy = annot["orig_x"], annot["orig_y"]
        # Estimate a reasonable cell-sized box (40px radius)
        radius = 40
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(img_w, cx + radius)
        y2 = min(img_h, cy + radius)
    elif annot["type"] == "freedraw":
        pts = annot["orig_points"]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(img_w, max(xs)), min(img_h, max(ys))
        # Ensure minimum box size
        if (x2 - x1) < 20:
            cx = (x1 + x2) / 2
            x1, x2 = max(0, cx - 20), min(img_w, cx + 20)
        if (y2 - y1) < 20:
            cy = (y1 + y2) / 2
            y1, y2 = max(0, cy - 20), min(img_h, cy + 20)
    else:
        return None

    bw = x2 - x1
    bh = y2 - y1
    cx_norm = ((x1 + x2) / 2) / img_w
    cy_norm = ((y1 + y2) / 2) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h

    return (cx_norm, cy_norm, w_norm, h_norm)


def mask_to_bboxes(mask, img_w, img_h):
    """
    Extract bounding boxes from a labeled instance mask.
    Returns list of (x_center, y_center, width, height) normalized.
    """
    props = measure.regionprops(mask)
    boxes = []
    for p in props:
        r0, c0, r1, c1 = p.bbox
        cx = ((c0 + c1) / 2) / img_w
        cy = ((r0 + r1) / 2) / img_h
        w = (c1 - c0) / img_w
        h = (r1 - r0) / img_h
        boxes.append((cx, cy, w, h, p.label))
    return boxes


def remove_fp_boxes(boxes, fp_annotations, img_w, img_h):
    """
    Remove predicted boxes that overlap with False Positive annotations.
    """
    fp_centers = []
    for annot in fp_annotations:
        if annot["type"] == "point":
            fp_centers.append((annot["orig_x"] / img_w, annot["orig_y"] / img_h))
        elif annot["type"] == "freedraw":
            pts = annot["orig_points"]
            cx = np.mean([p["x"] for p in pts]) / img_w
            cy = np.mean([p["y"] for p in pts]) / img_h
            fp_centers.append((cx, cy))

    if not fp_centers:
        return boxes

    filtered = []
    for box in boxes:
        cx, cy, w, h = box[:4]
        is_fp = False
        for fx, fy in fp_centers:
            # Check if FP center falls inside this box
            if abs(fx - cx) < w / 2 and abs(fy - cy) < h / 2:
                is_fp = True
                break
        if not is_fp:
            filtered.append(box)

    return filtered


def build_dataset():
    """Main dataset builder."""
    print("=" * 50)
    print("Building YOLOv8 Training Dataset")
    print("=" * 50)

    # Create output dirs
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg")) + \
                  list(IMAGE_DIR.glob("*.jpeg")) + list(IMAGE_DIR.glob("*.tif"))

    if not image_files:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        return

    # Find corrections
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

        # Try to load existing mask
        mask_path = MASK_DIR / f"cell_mask_{stem}.npy"
        mask_path_alt = MASK_DIR / "cell_overlay_filtered.jpg"  # fallback

        labels = []

        # Get predicted boxes from Cellpose mask if available
        if mask_path.exists():
            mask = np.load(mask_path)
            pred_boxes = mask_to_bboxes(mask, img_w, img_h)
        else:
            pred_boxes = []

        # Get corrections for this image
        annots = corrections_map.get(stem, corrections_map.get("manual_corrections", []))

        # Separate FP and missed
        fp_annots = [a for a in annots if a.get("label") == "False Positive"]
        missed_annots = [a for a in annots if a.get("label") == "Missed Cell"]

        # Remove FP from predicted boxes
        if fp_annots and pred_boxes:
            pred_boxes = remove_fp_boxes(pred_boxes, fp_annots, img_w, img_h)

        # Convert remaining predicted boxes to labels
        for box in pred_boxes:
            labels.append((CELL_CLASS_ID, box[0], box[1], box[2], box[3]))

        # Add missed cell annotations as new labels
        for annot in missed_annots:
            bbox = annotation_to_bbox(annot, img_w, img_h)
            if bbox:
                labels.append((CELL_CLASS_ID, bbox[0], bbox[1], bbox[2], bbox[3]))

        # Even if no corrections exist, we can still use the image
        # (with pred_boxes as labels, or empty if no mask)
        all_samples.append((img_path, labels))
        print(f"  {stem}: {len(labels)} labels ({len(missed_annots)} added, {len(fp_annots)} removed)")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx] if split_idx > 0 else all_samples
    val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]

    # If only 1 image, use it for both train and val
    if len(all_samples) == 1:
        train_samples = all_samples
        val_samples = all_samples

    # Write files
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_path, labels in samples:
            dst_img = DATASET_DIR / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            label_file = DATASET_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_file, "w") as f:
                for label in labels:
                    f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in label) + "\n")

    # Write data.yaml
    data_yaml = DATASET_DIR / "data.yaml"
    yaml_content = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
  0: cell
"""
    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    print(f"\nDataset built: {len(train_samples)} train, {len(val_samples)} val")
    print(f"Output: {DATASET_DIR.resolve()}")
    print(f"Config: {data_yaml.resolve()}")


if __name__ == "__main__":
    build_dataset()
