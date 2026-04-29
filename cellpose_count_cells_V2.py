# cellpose_count_cells.py

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from skimage import io, measure, segmentation
from cellpose import models

# Optional: set this if you know microscope scale
MICRON_PER_PIXEL = None  # example: 0.25

INPUT_DIR = Path("sample")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Find image
image_files = list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + list(INPUT_DIR.glob("*.tif"))
if not image_files:
    raise FileNotFoundError("No image found in sample/")

image_path = image_files[0]
print(f"Processing: {image_path}")

# Read image
img = io.imread(image_path)

# Cellpose model
model = models.CellposeModel(gpu=True, model_type="cyto3")

masks, flows, styles = model.eval(
    img,
    diameter=None,          # try 20–35 if result is bad
    channels=[0, 0],
    flow_threshold=0.4,
    cellprob_threshold=0.0
)

# Save mask
mask_path = OUTPUT_DIR / "cellpose_mask.png"
io.imsave(mask_path, masks.astype(np.uint16))

# Create overlay boundary
boundaries = segmentation.find_boundaries(masks, mode="outer")
overlay = img.copy()

if overlay.ndim == 2:
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

overlay[boundaries] = [255, 255, 0]  # yellow boundary

overlay_path = OUTPUT_DIR / "cellpose_overlay.png"
io.imsave(overlay_path, overlay.astype(np.uint8))

# Measurements
props = measure.regionprops(masks)

rows = []
for p in props:
    area_px = p.area
    perimeter_px = p.perimeter
    diameter_px = p.equivalent_diameter
    radius_px = diameter_px / 2
    volume_px3 = (4 / 3) * np.pi * radius_px**3

    row = {
        "cell_id": p.label,
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "centroid_x": p.centroid[1],
        "centroid_y": p.centroid[0],
        "equivalent_diameter_px": diameter_px,
        "estimated_volume_px3": volume_px3,
    }

    if MICRON_PER_PIXEL is not None:
        area_um2 = area_px * MICRON_PER_PIXEL**2
        diameter_um = diameter_px * MICRON_PER_PIXEL
        radius_um = diameter_um / 2
        volume_um3 = (4 / 3) * np.pi * radius_um**3

        row.update({
            "area_um2": area_um2,
            "equivalent_diameter_um": diameter_um,
            "estimated_volume_um3": volume_um3,
        })

    rows.append(row)

df = pd.DataFrame(rows)
csv_path = OUTPUT_DIR / "cell_measurements.csv"
df.to_csv(csv_path, index=False)

summary = {
    "image": str(image_path),
    "total_cells": len(props),
    "mean_area_px": float(df["area_px"].mean()) if len(df) else 0,
    "median_area_px": float(df["area_px"].median()) if len(df) else 0,
}

summary_path = OUTPUT_DIR / "summary.txt"
with open(summary_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("Done.")
print(f"Cells detected: {len(props)}")
print(f"Saved overlay: {overlay_path}")
print(f"Saved CSV: {csv_path}")