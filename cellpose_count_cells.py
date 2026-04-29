# cellpose_count_cells.py
import os
import cv2
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, measure, segmentation, color, feature
from skimage.segmentation import clear_border, watershed
from scipy.spatial.distance import cdist
from scipy import ndimage
from cellpose import models

# Filter warnings from skimage about low contrast images
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

# =====================================================================
# CONFIGURATION & DEFAULT PARAMETERS
# =====================================================================
MICRON_PER_PIXEL = None

# Cellpose Main Parameters
DIAMETER = 50
FLOW_THRESHOLD = 0.35
CELLPROB_THRESHOLD = -1.5

# Advanced Morphology Filtering
MIN_AREA_PX = 100
MAX_AREA_PX = 30000
MIN_SOLIDITY = 0.5
MAX_ECCENTRICITY = 0.98

# Sweep Configuration
RUN_PARAMETER_SWEEP = False
SWEEP_DIAMETERS = [40, 50, 60, 70]
SWEEP_FLOW_THRESHOLDS = [0.25, 0.3, 0.35, 0.4]
SWEEP_CELLPROB_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5]

INPUT_DIR = Path("sample")
OUTPUT_DIR = Path("output")
SWEEP_DIR = OUTPUT_DIR / "sweep"

def find_sample_image(input_dir=INPUT_DIR):
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + \
                  list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.tif"))
    if not image_files:
        raise FileNotFoundError(f"No image found in {input_dir}/")
    return image_files[0]

def extract_he_channels(img_rgb):
    """
    Separates the RGB image into Hematoxylin (nuclei) and Eosin (cytoplasm)
    using color deconvolution.
    Returns uint8 normalized images.
    """
    hed = color.separate_stains(img_rgb, color.hed_from_rgb)
    
    # H channel (Hematoxylin) -> nuclei
    h_channel = hed[:, :, 0]
    # Normalize to 0-255
    h_channel = np.clip((h_channel - h_channel.min()) / (h_channel.max() - h_channel.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
    
    # E channel (Eosin) -> cytoplasm
    e_channel = hed[:, :, 1]
    e_channel = np.clip((e_channel - e_channel.min()) / (e_channel.max() - e_channel.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
    
    return h_channel, e_channel

def preprocess_for_cytoplasm(e_channel):
    """Applies CLAHE to the Eosin channel to boost cell boundary contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(e_channel)

def preprocess_for_nuclei(h_channel):
    """Applies CLAHE to the Hematoxylin channel."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(h_channel)

def run_cellpose(img, model_type, diameter, flow_threshold, cellprob_threshold):
    model = models.CellposeModel(gpu=True, model_type=model_type)
    masks, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    return masks

def refine_with_watershed(cell_mask, nuclei_mask):
    """
    Uses nuclei centroids as seeds to split large/merged cellpose blobs.
    """
    # 1. Get nuclei centroids
    nuclei_props = measure.regionprops(nuclei_mask)
    if not nuclei_props:
        return cell_mask
        
    markers = np.zeros_like(cell_mask, dtype=np.int32)
    for i, p in enumerate(nuclei_props, start=1):
        r, c = map(int, p.centroid)
        r = min(max(r, 0), markers.shape[0]-1)
        c = min(max(c, 0), markers.shape[1]-1)
        markers[r, c] = i
        
    mask = cell_mask > 0
    distance = ndimage.distance_transform_edt(mask)
    
    refined_mask = watershed(-distance, markers, mask=mask)
    
    unseeded_mask = mask & (refined_mask == 0)
    if np.any(unseeded_mask):
        labeled_unseeded, num_features = ndimage.label(unseeded_mask)
        max_id = refined_mask.max()
        labeled_unseeded[labeled_unseeded > 0] += max_id
        refined_mask = refined_mask + labeled_unseeded
        
    return refined_mask

def filter_and_relabel_mask(mask, min_area, max_area, min_solidity, max_eccentricity, remove_border=False):
    if remove_border:
        mask = clear_border(mask)
        
    props = measure.regionprops(mask)
    filtered_mask = np.zeros_like(mask)
    new_label = 1
    
    for p in props:
        if p.area >= 5:
            if (min_area <= p.area <= max_area) and (p.solidity >= min_solidity) and (p.eccentricity <= max_eccentricity):
                filtered_mask[mask == p.label] = new_label
                new_label += 1
                
    return filtered_mask

def validate_cells_with_nuclei(cell_props, nuclei_props, cell_mask, nuclei_mask):
    """
    Cross-references cell masks with nuclei masks.
    Calculates number of nuclei inside, nearest nucleus distance, etc.
    """
    if not cell_props:
        return []
        
    nuc_centroids = np.array([p.centroid for p in nuclei_props]) if nuclei_props else np.empty((0, 2))
    
    results = []
    for cp in cell_props:
        r0, c0, r1, c1 = cp.bbox
        cell_nuc_mask = nuclei_mask[r0:r1, c0:c1]
        cell_obj_mask = cell_mask[r0:r1, c0:c1] == cp.label
        
        nuclei_inside_ids = np.unique(cell_nuc_mask[cell_obj_mask])
        nuclei_inside_ids = [n for n in nuclei_inside_ids if n > 0]
        nuc_count = len(nuclei_inside_ids)
        
        min_dist = -1.0
        if len(nuc_centroids) > 0:
            dists = cdist([cp.centroid], nuc_centroids)
            min_dist = np.min(dists)
            
        nuc_area_ratio = 0.0
        if nuc_count > 0:
            total_nuc_area = sum([np.sum(cell_nuc_mask[cell_obj_mask] == n_id) for n_id in nuclei_inside_ids])
            nuc_area_ratio = total_nuc_area / cp.area
            
        if nuc_count > 0 or (0 <= min_dist < 15):
            category = "Confident Cell"
        elif cp.area > 300 and cp.solidity > 0.8:
            category = "Possible Cell"
        else:
            category = "Likely False Positive"
            
        diam_px = cp.equivalent_diameter_area if hasattr(cp, 'equivalent_diameter_area') else cp.equivalent_diameter
        circularity = (4 * np.pi * cp.area) / (cp.perimeter**2) if cp.perimeter > 0 else 0
        
        res = {
            "object_id": cp.label,
            "type": "cell",
            "confidence_category": category,
            "area_px": cp.area,
            "perimeter_px": cp.perimeter,
            "centroid_x": cp.centroid[1],
            "centroid_y": cp.centroid[0],
            "equivalent_diameter_px": diam_px,
            "solidity": cp.solidity,
            "eccentricity": cp.eccentricity,
            "circularity": circularity,
            "nuclei_inside": nuc_count,
            "nearest_nucleus_distance_px": min_dist,
            "nucleus_to_cell_area_ratio": nuc_area_ratio
        }
        results.append(res)
        
    return results

def measure_objects_generic(props, obj_type="nucleus"):
    results = []
    for p in props:
        diam_px = p.equivalent_diameter_area if hasattr(p, 'equivalent_diameter_area') else p.equivalent_diameter
        circularity = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
        res = {
            "object_id": p.label,
            "type": obj_type,
            "confidence_category": "Confident",
            "area_px": p.area,
            "perimeter_px": p.perimeter,
            "centroid_x": p.centroid[1],
            "centroid_y": p.centroid[0],
            "equivalent_diameter_px": diam_px,
            "solidity": p.solidity,
            "eccentricity": p.eccentricity,
            "circularity": circularity,
            "nuclei_inside": 1,
            "nearest_nucleus_distance_px": 0.0,
            "nucleus_to_cell_area_ratio": 1.0
        }
        results.append(res)
    return results

def finalize_measurements(results_list, mpp=MICRON_PER_PIXEL):
    for r in results_list:
        radius_px = r["equivalent_diameter_px"] / 2.0
        r["estimated_volume_px3"] = (4.0 / 3.0) * np.pi * radius_px**3
        
        if mpp is not None:
            r["area_um2"] = r["area_px"] * (mpp**2)
            r["equivalent_diameter_um"] = r["equivalent_diameter_px"] * mpp
            radius_um = r["equivalent_diameter_um"] / 2.0
            r["estimated_volume_um3"] = (4.0 / 3.0) * np.pi * radius_um**3
            r["nearest_nucleus_distance_um"] = r["nearest_nucleus_distance_px"] * mpp
            
    return pd.DataFrame(results_list)

def draw_overlay(img, mask, color=[255, 255, 0]):
    boundaries = segmentation.find_boundaries(mask, mode="outer")
    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    overlay[boundaries] = color
    return overlay

def save_summary(output_path, image_name, df_cells, df_nuc):
    summary = {
        "image": str(image_name),
        "total_cells": len(df_cells),
        "total_nuclei": len(df_nuc),
        "confident_cells": len(df_cells[df_cells["confidence_category"] == "Confident Cell"]) if not df_cells.empty else 0,
        "possible_cells": len(df_cells[df_cells["confidence_category"] == "Possible Cell"]) if not df_cells.empty else 0,
        "false_positive_risk": len(df_cells[df_cells["confidence_category"] == "Likely False Positive"]) if not df_cells.empty else 0,
        "mean_cell_area_px": float(df_cells["area_px"].mean()) if not df_cells.empty else 0,
        "median_cell_area_px": float(df_cells["area_px"].median()) if not df_cells.empty else 0,
        "cells_with_nuclei_percentage": float((df_cells["nuclei_inside"] > 0).mean() * 100) if not df_cells.empty else 0,
    }
    with open(output_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary

def run_sweep(cyto_img, img_rgb, output_dir=SWEEP_DIR):
    output_dir.mkdir(exist_ok=True)
    sweep_results = []
    print("\n--- Running Cell Parameter Sweep ---")
    for diam in SWEEP_DIAMETERS:
        for flow in SWEEP_FLOW_THRESHOLDS:
            for prob in SWEEP_CELLPROB_THRESHOLDS:
                print(f"Sweep -> Diam: {diam}, Flow: {flow}, Prob: {prob}")
                raw_masks = run_cellpose(cyto_img, "cyto3", diam, flow, prob)
                filtered_masks = filter_and_relabel_mask(raw_masks, MIN_AREA_PX, MAX_AREA_PX, MIN_SOLIDITY, MAX_ECCENTRICITY)
                
                props = measure.regionprops(filtered_masks)
                cell_count = len(props)
                mean_area = np.mean([p.area for p in props]) if cell_count > 0 else 0
                
                overlay = draw_overlay(img_rgb, filtered_masks, color=[255, 255, 0])
                sweep_name = f"sweep_d{diam}_f{flow}_p{prob}.jpg"
                overlay_path = output_dir / sweep_name
                io.imsave(overlay_path, overlay, check_contrast=False)
                
                sweep_results.append({
                    "diameter": diam,
                    "flow_threshold": flow,
                    "cellprob_threshold": prob,
                    "cell_count": cell_count,
                    "mean_cell_area_px": mean_area,
                    "overlay_path": str(overlay_path)
                })
                
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(output_dir / "sweep_results.csv", index=False)
    print("--- Sweep Complete ---\n")
    return df_sweep

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        image_path = find_sample_image()
    except FileNotFoundError as e:
        print(e)
        return
        
    print(f"Processing: {image_path}")
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    h_channel, e_channel = extract_he_channels(img)
    io.imsave(OUTPUT_DIR / "he_deconv_hematoxylin.png", h_channel, check_contrast=False)
    io.imsave(OUTPUT_DIR / "he_deconv_eosin.png", e_channel, check_contrast=False)

    nuc_preprocessed = preprocess_for_nuclei(h_channel)
    cyto_preprocessed = preprocess_for_cytoplasm(e_channel)
    io.imsave(OUTPUT_DIR / "preprocessed_nuclei.png", nuc_preprocessed, check_contrast=False)
    io.imsave(OUTPUT_DIR / "preprocessed_for_cellpose.png", cyto_preprocessed, check_contrast=False)

    print("\n--- Running Nuclei Segmentation ---")
    raw_nuc_masks = run_cellpose(nuc_preprocessed, "nuclei", DIAMETER * 0.5, FLOW_THRESHOLD, CELLPROB_THRESHOLD)
    filtered_nuc_masks = filter_and_relabel_mask(raw_nuc_masks, MIN_AREA_PX * 0.1, MAX_AREA_PX * 0.3, 0.4, 0.99)
    nuc_props = measure.regionprops(filtered_nuc_masks)
    
    res_nuc = measure_objects_generic(nuc_props, "nucleus")
    df_nuclei = finalize_measurements(res_nuc)
    df_nuclei.to_csv(OUTPUT_DIR / "nuclei_measurements.csv", index=False)
    io.imsave(OUTPUT_DIR / "nuclei_overlay_filtered.jpg", draw_overlay(img, filtered_nuc_masks, [0, 255, 255]), check_contrast=False)

    print("\n--- Running Whole-Cell Segmentation ---")
    if RUN_PARAMETER_SWEEP:
        run_sweep(cyto_preprocessed, img)

    raw_cell_masks = run_cellpose(cyto_preprocessed, "cyto3", DIAMETER, FLOW_THRESHOLD, CELLPROB_THRESHOLD)
    refined_cell_masks = refine_with_watershed(raw_cell_masks, filtered_nuc_masks)
    
    filtered_cell_masks = filter_and_relabel_mask(refined_cell_masks, MIN_AREA_PX, MAX_AREA_PX, MIN_SOLIDITY, MAX_ECCENTRICITY)
    cell_props = measure.regionprops(filtered_cell_masks)
    
    print("Validating cells with nuclei...")
    res_cells = validate_cells_with_nuclei(cell_props, nuc_props, filtered_cell_masks, filtered_nuc_masks)
    df_cells = finalize_measurements(res_cells)
    df_cells.to_csv(OUTPUT_DIR / "cell_measurements.csv", index=False)
    io.imsave(OUTPUT_DIR / "cell_overlay_filtered.jpg", draw_overlay(img, filtered_cell_masks, [255, 255, 0]), check_contrast=False)
    
    save_summary(OUTPUT_DIR / "summary.txt", image_path, df_cells, df_nuclei)

    print("\n" + "="*40)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("="*40)

if __name__ == "__main__":
    main()
