# inference_yolo.py
"""
Runs YOLOv8 SEGMENTATION inference and converts predicted masks into an
instance-labeled mask compatible with the existing Cellpose pipeline.

Usage:
    from inference_yolo import run_yolo_inference
    mask = run_yolo_inference(image)
"""

from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("runs/cell_segmenter/weights/best.pt")


def run_yolo_inference(img, model_path=DEFAULT_MODEL_PATH, conf=0.25, imgsz=1024):
    """
    Run YOLOv8 segmentation on an image and produce an instance-labeled mask.

    Args:
        img: numpy array (H, W, 3) RGB image.
        model_path: Path to trained YOLOv8-seg weights.
        conf: Confidence threshold for detections.
        imgsz: Inference image size.

    Returns:
        mask: numpy array (H, W) int32, each detected cell labeled 1, 2, 3, ...
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"YOLO segmentation model not found at {model_path}. "
            "Train a model first with 'python train_yolo.py'."
        )

    model = YOLO(str(model_path))

    results = model.predict(
        source=img,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    h, w = img.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)

    if not results or results[0].masks is None:
        return instance_mask

    # results[0].masks.data is a tensor of shape (N, mask_h, mask_w) with float values
    masks_tensor = results[0].masks.data.cpu().numpy()

    for idx, seg_mask in enumerate(masks_tensor):
        label_id = idx + 1

        # Resize mask from YOLO output size to original image size
        if seg_mask.shape[0] != h or seg_mask.shape[1] != w:
            seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Threshold the mask (YOLO outputs soft probabilities)
        binary = (seg_mask > 0.5).astype(np.uint8)

        # Only fill pixels not already claimed by a higher-confidence instance
        # (YOLO returns results sorted by confidence, highest first)
        instance_mask[np.logical_and(binary == 1, instance_mask == 0)] = label_id

    return instance_mask


def get_model_info(model_path=DEFAULT_MODEL_PATH):
    """Returns basic info about a trained model, or None if not found."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    return {
        "path": str(model_path.resolve()),
        "size_mb": model_path.stat().st_size / (1024 * 1024),
        "exists": True,
    }


if __name__ == "__main__":
    from skimage import io as skio
    from cellpose_count_cells import find_sample_image

    img_path = find_sample_image()
    img = skio.imread(img_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    print(f"Running YOLO segmentation on {img_path}...")
    mask = run_yolo_inference(img)
    n_objects = mask.max()
    print(f"Detected {n_objects} objects with pixel-level masks")

    if n_objects > 0:
        from skimage import measure
        props = measure.regionprops(mask)
        areas = [p.area for p in props]
        print(f"Mean area: {np.mean(areas):.0f} px, Median: {np.median(areas):.0f} px")
