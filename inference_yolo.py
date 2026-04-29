# inference_yolo.py
"""
Runs YOLOv8 inference and converts bounding box predictions into an
instance-labeled mask compatible with the existing Cellpose pipeline.

Usage:
    from inference_yolo import run_yolo_inference
    mask = run_yolo_inference(image, model_path="runs/detect/cell_detector/weights/best.pt")
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("runs/detect/cell_detector/weights/best.pt")


def run_yolo_inference(img, model_path=DEFAULT_MODEL_PATH, conf=0.25, imgsz=1024):
    """
    Run YOLOv8 detection on an image and convert predictions to an instance mask.
    
    Args:
        img: numpy array (H, W, 3) RGB image.
        model_path: Path to trained YOLOv8 weights.
        conf: Confidence threshold for detections.
        imgsz: Inference image size.
    
    Returns:
        mask: numpy array (H, W) with each detected cell labeled 1, 2, 3, ...
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"YOLO model not found at {model_path}. "
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
    mask = np.zeros((h, w), dtype=np.int32)

    if not results or len(results[0].boxes) == 0:
        return mask

    boxes = results[0].boxes
    label_id = 1

    for box in boxes:
        # Get box coordinates in pixel space
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Fill the bounding box region as a new instance
        # Only fill pixels that aren't already claimed by a higher-confidence detection
        region = mask[y1:y2, x1:x2]
        region[region == 0] = label_id
        label_id += 1

    return mask


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

    print(f"Running YOLO inference on {img_path}...")
    mask = run_yolo_inference(img)
    print(f"Detected {mask.max()} objects")
