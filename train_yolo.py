# train_yolo.py
"""
Trains a YOLOv8 detection model on the dataset built by build_training_dataset.py.

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(epochs=50, imgsz=1024, batch=4, model_size="n"):
    """
    Train YOLOv8 on the cell detection dataset.
    
    Args:
        epochs: Number of training epochs.
        imgsz: Image size for training (1024 recommended for pathology).
        batch: Batch size (4 is safe for M1 8GB RAM).
        model_size: YOLO model size - 'n' (nano), 's' (small), 'm' (medium).
    """
    data_yaml = Path("datasets/data.yaml")
    if not data_yaml.exists():
        print("ERROR: datasets/data.yaml not found.")
        print("Run 'python build_training_dataset.py' first.")
        return None

    model_name = f"yolov8{model_size}.pt"
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    print(f"\nStarting training:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Dataset: {data_yaml.resolve()}")
    print()

    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,        # Early stopping
        save=True,
        project="runs/detect",
        name="cell_detector",
        exist_ok=True,       # Overwrite previous run
        verbose=True,
    )

    best_weights = Path("runs/detect/cell_detector/weights/best.pt")
    if best_weights.exists():
        print(f"\nTraining complete!")
        print(f"Best model saved to: {best_weights.resolve()}")
    else:
        print("\nTraining finished but best.pt not found. Check runs/detect/cell_detector/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 cell detector")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m"])
    args = parser.parse_args()

    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_size=args.model)
