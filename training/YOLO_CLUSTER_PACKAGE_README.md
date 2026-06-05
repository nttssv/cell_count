# YOLO Cluster Training Package

This package is intended to be pulled onto the GPU cluster and run from the repository root.

## Included assets

- YOLO segmentation dataset: `training_data/dataset/yolo_seg_dataset`
- Base YOLO model: `yolov8s-seg.pt`
- Previous local YOLO best model: `training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt`
- Live training notebook: `training/yolo_cluster_live_training.ipynb`
- Python training wrapper: `training/run_yolo_segment_train.py`

## Cluster usage

```bash
cd /home/jovyan/Desktop/<repo-folder>
jupyter lab
```

Open:

```text
training/yolo_cluster_live_training.ipynb
```

Run cells from top to bottom. The notebook now builds a temporary two-class dataset before training:

- keeps `nucleus`
- keeps `clear_cell_boundary`
- drops `compact_cell_boundary`
- drops `stroma`
- oversamples hard/dense train tiles: `p2_tile_12`, `p2_tile_13`, `p2_tile_14`, `p2_tile_16`, `p2_tile_18`

The generated dataset is written to:

```text
outputs/yolo_cluster_live/datasets/yolo_2class_nucleus_clear_boundary_oversampled/
```

Training outputs are written into:

```text
outputs/yolo_cluster_live/<run_name>/
```

Key outputs after training:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`
- `live_metrics.png`
- `yolo_live_training_summary.json`

The notebook starts from the newest available YOLO `best.pt` under `outputs/yolo_cluster_live/*/weights/best.pt` when present. If no previous run is available, it falls back to `training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt`, then `yolov8s-seg.pt`.

The active training settings use lighter boundary-preserving augmentation:

- `lr0=0.0005` when fine-tuning from an existing best checkpoint
- `mosaic=0.3`
- `close_mosaic=40`
- `copy_paste=0.0`
- `degrees=5`
- `translate=0.04`
- `scale=0.2`

The notebook also copies the final best two-class model to:

```text
training_data/reference_models/yolo_2class_nucleus_clear_boundary_best.pt
```

## Notes

- Do not use old `/Volumes/T9/...` paths on the cluster.
- The committed source `data.yaml` is portable, and the notebook writes an absolute generated `data.yaml` for the two-class runtime dataset.
- The notebook calls `training/run_yolo_segment_train.py`; it does not rely on `python -m ultralytics`.
- If another GPU training job is running, wait for it to finish unless you intentionally want to share the GPU.
