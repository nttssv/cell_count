# YOLO Cluster Training Package

This package is intended to be pulled onto the GPU cluster and run from the repository root.

## Included assets

- YOLO segmentation dataset: `training_data/dataset/yolo_seg_dataset`
- Base YOLO model: `yolov8s-seg.pt`
- Previous local YOLO best model: `training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt`
- Live training notebook: `training/yolo_cluster_live_training.ipynb`

## Cluster usage

```bash
cd /home/jovyan/Desktop/<repo-folder>
jupyter lab
```

Open:

```text
training/yolo_cluster_live_training.ipynb
```

Run cells from top to bottom. The notebook writes a cluster-specific runtime YAML file and trains into:

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

The notebook also copies the final best model to:

```text
training_data/reference_models/yolo_sam31_p2_24tiles_best.pt
```

## Notes

- Do not use old `/Volumes/T9/...` paths on the cluster.
- The committed `data.yaml` is portable, and the notebook writes an absolute `data_cluster_runtime.yaml` at runtime.
- If another GPU training job is running, wait for it to finish unless you intentionally want to share the GPU.
