# YOLO Cluster Training Package

This training package is intentionally separated from the annotation data repo.

## Repos

- Training/notebook repo: `https://github.com/nttssv/cell_count.git`
- Data repo: `https://github.com/nttssv/training_pa_he_annotation.git`

The notebook expects the data repo at:

```text
~/Desktop/training_pa_he_annotation
```

You can override that path by setting `PA_HE_DATA_REPO` before launching Jupyter.

## Cluster Setup

Clone or update the data repo:

```bash
cd ~/Desktop
if [ -d training_pa_he_annotation ]; then
  cd training_pa_he_annotation && git pull --ff-only
else
  git clone https://github.com/nttssv/training_pa_he_annotation.git
fi
```

Clone or update the training repo:

```bash
cd ~/Desktop
if [ -d cell_count_yolo ]; then
  cd cell_count_yolo && git pull --ff-only
else
  git clone --branch codex/yolo-cluster-training-package-github https://github.com/nttssv/cell_count.git cell_count_yolo
fi
```

Install the YOLO dependencies if the cluster reset removed them:

```bash
/opt/conda/bin/python -m pip install --user --no-cache-dir --force-reinstall \
  "numpy==1.26.4" "opencv-python==4.10.0.84" \
  "ultralytics==8.4.60" pandas matplotlib pyyaml tqdm tensorboard
```

Then start Jupyter from the training repo:

```bash
cd ~/Desktop/cell_count_yolo
jupyter lab
```

Open:

```text
training/yolo_cluster_live_training.ipynb
```

Run cells from top to bottom.

## What The Notebook Builds

The notebook calls:

```text
training/build_pa_he_2class_dataset.py
```

It reads from the separate data repo:

```text
~/Desktop/training_pa_he_annotation/yolo_seg_dataset
~/Desktop/training_pa_he_annotation/auxiliary_masks
```

and writes a generated runtime dataset inside the training repo:

```text
outputs/yolo_cluster_live/datasets/pa_he_2class_boundary_plus_uncertain/
```

The generated YOLO classes are:

```text
0 nucleus
1 cell_boundary
```

`cell_boundary` is built from:

```text
clear_cell_boundary + GT uncertain cell boundary
```

These source classes are intentionally dropped:

```text
compact_cell_boundary
stroma
```

## Training Outputs

Training outputs are written into:

```text
outputs/yolo_cluster_live/<run_name>/
```

Key outputs:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`
- `live_metrics.png`
- `yolo_live_training_summary.json`
- qualitative comparison PNGs under `comparison_original_gt_pred_conf*/`

The notebook also copies the final best model to:

```text
training_data/reference_models/yolo_pa_he_2class_boundary_plus_uncertain_best.pt
```

## Notes

- The training repo does not need to contain the annotation data.
- Do not use old `/Volumes/T9/...` paths on the cluster.
- The notebook writes a cluster-local `data.yaml` after building the runtime dataset.
- The notebook calls `training/run_yolo_segment_train.py`; it does not rely on `python -m ultralytics`.
- `workers=0` is intentional to avoid `/dev/shm` shared-memory crashes on Jupyter GPU clusters.
