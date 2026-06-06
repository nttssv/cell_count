# CellSeg1 Cluster Training Package

This package trains a CellSeg1/SAM LoRA model on the CGH PA P2 cell-boundary
instance masks copied from the T9 YOLO training export.

## Included Assets

- Portable CellSeg1 instance dataset:
  `training_data/dataset/cellseg1_instance_train`
- Primary training images:
  `training_data/dataset/cellseg1_instance_train/images`
- Primary instance masks:
  `training_data/dataset/cellseg1_instance_train/masks`
- Dataset metadata and QC summary:
  `training_data/dataset/cellseg1_instance_train/dataset_summary.json`
- Live training notebook:
  `training/cellseg1_cluster_live_training.ipynb`
- Python training wrapper:
  `training/run_cellseg1_lora_train.py`

## Dataset Snapshot

- 31 tiles
- 678 trainable cell-boundary instances
- 557 clear-cell boundary instances
- 121 compact-cell boundary instances
- 766 in-tile nuclei preserved in metadata/auxiliary masks

CellSeg1 treats clear and compact boundaries as one positive instance class.
Class-specific clear/compact classification still needs a separate classifier
or the existing YOLO/SAM3 multiclass route.

## Cluster Usage

```bash
cd /home/jovyan/Desktop/<repo-folder>
jupyter lab
```

Open:

```text
training/cellseg1_cluster_live_training.ipynb
```

Run cells from top to bottom. The notebook will:

1. locate the portable dataset, or rebuild a clean runtime copy from T9 if T9
   is mounted;
2. clone `https://github.com/Nuisal/cellseg1.git` into
   `outputs/cellseg1_cluster_live/cellseg1_repo` if needed;
3. optionally install CellSeg1 requirements when
   `CELLSEG1_INSTALL_REQUIREMENTS=1`;
4. download the SAM ViT-H checkpoint if no checkpoint path is provided;
5. write a runtime CellSeg1 YAML config;
6. train LoRA weights;
7. save summaries and qualitative GT-vs-prediction images.

During training, the notebook starts CellSeg1 in a subprocess, writes
`cellseg1_train_live.log`, and refreshes an in-notebook live dashboard with GPU
status, elapsed time, checkpoint status, and the latest training log tail.

Install CUDA PyTorch separately for the assigned cluster GPU before running the
training cell. CellSeg1's repository requirements cover the image-processing
and Streamlit dependencies but not the cluster-specific PyTorch build.

## Useful Environment Overrides

```bash
export CELLSEG1_EPOCHS=120
export CELLSEG1_BATCH=1
export CELLSEG1_GRAD_ACCUM=32
export CELLSEG1_DUPLICATE_DATA=64
export CELLSEG1_INSTALL_REQUIREMENTS=1
export CELLSEG1_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export CELLSEG1_LIVE_INTERVAL_SECONDS=15
export CELLSEG1_LOG_TAIL_LINES=80
```

The default output root is:

```text
outputs/cellseg1_cluster_live/
```

The trained LoRA checkpoint is copied to:

```text
training_data/reference_models/cellseg1_cgh_p2_cell_boundary_lora.pth
```
