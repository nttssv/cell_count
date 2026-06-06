# CellSeg1 Cluster Training Package

This package trains a CellSeg1/SAM LoRA model while keeping annotation data in
a separate repo.

## Repos

- Training/notebook repo: `https://github.com/nttssv/cell_count.git`
- Data repo: `https://github.com/nttssv/training_pa_he_annotation.git`

The notebook expects the data repo at:

```text
~/Desktop/training_pa_he_annotation
```

You can override that path with:

```bash
export PA_HE_DATA_REPO=/path/to/training_pa_he_annotation
```

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

Clone or update the CellSeg1 training repo:

```bash
cd ~/Desktop
if [ -d cell_count_cellseg1 ]; then
  cd cell_count_cellseg1 && git pull --ff-only
else
  git clone --branch codex/training-data-only-cellseg1 https://github.com/nttssv/cell_count.git cell_count_cellseg1
fi
```

Install/repair basic Python dependencies if the cluster reset removed them:

```bash
/opt/conda/bin/python -m pip install --user --no-cache-dir \
  "numpy<2" pillow matplotlib pyyaml opencv-python pandas tqdm
```

CellSeg1 itself is cloned by the notebook into:

```text
outputs/cellseg1_cluster_live/cellseg1_repo
```

If the current cluster environment does not have CellSeg1 dependencies, set:

```bash
export CELLSEG1_INSTALL_REQUIREMENTS=1
```

Install the correct CUDA PyTorch build separately if `torch.cuda.is_available()`
is false.

## Run

Start Jupyter from the training repo:

```bash
cd ~/Desktop/cell_count_cellseg1
jupyter lab
```

Open:

```text
training/cellseg1_cluster_live_training.ipynb
```

Run cells from top to bottom.

## What The Notebook Uses

From the data repo, the notebook reads:

```text
train/images/*.png
train/masks/*.png
dataset_manifest.csv
cell_instances.csv
boundary_qc.csv
auxiliary_masks/
semantic_masks/
previews/
```

It creates a runtime copy inside the training repo:

```text
outputs/cellseg1_cluster_live/datasets/pa_he_cellseg1_instance_train/
```

CellSeg1 trains one positive instance class: trainable cell boundary.

The source `train/masks` currently include trainable clear and compact cell
boundaries. `GT uncertain cell boundary`, edge/invalid masks, nuclei, and
stroma are preserved as auxiliary/review data but are not positive CellSeg1
instances unless the data repo export changes `train/masks`.

## Dataset Snapshot

Current data repo export:

- 35 tiles
- 744 trainable cell-boundary instances
- 622 clear-cell boundary instances
- 122 compact-cell boundary instances
- 834 in-tile nuclei
- 428 uncertain boundary review/ignore regions
- 74 edge/invalid boundary ignore regions

## Training Outputs

Training outputs are written into:

```text
outputs/cellseg1_cluster_live/<run_name>/
```

Key outputs:

- `sam_lora_cgh_p2_cell_boundary.pth`
- `cellseg1_cgh_p2_runtime_config.yaml`
- `cellseg1_train_live.log`
- `cellseg1_training_summary.json`
- `dataset_runtime_summary.json`
- qualitative comparison PNGs under `comparison_original_gt_pred_*`

The notebook also copies the final LoRA checkpoint to:

```text
training_data/reference_models/cellseg1_cgh_p2_cell_boundary_lora.pth
```

## Useful Environment Overrides

```bash
export PA_HE_DATA_REPO=~/Desktop/training_pa_he_annotation
export CELLSEG1_EPOCHS=120
export CELLSEG1_BATCH=1
export CELLSEG1_GRAD_ACCUM=32
export CELLSEG1_DUPLICATE_DATA=64
export CELLSEG1_INSTALL_REQUIREMENTS=1
export CELLSEG1_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export CELLSEG1_LIVE_INTERVAL_SECONDS=15
export CELLSEG1_LOG_TAIL_LINES=80
```

For old packaged data only, set:

```bash
export CELLSEG1_ALLOW_PACKAGED_FALLBACK=1
```
