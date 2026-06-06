# CGH P2 CellSeg1 Instance Dataset

Portable CellSeg1 training dataset copied from:

```text
/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2
```

AppleDouble files (`._*`) were excluded during copy.

## Primary CellSeg1 Inputs

- `images/`: 31 H&E tile PNGs
- `masks/`: 31 cell-boundary instance masks with contiguous instance labels

## Preserved QC Context

- `metadata/dataset_manifest.csv`
- `metadata/cell_instances.csv`
- `metadata/boundary_qc.csv`
- `auxiliary_masks/`: nuclei, stroma, uncertain, and edge/invalid masks
- `semantic_masks/`: semantic review masks
- `previews/`: original annotation overlays
- `dataset_summary.json`: generated package summary

CellSeg1 trains a single positive instance class from `masks/`; clear and
compact boundary labels are preserved in metadata for downstream analysis.

Current snapshot:

- 678 trainable cell-boundary instances
- 557 clear-cell boundary instances
- 121 compact-cell boundary instances
- 766 in-tile nuclei in metadata/auxiliary masks

The exporter includes final training tiles named `P2 tile NN` and
`yolo_tile_NN`, and excludes temporary duplicate tile annotations such as
`cell_boundary_clean`, `nuclei_clean`, and unnumbered `yolo_tile`.
