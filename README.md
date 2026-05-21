# H&E Cell & Nuclei Quantification Pipeline

A digital pathology pipeline for instance-level cell and nuclei segmentation on H&E-stained histology images, with an interactive Streamlit dashboard and an active learning loop powered by YOLOv8.

---

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the automated pipeline (terminal mode)
python cellpose_count_cells.py

# 3. Launch the interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
Cellpose_testing/
├── app.py                         # Streamlit dashboard (main UI)
├── cellpose_count_cells.py        # Core segmentation pipeline
├── build_training_dataset.py      # Converts corrections → YOLO segmentation labels
├── train_yolo.py                  # YOLOv8-seg training script
├── inference_yolo.py              # YOLOv8-seg inference → instance mask
├── morphology_qc.py               # Adrenal H&E biological morphology QC
├── auto_train_loop.py             # Biology-constrained active-learning loop
├── .gitignore
│
├── sample/                        # Input histology images (place .png/.jpg here)
│   └── tile_*.png
│
├── corrections/                   # Manual correction JSON files (from dashboard)
│   └── manual_corrections.json
│
├── output/                        # Pipeline outputs (auto-generated, gitignored)
│   ├── he_deconv_hematoxylin.png  # Deconvolved H channel
│   ├── he_deconv_eosin.png        # Deconvolved E channel
│   ├── preprocessed_nuclei.png    # CLAHE-enhanced nuclei image
│   ├── preprocessed_for_cellpose.png  # CLAHE-enhanced cytoplasm image
│   ├── cell_overlay_filtered.jpg  # Yellow cell boundaries on original
│   ├── nuclei_overlay_filtered.jpg    # Cyan nuclei boundaries on original
│   ├── qc_overlay_combined.jpg    # Both overlays combined (QC)
│   ├── cell_measurements.csv      # Per-cell morphometry + confidence
│   ├── cell_mask_*.npy            # Saved instance masks for dataset building
│   ├── nuclei_measurements.csv    # Per-nucleus morphometry
│   ├── training_log.csv           # Biology-aware active-learning log
│   ├── qc_round_*/                # Active-learning QC overlays and review queues
│   └── sweep/                     # Parameter sweep results
│       ├── sweep_results.csv
│       └── sweep_d*_f*_p*.jpg
│
├── datasets/                      # YOLO training data (auto-generated, gitignored)
│   ├── data.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
│
├── output/active_learning_datasets/ # Round-specific YOLO datasets
│
├── runs/                          # YOLO training runs (auto-generated, gitignored)
│   └── segment/cell_segmenter/weights/best.pt
│
└── venv/                          # Python virtual environment (gitignored)
```

---

## File Descriptions

### `cellpose_count_cells.py` — Core Segmentation Pipeline

The backbone of the project. Contains all reusable functions:

| Function | Purpose |
|----------|---------|
| `extract_he_channels()` | H&E color deconvolution using `skimage.color.hed_from_rgb` to separate Hematoxylin (nuclei) and Eosin (cytoplasm) |
| `preprocess_for_cytoplasm()` | CLAHE contrast enhancement on the Eosin channel |
| `preprocess_for_nuclei()` | CLAHE contrast enhancement on the Hematoxylin channel |
| `run_cellpose()` | Runs Cellpose `CellposeModel.eval()` with configurable diameter, flow, and probability thresholds |
| `refine_with_watershed()` | Uses nuclei centroids as markers to split merged Cellpose blobs via watershed |
| `filter_and_relabel_mask()` | Filters objects by area, solidity, eccentricity; relabels consecutively |
| `validate_cells_with_nuclei()` | Cross-references cell masks with nuclei masks; classifies as Confident / Possible / Likely FP |
| `measure_objects_generic()` | Extracts morphometry (area, perimeter, circularity, solidity, etc.) |
| `finalize_measurements()` | Adds volume estimates and optional micron conversions |
| `draw_overlay()` | Renders boundary contours on the original image |
| `run_sweep()` | Parameter sweep across diameter × flow × cellprob combinations |

**Run standalone:** `python cellpose_count_cells.py`

---

### `app.py` — Streamlit Interactive Dashboard

The main user interface. Imports functions from `cellpose_count_cells.py` and provides:

- **Sidebar controls**: All tunable parameters (Cellpose settings, morphology filters, YOLO toggle)
- **Segmentation backend toggle**: Switch between Cellpose and trained YOLO model
- **7 tabs**:
  - **Original** — Zoomable original H&E image
  - **Cytoplasm/Cells** — Eosin deconvolution → CLAHE → cell overlay (3-panel)
  - **Nuclei** — Hematoxylin deconvolution → CLAHE → nuclei overlay (3-panel)
  - **QC Overlay** — Combined cells (yellow) + nuclei (cyan) validation view
  - **Data & Export** — Color-coded confidence table, scatter plots, CSV downloads
  - **Sweep Comparison** — Browse parameter sweep overlays with sortable table
  - **Manual Correction** — HTML5 canvas for annotating FPs and missed cells with zoom
- **Active Learning buttons**: Build dataset from corrections, retrain YOLO model
- **Zoomable viewers**: All images use a custom HTML5 canvas with 0.5× to 4.0× zoom

**Run:** `streamlit run app.py`

---

### `build_training_dataset.py` — Annotation → YOLO Segmentation Labels

Converts Cellpose masks + manual corrections into YOLO segmentation polygon labels:

1. Loads images from `sample/`, masks (`.npy`) from `output/`, corrections from `corrections/`
2. **Cellpose mask regions** → contour polygons extracted via `skimage.measure.find_contours()`
3. **Green annotations** (Missed Cell) → freedraw points used directly as polygon; point clicks → 16-point circle polygon
4. **Red annotations** (False Positive) → overlapping mask region removed from labels
5. Label format per line: `class x1 y1 x2 y2 x3 y3 ...` (normalized 0–1)
6. Outputs `datasets/` with `data.yaml`, `images/{train,val}/`, `labels/{train,val}/`
7. Auto splits 80% train / 20% val

**Run:** `python build_training_dataset.py`

---

### `train_yolo.py` — YOLOv8 Segmentation Training

Trains a YOLOv8 nano **segmentation** model on the polygon dataset.

- Uses `ultralytics` Python API with `yolov8n-seg.pt`
- Defaults: 50 epochs, imgsz=1024, batch=4 (safe for M1 Mac 8GB)
- Saves best weights to `runs/segment/cell_segmenter/weights/best.pt`
- Supports early stopping (patience=10)

**Run:**
```bash
python train_yolo.py                          # defaults
python train_yolo.py --epochs 100 --imgsz 640 # custom
```

---

### `inference_yolo.py` — YOLO Segmentation → Instance Mask

Bridges YOLOv8 segmentation predictions back into the existing pipeline:

1. Loads trained model from `runs/segment/cell_segmenter/weights/best.pt`
2. Runs inference with configurable confidence threshold
3. Extracts pixel-level masks from `results[0].masks.data`, resizes to original dimensions
4. Converts to a labeled instance mask (`np.array`, same format as Cellpose output)
5. The mask flows through the same filtering → validation → measurement → overlay pipeline

**Run standalone:** `python inference_yolo.py`

---

### `morphology_qc.py` — Biological Plausibility QC

Computes adrenal H&E-specific QC before pseudo-labels are reused:

- nuclei count, density, area, solidity, circularity, hole fraction, and fragmentation
- high-recall parenchyme context for pale clear-cell cytoplasm
- nuclei outside parenchyme, nuclei inside stroma, isolated nuclei, and topology consistency
- cytoplasm continuity, disconnected region count, vacuolation score, and biological score
- QC overlays for rejected nuclei, uncertain regions, confidence heatmaps, and parenchyme probability

### `auto_train_loop.py` — Biologically Constrained Active Learning

Runs the conservative loop:

1. Bootstrap masks with Cellpose.
2. Convert accepted nuclei/parenchyme masks to YOLOv8 segmentation labels.
3. Train `yolov8n-seg`.
4. Infer pseudo-labels for the next round.
5. Reject labels that fail morphology, topology, confidence, or biological plausibility QC.
6. Export uncertain/rejected regions for manual review before retraining.

**Run:**
```bash
python auto_train_loop.py --max-rounds 10 --epochs 25
```

Main outputs:

- `output/qc_round_<N>/` - raw image, nuclei overlay, parenchyme overlay, rejected nuclei, uncertain regions, QC panel, confidence heatmap, fragmentation map, and parenchyme probability map
- `output/training_log.csv` - nuclei/parenchyme/topology/model metrics per round
- `output/active_learning_datasets/round_<N>/` - YOLO datasets built only from accepted labels
- `runs/cell_segmenter_round_<N>/` - per-round model runs
- `runs/cell_segmenter/weights/best.pt` - best model by biological score

Manual corrections can be added as `corrections/<image_stem>.txt` YOLO labels or
as `corrections/<image_stem>_nuclei_mask.npy` and
`corrections/<image_stem>_parenchyme_mask.npy`.

---

## Active Learning Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Run Cellpose │────▶│  View Results │────▶│   Annotate   │
│  or YOLO      │     │  in Dashboard │     │  FP / Missed │
└──────────────┘     └──────────────┘     └──────┬───────┘
       ▲                                          │
       │                                          ▼
┌──────┴───────┐     ┌──────────────┐     ┌──────────────┐
│  Switch to   │◀────│ Train YOLO   │◀────│ Build Dataset│
│  YOLO backend│     │ (50 epochs)  │     │ from JSON    │
└──────────────┘     └──────────────┘     └──────────────┘
```

1. **Segment** → Run pipeline with Cellpose (or YOLO if already trained)
2. **Correct** → Use Manual Correction tab to mark false positives and missed cells
3. **Export** → Download `manual_corrections.json` to `corrections/`
4. **Build** → Click "Build Dataset from Corrections" in sidebar
5. **Train** → Click "Retrain YOLO Model" (~10-20 min on M1)
6. **Switch** → Select "Trained YOLO" in backend radio
7. **Repeat** → Each cycle improves the model

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `cellpose` | 4.0.1+ | Deep learning cell segmentation |
| `scikit-image` | — | Image I/O, region properties, color deconvolution |
| `opencv-python-headless` | — | Color space conversion, CLAHE |
| `streamlit` | 1.57+ | Interactive web dashboard |
| `plotly` | — | Interactive charts |
| `ultralytics` | 8.4+ | YOLOv8 training and inference |
| `pandas` | 3.0+ | Data tables |
| `numpy` | — | Array operations |
| `scipy` | — | Distance calculations, watershed |

Install all:
```bash
pip install cellpose scikit-image opencv-python-headless streamlit plotly ultralytics pandas numpy scipy
```

---

## Configuration

Key parameters are set at the top of `cellpose_count_cells.py`:

```python
MICRON_PER_PIXEL = None    # Set if pixel size is known (e.g., 0.25)
DIAMETER = 50              # Expected cell diameter in pixels
FLOW_THRESHOLD = 0.35      # Cellpose flow threshold
CELLPROB_THRESHOLD = -1.5  # Cellpose cell probability threshold
MIN_AREA_PX = 100          # Minimum cell area filter
MAX_AREA_PX = 30000        # Maximum cell area filter
MIN_SOLIDITY = 0.5         # Reject hollow/irregular shapes
MAX_ECCENTRICITY = 0.98    # Reject very elongated shapes
```

All of these can also be adjusted interactively in the Streamlit sidebar.

---

## Notes

- **GPU**: Cellpose and YOLO automatically use MPS (Apple Silicon) or CUDA if available, falling back to CPU
- **Image format**: Place `.png`, `.jpg`, `.jpeg`, or `.tif` files in `sample/`
- **Single image**: The pipeline works with a single image; more images improve YOLO training
- **Masks**: The cell mask uses **yellow** boundaries; nuclei use **cyan**
- **Confidence categories**: Each detected cell is classified as *Confident Cell*, *Possible Cell*, or *Likely False Positive* based on nuclei overlap
