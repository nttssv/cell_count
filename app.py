# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from skimage import io, measure, segmentation
import cv2
import traceback
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
import json

from cellpose_count_cells import (
    find_sample_image,
    extract_he_channels,
    preprocess_for_cytoplasm,
    preprocess_for_nuclei,
    run_cellpose,
    refine_with_watershed,
    filter_and_relabel_mask,
    validate_cells_with_nuclei,
    measure_objects_generic,
    finalize_measurements,
    draw_overlay,
    run_sweep,
    INPUT_DIR,
    OUTPUT_DIR,
    SWEEP_DIR
)

st.set_page_config(page_title="Advanced H&E Segmentation Dashboard", layout="wide")

def load_custom_css():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #C9D1D9; }
        [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
        h1, h2, h3, h4 { color: #E6EDF3 !important; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #58A6FF; }
        .custom-card {
            background-color: #161B22; border-radius: 8px; padding: 16px;
            border: 1px solid #30363D; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 15px;
        }
        .card-title { color: #8B949E; font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
        .card-value { color: #C9D1D9; font-size: 28px; font-weight: bold; }
        .value-cyan { color: #39D353 !important; }
        .value-yellow { color: #D2A8FF !important; }
        .stDataFrame { border-radius: 8px; border: 1px solid #30363D; }
        </style>
    """, unsafe_allow_html=True)

def metric_card(title, value, color_class=""):
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">{title}</div>
            <div class="card-value {color_class}">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def zoomable_image(image_path_or_array, caption="", uid="img", base_width=900):
    """
    Renders a zoomable image viewer using a self-contained HTML5 canvas.
    Works with file paths (str/Path) or numpy arrays.
    Each call must have a unique `uid` to avoid DOM id collisions.
    """
    import base64 as b64lib
    from PIL import Image as PILImage
    from io import BytesIO

    if isinstance(image_path_or_array, (str, Path)):
        pil_img = PILImage.open(image_path_or_array).convert("RGB")
    else:
        # numpy array
        import numpy as _np
        arr = image_path_or_array
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        pil_img = PILImage.fromarray(arr.astype(np.uint8))

    orig_w, orig_h = pil_img.size
    send_w = min(orig_w, 2000)
    send_h = int(orig_h * send_w / orig_w)
    pil_send = pil_img.resize((send_w, send_h))

    buf = BytesIO()
    pil_send.save(buf, format="PNG")
    img_b64 = b64lib.b64encode(buf.getvalue()).decode()

    base_h = int(orig_h * base_width / orig_w)

    html = f"""
    <style>
      .zv-{uid} {{ font-family:-apple-system,sans-serif; color:#C9D1D9; }}
      .zv-{uid} .zrow {{ display:flex; gap:8px; align-items:center; margin-bottom:6px; flex-wrap:wrap; }}
      .zv-{uid} .zrow button {{
        background:#21262D; color:#C9D1D9; border:1px solid #30363D; padding:3px 10px;
        border-radius:5px; cursor:pointer; font-size:12px;
      }}
      .zv-{uid} .zrow button:hover {{ background:#30363D; }}
      .zv-{uid} .zrow input[type=range] {{ width:140px; accent-color:#58A6FF; }}
      .zv-{uid} .zrow span {{ font-size:12px; color:#8B949E; }}
      .zv-{uid} .cwrap {{ overflow:auto; border:1px solid #30363D; border-radius:6px; max-height:70vh; }}
      .zv-{uid} canvas {{ display:block; }}
      .zv-{uid} .cap {{ font-size:12px; color:#8B949E; margin-top:4px; }}
    </style>
    <div class="zv-{uid}">
      <div class="zrow">
        <button onclick="zv_{uid}.setZoom(Math.max(0.5, zv_{uid}.z-0.25))">−</button>
        <input type="range" id="zs_{uid}" min="0.5" max="4.0" step="0.25" value="1.0"
               oninput="zv_{uid}.setZoom(parseFloat(this.value))">
        <button onclick="zv_{uid}.setZoom(Math.min(4.0, zv_{uid}.z+0.25))">+</button>
        <span id="zl_{uid}">1.0×</span>
        <button onclick="zv_{uid}.setZoom(1.0)">Reset</button>
      </div>
      <div class="cwrap"><canvas id="zc_{uid}" width="{base_width}" height="{base_h}"></canvas></div>
      <div class="cap">{caption} ({orig_w}×{orig_h} px)</div>
    </div>
    <script>
      (function() {{
        const c = document.getElementById('zc_{uid}');
        const ctx = c.getContext('2d');
        const sl = document.getElementById('zs_{uid}');
        const lb = document.getElementById('zl_{uid}');
        const bw = {base_width}, bh = {base_h};
        const img = new Image();
        const ns = {{ z: 1.0 }};
        img.onload = () => {{ ctx.drawImage(img, 0, 0, bw, bh); }};
        img.src = 'data:image/png;base64,{img_b64}';
        ns.setZoom = function(v) {{
          ns.z = Math.round(Math.max(0.5, Math.min(4.0, v)) * 4) / 4;
          c.width = Math.round(bw * ns.z);
          c.height = Math.round(bh * ns.z);
          sl.value = ns.z;
          lb.textContent = ns.z.toFixed(2) + '×';
          if (img.complete) ctx.drawImage(img, 0, 0, c.width, c.height);
        }};
        window['zv_{uid}'] = ns;
      }})();
    </script>
    """
    max_h = int(base_h * 4) + 60
    components.html(html, height=max_h, scrolling=True)

def main():
    load_custom_css()
    st.title("Advanced H&E Cell & Nuclei Quantification Dashboard")
    st.markdown("### Nuclei-Guided Segmentation, Validation, and Manual Correction")
    st.markdown("---")

    # SIDEBAR
    st.sidebar.header("Pipeline Parameters")
    
    with st.sidebar.expander("General Settings", expanded=True):
        micron_val = st.number_input("Micron per Pixel (0 if unknown)", value=0.0, step=0.05)
        mpp = None if micron_val == 0.0 else micron_val
        do_watershed = st.checkbox("Refine large cells with Watershed", value=True)
    
    with st.sidebar.expander("Cellpose Settings", expanded=True):
        cell_diameter = st.number_input("Cell Diameter (px)", value=50, step=5)
        nuc_diameter = st.number_input("Nucleus Diameter (px)", value=int(cell_diameter*0.5), step=5)
        flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, 0.35, 0.05)
        cellprob_threshold = st.slider("Cellprob Threshold", -6.0, 6.0, -1.5, 0.5)

    with st.sidebar.expander("Cell Morphology Filters", expanded=True):
        min_cell_area = st.number_input("Min Cell Area (px)", value=100, step=50)
        max_cell_area = st.number_input("Max Cell Area (px)", value=30000, step=1000)
        min_solidity = st.slider("Min Solidity (reject spindly)", 0.0, 1.0, 0.5, 0.05)
        max_eccentricity = st.slider("Max Eccentricity (reject lines)", 0.0, 1.0, 0.98, 0.01)

    with st.sidebar.expander("Nuclei Morphology Filters", expanded=False):
        min_nuc_area = st.number_input("Min Nucleus Area (px)", value=20, step=10)
        max_nuc_area = st.number_input("Max Nucleus Area (px)", value=10000, step=100)

    st.sidebar.markdown("---")
    do_sweep = st.sidebar.checkbox("Run Parameter Sweep", value=False)
    run_btn = st.sidebar.button("Run Segmentation Pipeline", type="primary")

    try:
        img_path = find_sample_image(INPUT_DIR)
        st.sidebar.success(f"Image loaded: {img_path.name}")
    except FileNotFoundError:
        st.error(f"No images found in {INPUT_DIR}. Please add a sample image.")
        return

    # PIPELINE EXECUTION
    if run_btn:
        with st.spinner("Processing image (Deconvolution, Segmentations, Validation)..."):
            try:
                OUTPUT_DIR.mkdir(exist_ok=True)
                img = io.imread(img_path)
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]

                st.toast("Deconvolving H&E...")
                h_channel, e_channel = extract_he_channels(img)
                io.imsave(OUTPUT_DIR / "he_deconv_hematoxylin.png", h_channel, check_contrast=False)
                io.imsave(OUTPUT_DIR / "he_deconv_eosin.png", e_channel, check_contrast=False)

                nuc_preprocessed = preprocess_for_nuclei(h_channel)
                cyto_preprocessed = preprocess_for_cytoplasm(e_channel)
                io.imsave(OUTPUT_DIR / "preprocessed_nuclei.png", nuc_preprocessed, check_contrast=False)
                io.imsave(OUTPUT_DIR / "preprocessed_for_cellpose.png", cyto_preprocessed, check_contrast=False)

                # NUCLEI
                st.toast("Running Nuclei Segmentation...")
                raw_nuc_masks = run_cellpose(nuc_preprocessed, "nuclei", nuc_diameter, flow_threshold, cellprob_threshold)
                filtered_nuc_masks = filter_and_relabel_mask(raw_nuc_masks, min_nuc_area, max_nuc_area, 0.4, 0.99)
                nuc_props = measure.regionprops(filtered_nuc_masks)
                
                res_nuc = measure_objects_generic(nuc_props, "nucleus")
                df_nuclei = finalize_measurements(res_nuc, mpp)
                df_nuclei.to_csv(OUTPUT_DIR / "nuclei_measurements.csv", index=False)
                io.imsave(OUTPUT_DIR / "nuclei_overlay_filtered.jpg", draw_overlay(img, filtered_nuc_masks, [0, 255, 255]), check_contrast=False)

                # CELLS
                st.toast("Running Cell Segmentation...")
                if do_sweep:
                    st.info("Running parameter sweep... Please wait.")
                    run_sweep(cyto_preprocessed, img)

                raw_cell_masks = run_cellpose(cyto_preprocessed, "cyto3", cell_diameter, flow_threshold, cellprob_threshold)
                
                if do_watershed:
                    st.toast("Running Nuclei-seeded Watershed...")
                    raw_cell_masks = refine_with_watershed(raw_cell_masks, filtered_nuc_masks)
                
                filtered_cell_masks = filter_and_relabel_mask(raw_cell_masks, min_cell_area, max_cell_area, min_solidity, max_eccentricity)
                cell_props = measure.regionprops(filtered_cell_masks)

                st.toast("Validating Cells with Nuclei...")
                res_cells = validate_cells_with_nuclei(cell_props, nuc_props, filtered_cell_masks, filtered_nuc_masks)
                df_cells = finalize_measurements(res_cells, mpp)
                df_cells.to_csv(OUTPUT_DIR / "cell_measurements.csv", index=False)
                
                cell_overlay = draw_overlay(img, filtered_cell_masks, [255, 255, 0])
                io.imsave(OUTPUT_DIR / "cell_overlay_filtered.jpg", cell_overlay, check_contrast=False)
                
                # QC Overlay (Cells yellow, Nuclei cyan)
                qc_overlay = draw_overlay(cell_overlay, filtered_nuc_masks, [0, 255, 255])
                io.imsave(OUTPUT_DIR / "qc_overlay_combined.jpg", qc_overlay, check_contrast=False)

                st.success("Segmentation & Validation Complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())

    # LOAD DATA
    cell_csv_path = OUTPUT_DIR / "cell_measurements.csv"
    nuc_csv_path = OUTPUT_DIR / "nuclei_measurements.csv"
    
    df_c = pd.DataFrame()
    df_n = pd.DataFrame()
    if cell_csv_path.exists(): df_c = pd.read_csv(cell_csv_path)
    if nuc_csv_path.exists(): df_n = pd.read_csv(nuc_csv_path)

    # Fallback for old CSVs that don't have the new metrics yet
    if not df_c.empty and 'confidence_category' not in df_c.columns:
        st.warning("⚠️ Old measurement format detected. Please click 'Run Segmentation Pipeline' in the sidebar to generate the new advanced metrics.")
        for col in ['confidence_category', 'solidity', 'eccentricity', 'circularity', 'nuclei_inside', 'nearest_nucleus_distance_px']:
            if col not in df_c.columns:
                df_c[col] = "Unknown" if col == "confidence_category" else 0.0

    # KPI DASHBOARD
    if not df_c.empty and not df_n.empty:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: metric_card("Total Cells", len(df_c), "value-yellow")
        with c2: metric_card("Total Nuclei", len(df_n), "value-cyan")
        with c3: 
            conf_percent = (len(df_c[df_c['confidence_category'] == 'Confident Cell']) / len(df_c) * 100) if len(df_c)>0 else 0
            metric_card("Confident Cells", f"{conf_percent:.1f}%")
        with c4: 
            fp_risk = (len(df_c[df_c['confidence_category'] == 'Likely False Positive']) / len(df_c) * 100) if len(df_c)>0 else 0
            metric_card("FP Risk", f"{fp_risk:.1f}%", "value-yellow" if fp_risk > 15 else "")
        with c5: metric_card("Mean Cell Area", f"{df_c['area_px'].mean():.0f} px")
        with c6: metric_card("Est Cell Vol", f"{df_c['estimated_volume_px3'].sum():.1e}")

    # TABS
    tab_orig, tab_cyto, tab_nuc, tab_qc, tab_data, tab_sweep, tab_manual = st.tabs([
        "Original", 
        "Cytoplasm/Cells", 
        "Nuclei", 
        "QC Overlay", 
        "Data & Export",
        "Sweep Comparison",
        "Manual Correction"
    ])

    try:
        orig = io.imread(img_path)
    except:
        orig = None

    with tab_orig:
        if orig is not None:
            zoomable_image(orig, caption="Original H&E Core", uid="orig")

    with tab_cyto:
        c1, c2, c3 = st.columns(3)
        with c1:
            if (OUTPUT_DIR / "he_deconv_eosin.png").exists():
                zoomable_image(OUTPUT_DIR / "he_deconv_eosin.png", caption="Deconvolved Eosin (Cytoplasm)", uid="eosin", base_width=400)
        with c2:
            if (OUTPUT_DIR / "preprocessed_for_cellpose.png").exists():
                zoomable_image(OUTPUT_DIR / "preprocessed_for_cellpose.png", caption="CLAHE Preprocessed", uid="cyto_clahe", base_width=400)
        with c3:
            if (OUTPUT_DIR / "cell_overlay_filtered.jpg").exists():
                zoomable_image(OUTPUT_DIR / "cell_overlay_filtered.jpg", caption="Cell Boundaries", uid="cell_ov", base_width=400)

    with tab_nuc:
        c1, c2, c3 = st.columns(3)
        with c1:
            if (OUTPUT_DIR / "he_deconv_hematoxylin.png").exists():
                zoomable_image(OUTPUT_DIR / "he_deconv_hematoxylin.png", caption="Deconvolved Hematoxylin (Nuclei)", uid="hema", base_width=400)
        with c2:
            if (OUTPUT_DIR / "preprocessed_nuclei.png").exists():
                zoomable_image(OUTPUT_DIR / "preprocessed_nuclei.png", caption="CLAHE Preprocessed", uid="nuc_clahe", base_width=400)
        with c3:
            if (OUTPUT_DIR / "nuclei_overlay_filtered.jpg").exists():
                zoomable_image(OUTPUT_DIR / "nuclei_overlay_filtered.jpg", caption="Nuclei Boundaries", uid="nuc_ov", base_width=400)

    with tab_qc:
        st.subheader("Validation Overlay")
        if (OUTPUT_DIR / "qc_overlay_combined.jpg").exists():
            zoomable_image(OUTPUT_DIR / "qc_overlay_combined.jpg", caption="Cells (Yellow) + Nuclei (Cyan)", uid="qc")
        else:
            st.info("Run pipeline to view QC overlay.")

    with tab_data:
        if not df_c.empty:
            st.subheader("Cell Measurements & Confidence Categories")
            
            display_cols = ['object_id', 'confidence_category', 'area_px', 'solidity', 'eccentricity', 
                            'circularity', 'nuclei_inside', 'nearest_nucleus_distance_px']
            st.dataframe(df_c[display_cols].style.map(
                lambda val: 'color: #39D353' if val == 'Confident Cell' else ('color: #F1E05A' if val == 'Possible Cell' else 'color: #FF7B72'),
                subset=['confidence_category']
            ), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                csv_data_c = df_c.to_csv(index=False).encode('utf-8')
                st.download_button("Download Cell Data (CSV)", csv_data_c, "cell_measurements.csv", "text/csv", type="primary")
            with c2:
                if not df_n.empty:
                    csv_data_n = df_n.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Nuclei Data (CSV)", csv_data_n, "nuclei_measurements.csv", "text/csv")
            
            st.markdown("---")
            fig = px.scatter(df_c, x="area_px", y="solidity", color="confidence_category",
                             color_discrete_map={"Confident Cell": "#39D353", "Possible Cell": "#F1E05A", "Likely False Positive": "#FF7B72"},
                             hover_data=['object_id', 'nuclei_inside'])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#C9D1D9")
            st.plotly_chart(fig, use_container_width=True)

    with tab_sweep:
        sweep_csv = SWEEP_DIR / "sweep_results.csv"
        if sweep_csv.exists():
            df_sweep = pd.read_csv(sweep_csv)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(df_sweep.sort_values("cell_count", ascending=False), use_container_width=True, hide_index=True)
                sweep_options = df_sweep.apply(lambda r: f"d{r['diameter']}_f{r['flow_threshold']}_p{r['cellprob_threshold']} ({r['cell_count']} cells)", axis=1).tolist()
                selected_sweep = st.selectbox("View Sweep Overlay", sweep_options)
                idx = sweep_options.index(selected_sweep)
                selected_overlay_path = df_sweep.iloc[idx]["overlay_path"]
            with c2:
                if Path(selected_overlay_path).exists():
                    zoomable_image(selected_overlay_path, caption="Sweep Result", uid="sweep_ov")
        else:
            st.info("No sweep data found.")

    with tab_manual:
        st.subheader("Manual Correction Marking")
        st.write("Click to mark False Positives (red ●) or Missed Cells (green ●). Use freedraw for freeform marking. Zoom in/out to inspect details.")

        if (OUTPUT_DIR / "qc_overlay_combined.jpg").exists():
            import base64 as b64lib
            from PIL import Image as PILImage
            from io import BytesIO

            bg_pil = PILImage.open(OUTPUT_DIR / "qc_overlay_combined.jpg").convert("RGB")
            orig_w, orig_h = bg_pil.size

            # We send the full-resolution image to the browser and let JS handle zoom.
            # Cap at 2000px wide to keep the base64 payload reasonable.
            send_w = min(orig_w, 2000)
            send_scale = send_w / orig_w
            send_h = int(orig_h * send_scale)
            bg_send = bg_pil.resize((send_w, send_h))

            buf = BytesIO()
            bg_send.save(buf, format="PNG")
            img_b64 = b64lib.b64encode(buf.getvalue()).decode()

            # Initial canvas size (1.0x zoom maps to base_w)
            base_w = 900
            base_h = int(orig_h * base_w / orig_w)

            canvas_html = f"""
            <style>
              body {{ margin:0; padding:0; background:#0E1117; font-family:-apple-system,sans-serif; color:#C9D1D9; }}
              #toolbar {{ display:flex; gap:10px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }}
              #toolbar button {{
                background:#21262D; color:#C9D1D9; border:1px solid #30363D; padding:5px 12px;
                border-radius:6px; cursor:pointer; font-size:13px;
              }}
              #toolbar button:hover {{ background:#30363D; }}
              #toolbar button.active {{ background:#1F6FEB; border-color:#58A6FF; color:#fff; }}
              .sep {{ color:#30363D; font-size:16px; }}
              #zoomRow {{ display:flex; gap:10px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }}
              #zoomRow input[type=range] {{ width:180px; accent-color:#58A6FF; }}
              #zoomRow span {{ font-size:13px; color:#8B949E; }}
              #zoomRow button {{ background:#21262D; color:#C9D1D9; border:1px solid #30363D; padding:3px 10px; border-radius:5px; cursor:pointer; font-size:13px; }}
              #canvasWrap {{ overflow:auto; border:1px solid #30363D; border-radius:6px; max-height:75vh; }}
              #annotCanvas {{ display:block; cursor:crosshair; }}
              #status {{ margin-top:6px; font-size:12px; color:#8B949E; min-height:18px; }}
              #sizeInfo {{ font-size:11px; color:#484F58; margin-top:2px; }}
            </style>

            <div id="toolbar">
              <button id="btnPoint" class="active" onclick="setMode('point')">● Point</button>
              <button id="btnDraw" onclick="setMode('freedraw')">✏ Freedraw</button>
              <span class="sep">│</span>
              <button id="btnFP" class="active" onclick="setType('fp')" style="color:#FF6B6B">● False Positive</button>
              <button id="btnMC" onclick="setType('mc')" style="color:#69DB7C">● Missed Cell</button>
              <span class="sep">│</span>
              <button onclick="undoLast()">↩ Undo</button>
              <button onclick="clearAll()">✕ Clear</button>
              <button onclick="downloadJSON()" style="background:#238636;border-color:#2EA043;color:#fff">⬇ Download JSON</button>
            </div>

            <div id="zoomRow">
              <button onclick="setZoom(Math.max(0.5, zoom-0.25))">−</button>
              <input type="range" id="zoomSlider" min="0.5" max="4.0" step="0.25" value="1.0"
                     oninput="setZoom(parseFloat(this.value))">
              <button onclick="setZoom(Math.min(4.0, zoom+0.25))">+</button>
              <span id="zoomLabel">1.0×</span>
              <button onclick="setZoom(1.0)" style="margin-left:8px">Reset</button>
              <button onclick="fitToWindow()">Fit</button>
            </div>

            <div id="canvasWrap">
              <canvas id="annotCanvas" width="{base_w}" height="{base_h}"></canvas>
            </div>
            <div id="status">Ready — click on the image to annotate</div>
            <div id="sizeInfo">Original: {orig_w}×{orig_h} px | Display: {base_w}×{base_h} px | Zoom: 1.0×</div>

            <script>
              const canvas = document.getElementById('annotCanvas');
              const ctx = canvas.getContext('2d');
              const statusEl = document.getElementById('status');
              const sizeInfoEl = document.getElementById('sizeInfo');
              const zoomSlider = document.getElementById('zoomSlider');
              const zoomLabel = document.getElementById('zoomLabel');

              const ORIG_W = {orig_w};
              const ORIG_H = {orig_h};
              const BASE_W = {base_w};
              const BASE_H = {base_h};

              let zoom = 1.0;
              let mode = 'point';
              let corrType = 'fp';
              let drawing = false;
              let currentPath = [];

              // All annotations store coordinates in ORIGINAL image space
              let annotations = [];

              const bgImg = new window.Image();
              bgImg.onload = () => {{ redraw(); statusEl.textContent = 'Image loaded — ready to annotate'; }};
              bgImg.src = 'data:image/png;base64,{img_b64}';

              function dispW() {{ return Math.round(BASE_W * zoom); }}
              function dispH() {{ return Math.round(BASE_H * zoom); }}
              function scaleX() {{ return ORIG_W / dispW(); }}
              function scaleY() {{ return ORIG_H / dispH(); }}
              // Convert original coords to current display coords
              function toDisp(ox, oy) {{ return {{ x: ox / scaleX(), y: oy / scaleY() }}; }}
              // Convert display coords to original coords
              function toOrig(dx, dy) {{ return {{ x: Math.round(dx * scaleX()), y: Math.round(dy * scaleY()) }}; }}

              function getColor() {{ return corrType === 'fp' ? '#FF4444' : '#44FF44'; }}
              function getLabel() {{ return corrType === 'fp' ? 'False Positive' : 'Missed Cell'; }}

              function setZoom(z) {{
                zoom = Math.round(z * 4) / 4;  // snap to 0.25 steps
                zoom = Math.max(0.5, Math.min(4.0, zoom));
                canvas.width = dispW();
                canvas.height = dispH();
                zoomSlider.value = zoom;
                zoomLabel.textContent = zoom.toFixed(2) + '×';
                sizeInfoEl.textContent = `Original: ${{ORIG_W}}×${{ORIG_H}} px | Display: ${{dispW()}}×${{dispH()}} px | Zoom: ${{zoom.toFixed(2)}}×`;
                redraw();
              }}

              function fitToWindow() {{
                // Fit to the wrapper width (~900px for most screens)
                const wrap = document.getElementById('canvasWrap');
                const fitZoom = Math.min((wrap.clientWidth - 4) / BASE_W, 1.0);
                setZoom(Math.max(0.5, fitZoom));
              }}

              function setMode(m) {{
                mode = m;
                document.getElementById('btnPoint').classList.toggle('active', m === 'point');
                document.getElementById('btnDraw').classList.toggle('active', m === 'freedraw');
                canvas.style.cursor = m === 'point' ? 'crosshair' : 'default';
                statusEl.textContent = 'Mode: ' + m;
              }}

              function setType(t) {{
                corrType = t;
                document.getElementById('btnFP').classList.toggle('active', t === 'fp');
                document.getElementById('btnMC').classList.toggle('active', t === 'mc');
                statusEl.textContent = 'Type: ' + getLabel();
              }}

              function getPos(e) {{
                const rect = canvas.getBoundingClientRect();
                return {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};
              }}

              canvas.addEventListener('mousedown', (e) => {{
                const pos = getPos(e);
                const orig = toOrig(pos.x, pos.y);
                if (mode === 'point') {{
                  annotations.push({{ type: 'point', label: getLabel(), orig_x: orig.x, orig_y: orig.y, color: getColor() }});
                  redraw();
                  statusEl.textContent = `Marked ${{getLabel()}} at orig(${{orig.x}}, ${{orig.y}}) — Total: ${{annotations.length}}`;
                }} else if (mode === 'freedraw') {{
                  drawing = true;
                  currentPath = [orig];
                  ctx.beginPath();
                  ctx.moveTo(pos.x, pos.y);
                }}
              }});

              canvas.addEventListener('mousemove', (e) => {{
                if (!drawing) return;
                const pos = getPos(e);
                currentPath.push(toOrig(pos.x, pos.y));
                ctx.strokeStyle = getColor();
                ctx.lineWidth = Math.max(2, 3 * zoom);
                ctx.lineCap = 'round';
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
              }});

              canvas.addEventListener('mouseup', () => {{
                if (!drawing) return;
                drawing = false;
                if (currentPath.length > 1) {{
                  annotations.push({{ type: 'freedraw', label: getLabel(), color: getColor(), orig_points: currentPath }});
                  statusEl.textContent = `Freedraw ${{getLabel()}} saved — Total: ${{annotations.length}}`;
                }}
                currentPath = [];
              }});

              function redraw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (bgImg.complete) ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);

                const markerR = Math.max(5, 8 * zoom);
                const crossR = Math.max(8, 12 * zoom);

                annotations.forEach(a => {{
                  if (a.type === 'point') {{
                    const d = toDisp(a.orig_x, a.orig_y);
                    ctx.beginPath();
                    ctx.arc(d.x, d.y, markerR, 0, 2 * Math.PI);
                    ctx.fillStyle = a.color + '88';
                    ctx.fill();
                    ctx.strokeStyle = a.color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(d.x - crossR, d.y); ctx.lineTo(d.x + crossR, d.y);
                    ctx.moveTo(d.x, d.y - crossR); ctx.lineTo(d.x, d.y + crossR);
                    ctx.strokeStyle = a.color;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                  }} else if (a.type === 'freedraw') {{
                    ctx.beginPath();
                    a.orig_points.forEach((p, i) => {{
                      const d = toDisp(p.x, p.y);
                      i === 0 ? ctx.moveTo(d.x, d.y) : ctx.lineTo(d.x, d.y);
                    }});
                    ctx.strokeStyle = a.color;
                    ctx.lineWidth = Math.max(2, 3 * zoom);
                    ctx.lineCap = 'round';
                    ctx.stroke();
                  }}
                }});
              }}

              function undoLast() {{
                annotations.pop();
                redraw();
                statusEl.textContent = `Undo — Total: ${{annotations.length}}`;
              }}

              function clearAll() {{
                annotations = [];
                redraw();
                statusEl.textContent = 'Cleared all annotations';
              }}

              function downloadJSON() {{
                const payload = {{
                  annotations: annotations,
                  original_size: {{ width: ORIG_W, height: ORIG_H }},
                  zoom_at_export: zoom,
                  timestamp: new Date().toISOString()
                }};
                const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'manual_corrections.json';
                a.click();
                statusEl.textContent = 'Downloaded manual_corrections.json';
              }}
            </script>
            """

            # Max canvas height at 4x zoom + toolbar
            max_iframe_h = int(base_h * 4) + 120
            components.html(canvas_html, height=max_iframe_h, scrolling=True)

        else:
            st.info("Run the pipeline first to generate the QC overlay.")

if __name__ == "__main__":
    main()
