"""
app.py

Usage:
    streamlit run app.py
"""
import io
import json
import time
import cv2
import numpy as np
import torch
import yaml
import matplotlib.cm as cm
import streamlit as st
from pathlib import Path
from PIL import Image

from src.inference import build_model, infer_full_tile
from src.postprocess import postprocess_pipeline, extract_building_polygons
from scripts.vectorize_to_geojson import read_geotransform, build_geojson

st.set_page_config(page_title="Building Segmentation", layout="wide")
st.title("Building Segmentation")


def find_experiments():
    configs_dir = Path("configs")
    checkpoints_dir = Path("checkpoints")
    experiments = []

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        exp_name = cfg.get("experiment_name", cfg_path.stem)
        ckpt_path = checkpoints_dir / f"{exp_name}_best.pth"

        if ckpt_path.exists():
            arch = cfg["model"]["architecture"]
            encoder = cfg["model"]["encoder_name"]
            label = f"{arch} / {encoder}"
            experiments.append((label, cfg_path, ckpt_path, cfg))

    return experiments


experiments = find_experiments()

if not experiments:
    st.error("No paired (config + checkpoint) found. "
             "Make sure configs/ and checkpoints/ exist and names match.")
    st.stop()


st.sidebar.header("Model")

labels = [e[0] for e in experiments]
choice = st.sidebar.selectbox("Architecture", range(len(labels)),
                               format_func=lambda i: labels[i])

_, config_path, checkpoint_path, cfg = experiments[choice]

st.sidebar.divider()
st.sidebar.header("Inference")

threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Probability cutoff for binary mask. Higher = stricter (fewer false positives).")

tta = st.sidebar.checkbox("TTA (slower, more accurate)", value=False,
                          help="Averages predictions over 4 rotations and 2 flips.")

show_pipeline = st.sidebar.checkbox("Show postprocessing steps", value=False,
                                    help="Shows morphology cleanup and polygon extraction steps.")

show_vertices = st.sidebar.checkbox("Show polygon vertices", value=False,
                                    help="Marks each polygon vertex with a red dot on the overlay.")

min_area = st.sidebar.slider("Min building area (px)", 10, 200, 40, 5,
                             help="Contours smaller than this are discarded as noise.")

simplify_frac = st.sidebar.slider("Simplify (Douglas-Peucker)", 0.001, 0.03, 0.005, 0.001, format="%.3f",
                                  help="How aggressively to simplify polygon shape. Higher = fewer vertices.")

denoise_px = st.sidebar.slider("Denoise (px)", 0.0, 5.0, 3.0, 0.5,
                               help="Removes pixel-staircase noise on straight edges before shape simplification.")


@st.cache_resource(show_spinner="Loading model...")
def load(config_str, checkpoint_str):
    with open(config_str, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(c, checkpoint_str, device)
    return model, device

model, device = load(str(config_path), str(checkpoint_path))
st.sidebar.caption(f"{'GPU' if device.type == 'cuda' else 'CPU'}")


uploaded = st.file_uploader(
    "Upload aerial image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is None:
    st.info("Upload an aerial image to run inference.")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_rgb = np.array(img_pil)
img_bgr = img_rgb[:, :, ::-1].copy()

h, w = img_bgr.shape[:2]
st.caption(f"{w}×{h} px")

transform_fn, crs = None, None
if uploaded.name.lower().endswith((".tif", ".tiff")):
    transform_fn, crs = read_geotransform(io.BytesIO(uploaded.getvalue()))

run = st.button("Run model")

if not run:
    st.stop()

with st.spinner("Running inference..."):
    patch = cfg["dataset"]["patch_size"]
    start_time = time.time()
    prob = infer_full_tile(model, img_bgr, patch, overlap=128,
                             device=device, batch_size=4, tta=tta)
    elapsed = time.time() - start_time
    binary = (prob > threshold).astype(np.uint8) * 255

conf_rgb = (cm.jet(prob)[:, :, :3] * 255).astype(np.uint8)


if show_pipeline:
    morph, approx, polys = postprocess_pipeline(binary,
                                                min_area=min_area,
                                                simplify_frac=simplify_frac,
                                                denoise_px=denoise_px)
    overlay = img_rgb.copy()
    green = np.zeros_like(img_rgb)
    for poly, _ in polys:
        cv2.fillPoly(green, [poly.astype(np.int32)], (0, 255, 0))
    cv2.addWeighted(green, 0.4, overlay, 1.0, 0, overlay)

    if show_vertices:
        for poly, _ in polys:
            for x, y in poly:
                cv2.circle(overlay, (int(x), int(y)), 3, (255, 0, 0), -1)

    cols = st.columns(5)
    cols[0].image(img_rgb, caption="Image", use_container_width=True)
    cols[1].image(conf_rgb, caption="Confidence", use_container_width=True)
    cols[2].image(binary, caption="Raw mask", use_container_width=True)
    cols[3].image(morph, caption="Morphology", use_container_width=True)
    cols[4].image(overlay, caption="Douglas-Peucker", use_container_width=True)
else:
    polys = extract_building_polygons(binary, min_area=40)
    cols = st.columns(3)
    cols[0].image(img_rgb, caption="Image", use_container_width=True)
    cols[1].image(conf_rgb, caption=f"Confidence (t={threshold})", use_container_width=True)
    cols[2].image(binary, caption="Predicted mask",  use_container_width=True)


st.caption(f"Inference time: {elapsed:.2f}s | Buildings detected: {len(polys)}")

buf = io.BytesIO()
Image.fromarray(binary).save(buf, format="PNG")
st.download_button("Download mask", buf.getvalue(), "mask.png", "image/png")

geojson_fc, wgs84 = build_geojson(polys, transform_fn, crs, uploaded.name)
crs_label = "WGS84" if wgs84 else (f"UTM EPSG:{crs}" if transform_fn else "pixel coordinates")
st.download_button(
    "Download GeoJSON",
    json.dumps(geojson_fc),
    "buildings.geojson",
    "application/geo+json",
)
st.caption(f"GeoJSON coordinates: {crs_label}")
