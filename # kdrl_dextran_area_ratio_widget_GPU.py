# kdrl_dextran_area_ratio_widget.py
"""
GPU‑accelerated (pyclesperanto) Napari‑magicgui widget
to measure the area ratio between KDRL (green) and Dextran‑70 kDa (red)
in a 2‑channel still image.

Highlights
──────────
• Gaussian blur, morphological ops, Otsu thresholding on GPU
• Green‑channel background subtraction + optional CLAHE
• Interactive ROI deletion in green mask
• Version‑safe small‑object removal (CPU fallback → always works)
• CSV export of green / red areas and ratio
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
import napari
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
import pyclesperanto_prototype as cle
from skimage import exposure               # for CLAHE
from skimage.measure import label as sklabel, regionprops

viewer = napari.Viewer()
state = {
    "folder": None,
    "file": None,
    "data": None,
    "masks": {"green": None, "red": None},
}

# -------------------------------------------------------------------- #
#                               HELPERS                                #
# -------------------------------------------------------------------- #
def _list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png"})


def _read_two_channel(path: Path):
    arr = tifffile.imread(str(path)).astype("float32")
    if arr.ndim != 3 or 2 not in arr.shape:
        raise ValueError(f"{path.name}: expected 2‑channel still, got {arr.shape}")
    arr = arr if arr.shape[-1] == 2 else arr.transpose(1, 2, 0)
    arr -= arr.min(axis=(0, 1), keepdims=True)
    arr /= arr.max(axis=(0, 1), keepdims=True) + 1e-9
    return arr


def _preprocess_green(img, bg_radius: int, clahe: bool):
    gpu = cle.push(img)
    if bg_radius > 0:
        gpu = cle.top_hat_box(gpu, radius_x=bg_radius, radius_y=bg_radius)
    img = cle.pull(gpu)
    if clahe:
        img = exposure.equalize_adapthist(img, clip_limit=0.02)
    return img


def _segment(img, sigma: float, threshold: float, min_size: int) -> np.ndarray:
    """Return binary mask (uint8) where 1 = object pixels."""
    gpu = cle.push(img)

    if sigma > 0:
        gpu = cle.gaussian_blur(gpu, sigma_x=sigma, sigma_y=sigma)

    mask_gpu = cle.threshold_otsu(gpu) if threshold == 0 else cle.greater_constant(gpu, constant=threshold)
    mask_gpu = cle.erode_sphere(mask_gpu)

    # ---------------------------------------------------------------- #
    #  Version‑agnostic small‑object removal (CPU)                     #
    # ---------------------------------------------------------------- #
    mask_np = cle.pull(mask_gpu).astype(bool)
    labeled = sklabel(mask_np)
    keep = {prop.label for prop in regionprops(labeled) if prop.area >= min_size}
    filtered = np.isin(labeled, list(keep)).astype(np.uint8)

    return filtered


# -------------------------------------------------------------------- #
#                          MAGICGUI WIDGETS                            #
# -------------------------------------------------------------------- #
@magicgui(directory={"mode": "d"}, call_button="Set Folder")
def choose_folder(directory: Path):
    state["folder"] = directory
    files = _list_images(directory)
    file_select.file.choices = [f.name for f in files]
    print(f"📁 Folder: {directory}  ({len(files)} images)")


@magicgui(file={"widget_type": "ComboBox"}, call_button="Load Image")
def file_select(file: str):
    path = state["folder"] / file
    data = _read_two_channel(path)
    state.update({"file": path, "data": data, "masks": {"green": None, "red": None}})
    viewer.layers.clear()
    viewer.add_image(data[..., 0], name="KDRL – green", colormap="green")
    viewer.add_image(data[..., 1], name="Dextran – red", colormap="magenta")
    print(f"📂 Loaded {path.name}")


def make_seg_widget(channel_key: str, idx: int):
    extras = (
        {
            "bg_radius": {"label": "BG radius px", "widget_type": "IntSlider", "min": 0, "max": 100, "step": 1, "value": 15},
            "clahe": {"label": "CLAHE", "widget_type": "CheckBox", "value": False},
        }
        if channel_key == "green"
        else {}
    )

    @magicgui(
        call_button=f"Segment {channel_key}",
        threshold={"label": "Thresh (0→Otsu)", "widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.01, "value": 0},
        sigma={"widget_type": "FloatSlider", "min": 0, "max": 5, "step": 0.1, "value": 1},
        min_size={"widget_type": "IntSlider", "min": 5, "max": 20000, "step": 5, "value": 50},
        **extras,
    )
    def segment(threshold=0.0, sigma=1.0, min_size=50, bg_radius=15, clahe=False):
        if state["data"] is None:
            QMessageBox.warning(viewer.window.qt_viewer, "Error", "Load an image first.")
            return

        img = state["data"][..., idx]
        if channel_key == "green":
            img = _preprocess_green(img, bg_radius, clahe)

        mask = _segment(img, sigma, threshold, min_size)

        # replace previous mask if it exists
        if (prev := state["masks"].get(channel_key)) and prev in viewer.layers:
            viewer.layers.remove(prev)

        lbl = viewer.add_labels(
            mask,
            name=f"{channel_key} mask",
            color={1: "lime" if channel_key == "green" else "yellow"},
        )
        lbl.metadata["area"] = int(mask.sum())
        state["masks"][channel_key] = lbl

    return segment


@magicgui(call_button="❌ Remove Selected ROIs")
def remove_green_labels():
    mask_layer = state["masks"].get("green")
    if not mask_layer:
        QMessageBox.warning(viewer.window.qt_viewer, "Error", "No green mask present.")
        return
    sel = mask_layer.selected_label
    labels = [sel] if isinstance(sel, int) else sel
    data = mask_layer.data.copy()
    for lbl in labels:
        if lbl > 0:
            data[data == lbl] = 0
    mask_layer.data = data
    mask_layer.metadata["area"] = int((data > 0).sum())
    print(f"🧽 Removed labels: {labels}. New area = {mask_layer.metadata['area']}")


@magicgui(call_button="Compute Area Ratio")
def compute_ratio():
    g, r = state["masks"].get("green"), state["masks"].get("red")
    if not g or not r:
        QMessageBox.warning(viewer.window.qt_viewer, "Missing masks", "Segment both channels first.")
        return

    ag, ar = g.metadata.get("area", 0), r.metadata.get("area", 0)
    if ar == 0:
        QMessageBox.warning(viewer.window.qt_viewer, "Error", "Red mask area is zero.")
        return

    ratio = ag / ar
    input_name = state["file"].stem  # e.g. "image001" from image001.tif
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{input_name}_area_ratio_{ts}.csv"
    csv_path = state["folder"] / csv_name

    df = pd.DataFrame({
        "file": [state["file"].name],
        "green_px": [ag],
        "red_px": [ar],
        "green/red": [ratio],
    })
    df.to_csv(csv_path, index=False)

    print(f"✅ Ratio = {ratio:.4f}\n📄 Saved → {csv_path}")
    QMessageBox.information(
        viewer.window.qt_viewer,
        "Area ratio",
        f"Green/Red = {ratio:.4f}\nCSV saved to:\n{csv_path}",
    )


# -------------------------------------------------------------------- #
#                      ADD WIDGETS AND START NAPARI                    #
# -------------------------------------------------------------------- #
viewer.window.add_dock_widget(choose_folder, area="right")
viewer.window.add_dock_widget(file_select, area="right")
viewer.window.add_dock_widget(make_seg_widget("green", 0), area="right")
viewer.window.add_dock_widget(remove_green_labels, area="right")
viewer.window.add_dock_widget(make_seg_widget("red", 1), area="right")
viewer.window.add_dock_widget(compute_ratio, area="right")

napari.run()
