import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage import filters, measure, morphology, exposure
from tkinter import filedialog, Tk
from scipy.ndimage import gaussian_filter
import tifffile
import napari
import pandas as pd
from magicgui import magicgui

# Qt helpers
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox

# Optional: pyclesperanto GPU support
try:
    import pyclesperanto_prototype as cle
    CLE_AVAILABLE = True
    cle.select_device("opencl")
    print(f"⚡ Using GPU: {cle.get_device()}")
except ImportError:
    CLE_AVAILABLE = False
    print("⚠️ pyclesperanto not found. Falling back to CPU filtering.")

def gpu_gaussian_blur(image, sigma):
    if not CLE_AVAILABLE:
        return gaussian_filter(image, sigma=sigma)
    img_gpu = cle.push(image.astype(np.float32))
    blurred = cle.gaussian_blur(img_gpu, sigma_x=sigma, sigma_y=sigma)
    return cle.pull(blurred)

# ---------------------- File Selection Dialog ----------------------
root = Tk()
root.withdraw()
tif_path = filedialog.askopenfilename(
    title="Select 1- or 2-channel time-lapse TIFF",
    filetypes=[("TIFF files", "*.tif *.tiff")]
)
if not tif_path:
    raise SystemExit("❌ No file selected.")

print(f"📂 Loading: {os.path.basename(tif_path)}")
img_stack = tifffile.imread(tif_path).astype(np.float32)

# ---------------------- Determine Image Shape ----------------------
if img_stack.ndim == 4:
    if img_stack.shape[1] == 2:
        img_stack = np.transpose(img_stack, (0, 2, 3, 1))
    elif img_stack.shape[-1] == 2:
        pass
    else:
        raise ValueError("Expected 2 channels in either second or last dimension.")
    n_frames, height, width, _ = img_stack.shape
    leakage = img_stack[..., 1]
    vessel = img_stack[0, ..., 0]
    vessel_available = True
elif img_stack.ndim == 3:
    n_frames, height, width = img_stack.shape
    leakage = img_stack
    vessel_available = False
else:
    raise ValueError("Unsupported image shape.")

# Normalize Green Channel (for raw histogram)
global_min = np.percentile(leakage, 1)
global_max = np.percentile(leakage, 99)
img_norm = np.clip((leakage - global_min) / (global_max - global_min), 0, 1)

# Will be set after widget 1 runs
_last_kymo_img = {"img": None}

# ---------------------- Napari Viewer ----------------------
viewer = napari.Viewer()
time_layer = viewer.add_image(
    img_norm,
    name="Time-lapse (Green Channel)",
    colormap="inferno",
    contrast_limits=(0, 1),
    cache=True,
    blending="translucent_no_depth",   # default to match screenshot
    interpolation="nearest",
)

# ---------------------- Leakage Map Widget (Widget 1) ----------------------
@magicgui(
    delta_threshold={"label": "∆ Threshold", "min": 0, "max": 1000, "step": 10, "widget_type": "Slider"},
    lower_norm_cutoff={"label": "Lower ∆ norm cutoff", "min": 0.0, "max": 0.5, "step": 0.01},
    smoothing_enabled={"label": "Smooth Leakage Map"},
    background_subtract={"label": "Subtract Background First"},
    call_button="Apply and Save"
)
def leakage_widget(
    delta_threshold: int = 100,
    lower_norm_cutoff: float = 0.15,      # screenshot default
    smoothing_enabled: bool = True,
    background_subtract: bool = False
):
    data = leakage.copy()
    if background_subtract:
        background = np.min(data, axis=0)
        data = data - background

    delta_stack = data[1:] - data[:-1]
    rise_map = np.argmax(delta_stack > delta_threshold, axis=0)
    valid_mask = np.max(delta_stack, axis=0) > delta_threshold

    # Margins: left=10, right/top/bottom=100
    top, bottom, left, right = 100, 100, 10, 100
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[top:height - bottom, left:width - right] = True

    leakage_map = np.full_like(rise_map, np.nan, dtype=np.float32)
    leakage_map[valid_mask & edge_mask] = rise_map[valid_mask & edge_mask]

    # Vessel outline (optional)
    vessel_contours = []
    if vessel_available:
        vessel_smooth = gpu_gaussian_blur(vessel, sigma=1)
        vessel_thresh = filters.threshold_otsu(vessel_smooth)
        vessel_mask = morphology.remove_small_objects(vessel_smooth > (0.75 * vessel_thresh), 32)
        vessel_mask = morphology.binary_dilation(vessel_mask, morphology.disk(2))
        vessel_contours = measure.find_contours(vessel_mask, level=0.5)

    leak_vis = gpu_gaussian_blur(leakage_map, sigma=0.5) if smoothing_enabled else leakage_map
    basepath = os.path.splitext(tif_path)[0]
    svg_path = basepath + "_leakage_map.svg"
    png_path = basepath + "_leakage_map.png"

    # Save raw + smoothed maps
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(leakage_map, cmap='turbo'); cbar0 = fig.colorbar(im0, ax=axs[0]); cbar0.set_label("Frame of Onset")
    axs[0].set_title("Leakage Map (Raw)")
    im1 = axs[1].imshow(leak_vis, cmap='magma'); cbar1 = fig.colorbar(im1, ax=axs[1]); cbar1.set_label("Frame of Onset")
    axs[1].set_title("Leakage Map (Smoothed)")
    for c in vessel_contours:
        axs[0].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.0)
        axs[1].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.0)
    fig.tight_layout()
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"📄 Saved SVG: {svg_path}")
    print(f"📄 Saved PNG: {png_path}")

    # Rainbow ∆ figure + CSV area metrics
    max_delta = np.max(delta_stack, axis=0)
    threshold_mask = max_delta > delta_threshold
    norm_max_delta = np.zeros_like(max_delta)
    norm_max_delta[threshold_mask] = np.power(
        (max_delta[threshold_mask] - np.min(max_delta[threshold_mask])) /
        (np.max(max_delta[threshold_mask]) - np.min(max_delta[threshold_mask]) + 1e-6),
        0.5
    )
    mask_selected = (norm_max_delta >= lower_norm_cutoff) & (norm_max_delta <= 0.6) & edge_mask
    area_selected = int(np.sum(mask_selected))
    area_total_valid = int(np.sum((norm_max_delta > 0) & edge_mask))

    csv_path = basepath + "_delta_area.csv"
    pd.DataFrame({
        "Filename": [os.path.basename(tif_path)],
        f"Area_{lower_norm_cutoff:.2f}_to_0.6_norm_delta (px)": [area_selected],
        "Total Valid Norm ∆ Area in Cropped Field (px)": [area_total_valid]
    }).to_csv(csv_path, index=False)
    print(f"📄 Saved CSV: {csv_path}")

    svg_path_delta = basepath + "_delta_intensity_map.svg"
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    imd0 = axs[0].imshow(max_delta, cmap='hot'); cbar0 = fig.colorbar(imd0, ax=axs[0]); cbar0.set_label("Raw ∆ Intensity")
    imd1 = axs[1].imshow(norm_max_delta, cmap='hot'); cbar1 = fig.colorbar(imd1, ax=axs[1]); cbar1.set_label("Normalized ∆")
    imd2 = axs[2].imshow(norm_max_delta, cmap='jet', vmin=0.01, vmax=0.6); cbar2 = fig.colorbar(imd2, ax=axs[2]); cbar2.set_label("Rainbow (Norm ∆)")
    axs[0].set_title("Raw ∆"); axs[1].set_title("Normalized ∆"); axs[2].set_title("Rainbow ∆")
    for ax in axs:
        for c in vessel_contours:
            ax.plot(c[:, 1], c[:, 0], 'w-', linewidth=1.0)
    fig.tight_layout()
    fig.savefig(svg_path_delta, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"📄 Saved SVG: {svg_path_delta}")

    # --- Add/Update leakage layer with screenshot defaults ---
    if "Leakage Initiation Map" in viewer.layers:
        layer = viewer.layers["Leakage Initiation Map"]
        layer.data = leakage_map
        layer.colormap = "turbo"
        layer.opacity = 0.70
        layer.blending = "translucent_no_depth"
        layer.interpolation = "nearest"
        layer.contrast_limits = (0, n_frames - 2)
    else:
        viewer.add_image(
            leakage_map,
            name="Leakage Initiation Map",
            colormap="turbo",
            opacity=0.70,
            contrast_limits=(0, n_frames - 2),
            blending="translucent_no_depth",
            interpolation="nearest",
        )
    viewer.reset_view()

    # ---- Calculate and store the min-max kymograph-normalized image ----
    kymo_img = np.max(data, axis=0) - np.min(data, axis=0)
    if np.max(kymo_img) != np.min(kymo_img):
        kymo_norm = (kymo_img - np.min(kymo_img)) / (np.max(kymo_img) - np.min(kymo_img) + 1e-6)
    else:
        kymo_norm = np.zeros_like(kymo_img)
    _last_kymo_img["img"] = kymo_norm

# --------------- Histogram/Preview widget for Outline Quantiles ---------------
class QuantileHistogramWidget(QWidget):
    def __init__(self, kymo_img, parent=None):
        super().__init__(parent)
        self.fig, self.ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Histogram: kymograph min-max image"))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.update_histogram(kymo_img, 0.01, 0.25, 0.13)  # defaults

    def update_histogram(self, kymo_img, low_q, high_q, abs_thresh):
        self.ax.clear()
        vals = kymo_img.flatten()
        vals = vals[~np.isnan(vals)]
        self.ax.hist(vals, bins=256, color='gray', alpha=0.7)
        q1, q2 = np.quantile(vals, low_q), np.quantile(vals, high_q)
        self.ax.axvline(q1, color='b', linestyle='--', label=f'Low q ({low_q:.2f})')
        self.ax.axvline(q2, color='r', linestyle='--', label=f'High q ({high_q:.2f})')
        self.ax.axvline(abs_thresh, color='g', linestyle=':', label=f'Thresh ({abs_thresh:.2f})')
        self.ax.set_title('Quantile Preview')
        self.ax.set_xlabel('Raw Intensity')
        self.ax.set_ylabel('Pixel Count')
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw_idle()

# ------------------ Epithelium Outline Widget (Widget 2) ------------------
@magicgui(
    low_quantile={"label": "Lower quantile", "min": 0.0, "max": 0.5, "step": 0.01},
    high_quantile={"label": "Upper quantile", "min": 0.05, "max": 1.0, "step": 0.01},
    abs_thresh={"label": "Absolute threshold", "min": 0.0, "max": 1.0, "step": 0.001},
    call_button="Detect Outline"
)
def outline_widget(
    low_quantile: float = 0.01,
    high_quantile: float = 0.25,
    abs_thresh: float = 0.13
):
    kymo_img = _last_kymo_img.get("img", None)
    if kymo_img is None:
        print("❌ You must run 'Apply and Save' in the leakage map widget first.")
        return
    if not (0 <= low_quantile < high_quantile <= 1.0):
        print("❌ Lower quantile must be less than upper quantile (range 0–1).")
        return
    try:
        q1 = np.quantile(kymo_img.flatten(), low_quantile)
        q2 = np.quantile(kymo_img.flatten(), high_quantile)
    except Exception as e:
        print(f"❌ Error computing quantiles: {e}")
        return

    stretch = exposure.rescale_intensity(kymo_img, in_range=(q1, q2), out_range=(0, 1))
    mask = stretch > abs_thresh
    if np.sum(mask) == 0:
        print("❌ No pixels above threshold. Try changing quantiles/threshold.")
        return
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if len(contours) == 0:
        print("❌ No outline found. Try adjusting quantiles/threshold.")
        return
    main_contour = max(contours, key=lambda c: c.shape[0])
    if not np.allclose(main_contour[0], main_contour[-1]):
        main_contour = np.vstack([main_contour, main_contour[0]])
    poly = main_contour[:, ::-1]  # (x, y)

    if "Epithelium Outline" in viewer.layers:
        viewer.layers.remove("Epithelium Outline")
    viewer.add_shapes(
        [poly], name="Epithelium Outline",
        shape_type="polygon",
        edge_color="yellow",
        edge_width=2,
        opacity=0.9,
        face_color=[0, 0, 0, 0]
    )
    print(f"✅ Outline with {poly.shape[0]} points. Low q={low_quantile:.2f} High q={high_quantile:.2f} Thresh={abs_thresh:.3f}")
    basepath = os.path.splitext(tif_path)[0]

    # Save outline on kymograph
    svg_path = basepath + "_epithelium_outline_dim.svg"
    plt.figure(figsize=(8, 8))
    plt.imshow(stretch, cmap='gray')
    plt.plot(poly[:, 0], poly[:, 1], 'y-', linewidth=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"📄 Dim intensity outline SVG saved: {svg_path}")

    # Overlay outline on leakage map PNG
    leakage_map_png_path = basepath + "_leakage_map.png"
    overlay_path = basepath + "_leakage_map_with_outline.png"
    try:
        import matplotlib.image as mpimg
        leakage_map_img = mpimg.imread(leakage_map_png_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(leakage_map_img)
        plt.plot(poly[:, 0], poly[:, 1], 'y-', linewidth=2)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"📄 Leakage map with outline PNG saved: {overlay_path}")
    except Exception as e:
        print(f"❌ Could not overlay outline on leakage map PNG: {e}")

# --- Connect histogram preview to outline widget ---
kymo_hist_widget = QuantileHistogramWidget(np.zeros((1, 1)))  # placeholder

def update_hist_widget():
    kymo_img = _last_kymo_img.get("img", None)
    if kymo_img is None or kymo_img.shape[0] < 2:
        return
    kymo_hist_widget.update_histogram(
        kymo_img,
        outline_widget.low_quantile.value,
        outline_widget.high_quantile.value,
        outline_widget.abs_thresh.value
    )

# Live updates for the histogram
outline_widget.low_quantile.changed.connect(update_hist_widget)
outline_widget.high_quantile.changed.connect(update_hist_widget)
outline_widget.abs_thresh.changed.connect(update_hist_widget)

def update_hist_after_widget1():
    kymo_img = _last_kymo_img.get("img", None)
    if kymo_img is not None and kymo_img.shape[0] > 2:
        kymo_hist_widget.update_histogram(
            kymo_img,
            outline_widget.low_quantile.value,
            outline_widget.high_quantile.value,
            outline_widget.abs_thresh.value
        )
leakage_widget.called.connect(update_hist_after_widget1)

# ---------------------- Show/Hide docks on demand ----------------------
widgets_added = {"leak": False, "outline": False, "hist": False}

def add_docks_if_needed():
    if not widgets_added["leak"]:
        viewer.window.add_dock_widget(leakage_widget, area="right")
        widgets_added["leak"] = True
    if not widgets_added["outline"]:
        viewer.window.add_dock_widget(outline_widget, area="right")
        widgets_added["outline"] = True
    if not widgets_added["hist"]:
        viewer.window.add_dock_widget(kymo_hist_widget, area="right")
        widgets_added["hist"] = True

# ---------------------- Auto-quantify, then prompt ----------------------
def initial_run_and_prompt():
    # Run once with defaults (saves figures/CSV and adds layer)
    print("🚀 Auto-quantifying with default parameters...")
    leakage_widget()   # call the function body
    update_hist_after_widget1()

    # Ask user if it looks OK
    msg = QMessageBox(viewer.window._qt_window)
    msg.setWindowTitle("Segmentation Check")
    msg.setText("Did the segmentation / leakage map look OK?")
    msg.setInformativeText("Choose 'No' to open tuning widgets and re-run. 'Yes' will keep current results.")
    msg.setIcon(QMessageBox.Question)
    yes_btn = msg.addButton("Yes, looks good", QMessageBox.YesRole)
    no_btn = msg.addButton("No, let me adjust", QMessageBox.NoRole)
    msg.exec_()

    if msg.clickedButton() is no_btn:
        print("🛠 Opening tuning widgets. Adjust settings and click 'Apply and Save' to re-export.")
        add_docks_if_needed()
    else:
        print("✅ Keeping results as-is. (Widgets remain hidden; viewer stays open.)")

# Schedule after the viewer is up so the dialog is visible
QTimer.singleShot(0, initial_run_and_prompt)

print("✅ Napari will auto-quantify, then ask for confirmation.\n"
      "If you choose NO, tuning widgets will appear; click 'Apply and Save' to re-export.")
napari.run()
