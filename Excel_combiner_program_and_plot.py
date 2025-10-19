import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from glob import glob

# --- Select directory ---
root = Tk(); root.withdraw()
data_dir = filedialog.askdirectory(title="Select folder with *_ghost_metrics.csv and *_ghost_angles.csv")
if not data_dir:
    raise SystemExit("No directory selected.")

# --- Load file lists ---
metrics_files = sorted(glob(os.path.join(data_dir, "*_ghost_metrics.csv")))
angle_files   = sorted(glob(os.path.join(data_dir, "*_ghost_angles.csv")))

if len(metrics_files) == 0:
    raise FileNotFoundError("No *_ghost_metrics.csv files found.")
if len(angle_files) == 0:
    print("ℹ️  No *_ghost_angles.csv found. Rose plot will be skipped.")

# ========== Helpers ==========
REQUIRED_COLS = {
    "frame", "time_sec",
    "velocity", "velocity_sem",
    "displacement", "displacement_sem"
}

def _read_metrics_with_sem(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {os.path.basename(path)}: {sorted(missing)}")
    out = df[list(REQUIRED_COLS)].copy()
    for c in REQUIRED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["source_file"] = os.path.basename(path).replace("_ghost_metrics.csv", "")
    return out.dropna(subset=["frame", "time_sec"])

def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return np.nanstd(x, ddof=1) / np.sqrt(x.size)

# ========== Load all metrics (WITH per-file SEM) ==========
metrics_per_file = []
for path in metrics_files:
    try:
        m = _read_metrics_with_sem(path)
        metrics_per_file.append(m)
    except Exception as e:
        print(f"⚠️  Skipping {os.path.basename(path)}: {e}")

if len(metrics_per_file) == 0:
    raise SystemExit("No valid metrics files after cleaning.")

all_metrics = pd.concat(metrics_per_file, ignore_index=True)

# ========== Stage 1: per-file timecourses ==========
# Keep each file’s own SEM (already computed in CSV)
per_file = (
    all_metrics
    .groupby(["source_file", "frame", "time_sec"], as_index=False)
    .agg(
        velocity_mean=("velocity", "mean"),
        velocity_sem =("velocity_sem", "mean"),          # use imported per-file SEM
        displacement_mean=("displacement", "mean"),
        displacement_sem =("displacement_sem", "mean")   # use imported per-file SEM
    )
)

# ========== Stage 2: across-file summary ==========
# Mean across files; SEM across files computed from per-file means
summary = (
    per_file
    .groupby(["frame", "time_sec"], as_index=False)
    .agg(
        velocity_mean=("velocity_mean", "mean"),
        velocity_sem =("velocity_mean", _sem),              # across-file SEM
        displacement_mean=("displacement_mean", "mean"),
        displacement_sem =("displacement_mean", _sem)       # across-file SEM
    )
)

# --- Clean NaNs ---
for col in ["time_sec", "velocity_mean", "velocity_sem", "displacement_mean", "displacement_sem"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce")
summary = summary.dropna(subset=["time_sec", "velocity_mean", "displacement_mean"])

# ========== Plots ==========
time_sec = summary["time_sec"].to_numpy()
v_mean   = summary["velocity_mean"].to_numpy()
v_sem    = summary["velocity_sem"].to_numpy()
d_mean   = summary["displacement_mean"].to_numpy()
d_sem    = summary["displacement_sem"].to_numpy()

# Single combined plots by default (no overlays)
DRAW_INDIVIDUAL_RIBBONS = False  # set True if you want per-file ribbons shown lightly

# Velocity
plt.figure()
if DRAW_INDIVIDUAL_RIBBONS:
    for sf, df in per_file.groupby("source_file"):
        t = df["time_sec"].to_numpy()
        m = df["velocity_mean"].to_numpy()
        s = df["velocity_sem"].to_numpy()
        if len(t) and np.isfinite(m).any():
            plt.plot(t, m, alpha=0.35, linewidth=1)
            if np.isfinite(s).any():
                plt.fill_between(t, m - 1.96*s, m + 1.96*s, alpha=0.10)

plt.plot(time_sec, v_mean, label="Velocity (mean across files)", color="orange", linewidth=2.5)
if np.isfinite(v_sem).any():
    plt.fill_between(time_sec, v_mean - 1.96*v_sem, v_mean + 1.96*v_sem,
                     alpha=0.3, color="orange", label="95% CI (across files)")
plt.xlabel("Time (sec)", fontsize=16)
plt.ylabel("Velocity (pixels/sec)", fontsize=16)
plt.title("Ghost Velocity (95% CI)", fontsize=18)
plt.legend()
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "combined_ghost_velocity.png"), dpi=300)
plt.savefig(os.path.join(data_dir, "combined_ghost_velocity.svg"), format="svg", bbox_inches="tight")
plt.close()

# Displacement
plt.figure()
if DRAW_INDIVIDUAL_RIBBONS:
    for sf, df in per_file.groupby("source_file"):
        t = df["time_sec"].to_numpy()
        m = df["displacement_mean"].to_numpy()
        s = df["displacement_sem"].to_numpy()
        if len(t) and np.isfinite(m).any():
            plt.plot(t, m, alpha=0.35, linewidth=1)
            if np.isfinite(s).any():
                plt.fill_between(t, m - 1.96*s, m + 1.96*s, alpha=0.10)

plt.plot(time_sec, d_mean, label="Displacement (mean across files)", color="orange", linewidth=2.5)
if np.isfinite(d_sem).any():
    plt.fill_between(time_sec, d_mean - 1.96*d_sem, d_mean + 1.96*d_sem,
                     alpha=0.3, color="orange", label="95% CI (across files)")
plt.xlabel("Time (sec)", fontsize=16)
plt.ylabel("Net Displacement (pixels)", fontsize=16)
plt.title("Ghost Displacement (95% CI)", fontsize=18)
plt.legend()
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "combined_ghost_displacement.png"), dpi=300)
plt.savefig(os.path.join(data_dir, "combined_ghost_displacement.svg"), format="svg", bbox_inches="tight")
plt.close()

# ========== Rose plot (across files) ==========
if len(angle_files) > 0:
    n_bins = 32
    angle_bins   = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers  = (angle_bins[:-1] + angle_bins[1:]) / 2
    replicate_counts = []

    for path in angle_files:
        try:
            df = pd.read_csv(path)
            if "angle_rad" not in df.columns:
                print(f"⚠️  Skipping angles in {os.path.basename(path)}: no 'angle_rad' column.")
                continue
            angles = pd.to_numeric(df["angle_rad"], errors="coerce").dropna().to_numpy()
            if angles.size == 0:
                continue
            counts, _ = np.histogram(angles, bins=angle_bins, density=True)
            replicate_counts.append(counts)
        except Exception as e:
            print(f"⚠️  Skipping angles in {os.path.basename(path)}: {e}")

    if len(replicate_counts) > 0:
        replicate_counts = np.asarray(replicate_counts, dtype=float)
        mean_vals = np.nanmean(replicate_counts, axis=0)
        sem_vals  = np.nanstd(replicate_counts, ddof=1, axis=0) / np.sqrt(replicate_counts.shape[0])

        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_subplot(111, polar=True)
        ax.bar(bin_centers, mean_vals, width=2*np.pi/n_bins, alpha=0.6, align="center", color="orange")

        # 95% CI error "spokes"
        for ang, m, s in zip(bin_centers, mean_vals, sem_vals):
            if np.isfinite(s):
                lo = max(0, m - 1.96*s)
                hi = m + 1.96*s
                ax.plot([ang, ang], [lo, hi], color="black", linewidth=1.2)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title("Combined Ghost Rose Plot with 95% CI (across files)", fontsize=18)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "combined_ghost_roseplot.png"), dpi=300)
        plt.savefig(os.path.join(data_dir, "combined_ghost_roseplot.svg"), format="svg", bbox_inches="tight")
        plt.close()
    else:
        print("ℹ️  No usable angle data to plot.")

# ========== Outputs ==========
per_file_out  = os.path.join(data_dir, "per_file_timecourse_means_with_imported_SEM.csv")
summary_out   = os.path.join(data_dir, "across_files_summary_with_sem.csv")
per_file.sort_values(["source_file", "frame"]).to_csv(per_file_out, index=False)
summary.sort_values("frame").to_csv(summary_out, index=False)

# Also write a raw dump and raw angles (optional)
angle_combined = []
for path in angle_files:
    try:
        df = pd.read_csv(path)
        if "angle_rad" in df.columns:
            tmp = df[["angle_rad"]].copy()
            tmp["source_file"] = os.path.basename(path).replace("_ghost_angles.csv", "")
            tmp["angle_index"] = tmp.index
            angle_combined.append(tmp)
    except Exception:
        pass

angle_df = pd.concat(angle_combined, ignore_index=True) if angle_combined else pd.DataFrame()

csv_path = os.path.join(data_dir, "combined_ghost_master_metrics_and_angles.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write("# --- Raw Ghost Metrics (all files, with imported SEMs) ---\n")
    all_metrics.to_csv(f, index=False)
    f.write("\n\n# --- Raw Ghost Angles (all files concatenated) ---\n")
    if not angle_df.empty:
        angle_df.to_csv(f, index=False)
    else:
        f.write("No angle data found.\n")

print(f"✅ Plots saved to: {data_dir}")
print(f"📄 Per-file (with imported SEM): {per_file_out}")
print(f"📄 Across-file summary (SEM across replicates): {summary_out}")
print(f"📄 Combined raw dump: {csv_path}")
