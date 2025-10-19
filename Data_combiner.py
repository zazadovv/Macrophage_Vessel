#!/usr/bin/env python3
"""
stack_area_ratio_tables.py
------------------------------------------------
Pick a folder full of *.csv files whose names contain 'DMSO' or 'Licofelone'.

For every file:
  • Load it intact (no thinning, no skipping)
  • Append two new columns:
        - SourceFile   (original filename)
        - Condition    (Licofelone, DMSO, or Other)
  • Vertically concatenate all rows into one master DataFrame

Finally save:
  • master_area_ratios.csv  (all rows from all files, stacked)
"""

import os
from glob import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# ---------- 1. Folder picker ----------
root = tk.Tk(); root.withdraw()
folder = filedialog.askdirectory(title="Select folder with .csv files")
if not folder:
    raise SystemExit("❌ No folder selected.")
print(f"\n📂 Reading from: {folder}")

# ---------- 2. Find CSVs ----------
csv_paths = sorted(glob(os.path.join(folder, "*.csv")))
if not csv_paths:
    raise SystemExit("❌ No .csv files found.")

# ---------- 3. Helper: infer condition ----------
def infer_condition(fname: str) -> str:
    lower = fname.lower()
    if "licofelone" in lower:
        return "Licofelone"
    if "dmso" in lower:
        return "DMSO"
    return "Other"

# ---------- 4. Load & stack ----------
stacked_rows = []
for path in csv_paths:
    fname = os.path.basename(path)
    cond  = infer_condition(fname)

    df = pd.read_csv(path)
    df["SourceFile"] = fname       # keep provenance
    df["Condition"]  = cond        # explicit group label

    stacked_rows.append(df)

master_df = pd.concat(stacked_rows, axis=0, ignore_index=True)
print(f"✅ Final shape: {master_df.shape}  (rows, columns)")

# ---------- 5. Save ----------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(folder, f"stacked_output_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

out_csv = os.path.join(out_dir, "master_area_ratios.csv")
master_df.to_csv(out_csv, index=False)

print(f"\n📄 Master CSV saved ➜ {out_csv}\n🎉 Done.")
