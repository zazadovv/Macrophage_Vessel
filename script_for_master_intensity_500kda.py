import os
import pandas as pd
from glob import glob
from tkinter import Tk, filedialog
import re

def parse_filename(fname):
    # Example: 20250514_alox5a_WT_2dpf_500kDa_NW_E1_F4_delta_area.csv
    # Adjust the regex if your filenames change!
    m = re.match(
        r'(\d+)_([a-zA-Z0-9]+)_([A-Za-z0-9]+)_([0-9dp]+)_(\d+kDa)_([A-Za-z0-9]+)_E(\d+)_F(\d+)_delta_area\.csv',
        fname)
    if m:
        return {
            'Date': m.group(1),
            'Genotype': m.group(2),
            'Condition': m.group(3),
            'DevStage': m.group(4),
            'MW': m.group(5),
            'Treatment': m.group(6),
            'ExpNum': int(m.group(7)),
            'FishNum': int(m.group(8))
        }
    else:
        return {
            'Date': '',
            'Genotype': '',
            'Condition': '',
            'DevStage': '',
            'MW': '',
            'Treatment': '',
            'ExpNum': -1,
            'FishNum': -1
        }

# --- GUI folder selection ---
root = Tk()
root.withdraw()
data_dir = filedialog.askdirectory(title="Select folder with _delta_area.csv files")
if not data_dir:
    raise SystemExit("No directory selected.")

csv_files = sorted(glob(os.path.join(data_dir, "*_delta_area.csv")))
if not csv_files:
    raise SystemExit("No delta_area CSV files found.")

dfs = []
for csv_file in csv_files:
    fname = os.path.basename(csv_file)
    meta = parse_filename(fname)
    df = pd.read_csv(csv_file)
    for key in ['Date', 'Genotype', 'Condition', 'DevStage', 'MW', 'Treatment', 'ExpNum', 'FishNum']:
        df[key] = meta[key]
    df.insert(0, "FileName", fname)
    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)
master_df = master_df.sort_values(['Genotype', 'Date', 'ExpNum', 'FishNum'])

# --- Save as CSV (no Excel, no openpyxl needed) ---
out_path = os.path.join(data_dir, "master_delta_area_analysis_sorted.csv")
master_df.to_csv(out_path, index=False)

print(f"✅ Master sorted CSV written to: {out_path}")
