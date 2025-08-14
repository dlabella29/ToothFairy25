#!/usr/bin/env python3
"""
Count label-value volumes (mm^3) across a folder of NIfTI label maps.

Outputs:
  - counts_per_file.csv   (rows = files, columns = label values, units mm³)
  - aggregate_counts.csv  (totals across all files and file-presence stats, units mm³)
  - Console summary incl. labels present in ALL files (set intersection)

Requirements:
  pip install nibabel numpy pandas tqdm
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import warnings

# ============================== USER CONFIG ==============================
#LABELS_DIR = Path(r"/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped3/labels_cropped/remapped/")
LABELS_DIR = Path(r"/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped2/labels_cropped/remapped/")
OUTPUT_DIR = LABELS_DIR  # change if you want CSVs elsewhere
INCLUDE_BACKGROUND = False  # set True if you also want label 0 counts
SKIP_LABELS = set()         # e.g., {99} to ignore certain labels
# ==========================================================================

def load_labels_and_voxel_volume(path: Path):
    """Load label map as integer array and return voxel volume in mm³."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if not np.issubdtype(arr.dtype, np.integer):
        warnings.warn(f"{path.name}: non-integer dtype {arr.dtype}; rounding to nearest int.")
        arr = np.rint(arr).astype(np.int64, copy=False)
    else:
        arr = arr.astype(np.int64, copy=False)

    # Get voxel volume from pixdim (affine-based fallback if missing)
    hdr = img.header
    zooms = hdr.get_zooms()[:3]  # x, y, z spacing
    voxel_vol_mm3 = float(np.prod(zooms))
    return arr, voxel_vol_mm3

def main():
    files = sorted(LABELS_DIR.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz files found in: {LABELS_DIR}")

    per_file_rows = []
    union_labels = set()
    per_file_label_sets = []

    for fp in tqdm(files, desc="Scanning label maps"):
        arr, voxel_vol_mm3 = load_labels_and_voxel_volume(fp)
        labels, counts = np.unique(arr, return_counts=True)

        # Optionally drop background and any skip labels
        mask = np.ones_like(labels, dtype=bool)
        if not INCLUDE_BACKGROUND:
            mask &= (labels != 0)
        if SKIP_LABELS:
            mask &= ~np.isin(labels, list(SKIP_LABELS))
        labels = labels[mask]
        counts = counts[mask]

        # Convert voxel counts to mm³
        volumes_mm3 = counts * voxel_vol_mm3

        # Record per-file dict
        row = {"file": fp.name, "total_volume_mm3": float(arr.size * voxel_vol_mm3)}
        nonzero_vol = float((arr != 0).sum() * voxel_vol_mm3) if not INCLUDE_BACKGROUND else row["total_volume_mm3"]
        row["nonzero_volume_mm3"] = nonzero_vol
        for lab, vol in zip(labels, volumes_mm3):
            row[int(lab)] = float(vol)

        per_file_rows.append(row)
        union_labels.update(map(int, labels))
        per_file_label_sets.append(set(map(int, labels)))

    # Build per-file DataFrame
    sorted_labels = sorted(union_labels)
    base_cols = ["file", "total_volume_mm3", "nonzero_volume_mm3"]
    df = pd.DataFrame(per_file_rows).fillna(0.0)
    for lab in sorted_labels:
        if lab not in df.columns:
            df[lab] = 0.0
    df = df[base_cols + sorted_labels]

    # Save per-file table
    per_file_csv = OUTPUT_DIR / "counts_per_file.csv"
    df.to_csv(per_file_csv, index=False)

    # Aggregate totals
    label_totals = {lab: float(df[lab].sum()) for lab in sorted_labels}
    files_with_label = {lab: int((df[lab] > 0).sum()) for lab in sorted_labels}

    agg = pd.DataFrame({
        "label": sorted_labels,
        "total_volume_mm3_across_all_files": [label_totals[lab] for lab in sorted_labels],
        "files_with_label": [files_with_label[lab] for lab in sorted_labels],
        "fraction_of_files_with_label": [files_with_label[lab] / len(files) for lab in sorted_labels],
        "mean_volume_mm3_per_file_when_present": [
            (label_totals[lab] / files_with_label[lab]) if files_with_label[lab] > 0 else 0.0
            for lab in sorted_labels
        ],
    })

    # Summary row
    summary_row = pd.DataFrame([{
        "label": "_SUMMARY_",
        "total_volume_mm3_across_all_files": float(df["total_volume_mm3"].sum()),
        "files_with_label": len(files),
        "fraction_of_files_with_label": 1.0,
        "mean_volume_mm3_per_file_when_present": float(df["nonzero_volume_mm3"].mean()) if not df.empty else 0.0,
    }])
    agg = pd.concat([agg, summary_row], ignore_index=True)

    agg_csv = OUTPUT_DIR / "aggregate_counts.csv"
    agg.to_csv(agg_csv, index=False)

    # Labels present in ALL files
    if per_file_label_sets:
        labels_in_all_files = set.intersection(*per_file_label_sets) if len(per_file_label_sets) > 1 else per_file_label_sets[0]
    else:
        labels_in_all_files = set()

    # Console summary
    print("\n=== Summary ===")
    print(f"Files scanned: {len(files)}")
    print(f"Per-file table: {per_file_csv}")
    print(f"Aggregate table: {agg_csv}")
    print(f"Labels observed across dataset: {sorted_labels}")
    print(f"Labels present in ALL files: {sorted(labels_in_all_files) if labels_in_all_files else 'None'}")

if __name__ == "__main__":
    main()
