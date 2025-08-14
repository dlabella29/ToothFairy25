#!/usr/bin/env python3
"""
Batch resample & window CT images (NIfTI .nii).

- Input:  /media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/imagesTrF
- Output: /media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/imagesResampledWindowed
- Resample spacing: [0.6, 0.6, 0.6] mm
- Intensity window (clip): [-1000, 3880] HU

Interpolation: Linear (images)
Output dtype: int16 (change OUTPUT_DTYPE if needed)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ============================== USER CONFIG ==============================
IN_DIR  = Path(r"/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/labelsTrF")
OUT_DIR = Path(r"/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/labelsResampledWindowed")
TARGET_SPACING: Tuple[float, float, float] = (0.6, 0.6, 0.6)  # mm
CLIP_LOW, CLIP_HIGH = -1000.0, 3880.0
OUTPUT_DTYPE = np.int16        # or np.float32
OVERWRITE = True
# ========================================================================

def compute_new_size(img: sitk.Image, out_spacing: Tuple[float, float, float]) -> Tuple[int, int, int]:
    in_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)
    in_size    = np.array(list(img.GetSize()),    dtype=np.int64)
    out_spacing = np.array(out_spacing, dtype=np.float64)
    scale = in_spacing / out_spacing
    out_size = np.maximum(np.round(in_size * scale).astype(np.int64), 1)
    return int(out_size[0]), int(out_size[1]), int(out_size[2])

def resample_image(img: sitk.Image, out_spacing: Tuple[float, float, float]) -> sitk.Image:
    out_size = compute_new_size(img, out_spacing)
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(out_spacing)
    rf.SetSize(out_size)
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0)
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    return rf.Execute(img_f)

def window_and_cast(img: sitk.Image, low: float, high: float, out_dtype=np.int16) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img)
    arr = np.clip(arr, low, high)
    if out_dtype is np.int16:
        arr = arr.astype(np.int16, copy=False)
        out = sitk.GetImageFromArray(arr, isVector=False)
        out.CopyInformation(img)
        return sitk.Cast(out, sitk.sitkInt16)
    elif out_dtype is np.float32:
        arr = arr.astype(np.float32, copy=False)
        out = sitk.GetImageFromArray(arr, isVector=False)
        out.CopyInformation(img)
        return sitk.Cast(out, sitk.sitkFloat32)
    else:
        raise ValueError("Unsupported OUTPUT_DTYPE; use np.int16 or np.float32")

def main():
    if not IN_DIR.is_dir():
        raise SystemExit(f"Input folder not found: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz files found in {IN_DIR}")

    print(f"Found {len(files)} files. Writing .nii to: {OUT_DIR}")
    for fp in tqdm(files, desc="Processing", unit="img"):
        out_name = fp.name.replace(".nii.gz", ".nii")  # ensure .nii
        out_path = OUT_DIR / out_name
        if out_path.exists() and not OVERWRITE:
            continue

        img = sitk.ReadImage(str(fp))
        res = resample_image(img, TARGET_SPACING)
        out = window_and_cast(res, CLIP_LOW, CLIP_HIGH, OUTPUT_DTYPE)

        # Save as uncompressed NIfTI (.nii)
        sitk.WriteImage(out, str(out_path), useCompression=False)

    print("Done.")

if __name__ == "__main__":
    main()
