#!/usr/bin/env python3
# crop_toothfairy_mandible_boxes.py
# Requires: SimpleITK, numpy
#   pip install SimpleITK numpy

import os
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk

# ============================== USER CONFIG ==============================
IMAGES_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped/images_cropped/")
LABELS_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped/labels_cropped/")

OUT_IMAGES_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped3/images_cropped/")
OUT_LABELS_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/cropped3/labels_cropped/")

MANDIBLE_LABEL = 1

# X/Y/Z extents to add around the derived anchors (in voxels)
X_HALF_WIDTH = 20            # xmin = xmid - X_HALF_WIDTH, xmax = xmid + X_HALF_WIDTH
Y_EXTENT     = 100             # yminNew = yminMandible, ymaxNew = yminMandible + Y_EXTENT
Z_BELOW_MAX  = 90             # zminNew = zmaxMandible - Z_BELOW_MAX, zmaxNew = zmaxMandible
# ========================================================================


def ensure_dirs():
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def list_label_files():
    # labels are "*.nii.gz" (no _0000)
    return sorted([p for p in LABELS_DIR.glob("*.nii.gz") if p.is_file()])


def load_nifti(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"{path.name}: expected 3D NIfTI, got dim={img.GetDimension()}")
    return img


def nii_to_numpy(img: sitk.Image) -> np.ndarray:
    # SimpleITK GetArrayFromImage returns array in [z, y, x] index order.
    return sitk.GetArrayFromImage(img)


def compute_crop_from_label(lbl_img: sitk.Image) -> tuple:
    """
    Returns (start_xyz, size_xyz) in SITK index space (x,y,z).
      yminNew = min y of mandible voxels
      ymaxNew = yminNew + Y_EXTENT
      xmiddle = mean x of mandible voxels on the slice where y == yminNew
      xminNew = xmiddle - X_HALF_WIDTH
      xmaxNew = xmiddle + X_HALF_WIDTH
      zminNew = zmaxMandible - Z_BELOW_MAX
      zmaxNew = zmaxMandible
    All clamped to the image bounds.
    """
    arr = nii_to_numpy(lbl_img)  # [z, y, x]
    sz = np.array(list(reversed(lbl_img.GetSize())))  # [z, y, x]
    mand = (arr == MANDIBLE_LABEL)
    if not mand.any():
        raise ValueError("Mandible label (1) not found in this label volume.")

    zyxs = np.argwhere(mand)  # [n, 3] -> z,y,x
    y_min_mand = int(zyxs[:, 1].min())
    z_max_mand = int(zyxs[:, 0].max())

    on_ymin = zyxs[zyxs[:, 1] == y_min_mand]
    if on_ymin.size == 0:
        raise RuntimeError("No mandible voxels found on the ymin slice (unexpected).")

    xmiddle = float(on_ymin[:, 2].mean())

    xmin = int(round(xmiddle - X_HALF_WIDTH))
    xmax = int(round(xmiddle + X_HALF_WIDTH))
    ymin = y_min_mand
    ymax = y_min_mand + Y_EXTENT
    zmin = z_max_mand - Z_BELOW_MAX
    zmax = z_max_mand

    # Clamp to image bounds
    def clamp(a, lo, hi): return max(lo, min(int(a), hi))

    x_size = sz[2]
    y_size = sz[1]
    z_size = sz[0]

    xmin = clamp(xmin, 0, x_size - 1)
    xmax = clamp(xmax, 0, x_size - 1)
    ymin = clamp(ymin, 0, y_size - 1)
    ymax = clamp(ymax, 0, y_size - 1)
    zmin = clamp(zmin, 0, z_size - 1)
    zmax = clamp(zmax, 0, z_size - 1)

    if xmax < xmin: xmin, xmax = xmax, xmin
    if ymax < ymin: ymin, ymax = ymax, ymin
    if zmax < zmin: zmin, zmax = zmax, zmin

    # Convert [z,y,x] -> (x,y,z) for SITK
    start_xyz = (xmin, ymin, zmin)
    size_xyz  = (xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1)
    return start_xyz, size_xyz


def extract_region(img: sitk.Image, start_xyz, size_xyz) -> sitk.Image:
    # Preserves spacing/direction and updates origin correctly
    return sitk.RegionOfInterest(img, size=size_xyz, index=start_xyz)


def find_image_for_label(lbl_path: Path) -> Path | None:
    """
    Expected pairing:
      label:  case.nii.gz
      image:  case_0000.nii.gz
    Fallback: try same name if present.
    """
    stem = lbl_path.name[:-7]  # strip ".nii.gz"
    img_candidate = IMAGES_DIR / f"{stem}_0000.nii.gz"
    if img_candidate.exists():
        return img_candidate
    same_name = IMAGES_DIR / lbl_path.name
    if same_name.exists():
        return same_name
    return None


def process_pair(lbl_path: Path):
    img_path = find_image_for_label(lbl_path)
    if img_path is None:
        print(f"[WARN] Missing image for label: {lbl_path.name} — expected "
              f"'{lbl_path.stem}_0000.nii.gz' or '{lbl_path.name}'. Skipping.")
        return False

    # Load
    lbl_img = load_nifti(lbl_path)
    img_img = load_nifti(img_path)

    # Sanity: sizes must match
    if lbl_img.GetSize() != img_img.GetSize():
        raise ValueError(f"Size mismatch for {lbl_path.name}: "
                         f"label {lbl_img.GetSize()} vs image {img_img.GetSize()}")

    # Compute crop
    start_xyz, size_xyz = compute_crop_from_label(lbl_img)

    # Extract from both, preserving world coords
    lbl_crop = extract_region(lbl_img, start_xyz, size_xyz)
    img_crop = extract_region(img_img, start_xyz, size_xyz)

    # Save using original naming convention:
    #   labels -> 'case.nii.gz'
    #   images -> 'case_0000.nii.gz' (or same-name if that was used)
    out_lbl = OUT_LABELS_DIR / lbl_path.name
    out_img = OUT_IMAGES_DIR / img_path.name
    sitk.WriteImage(lbl_crop, str(out_lbl), useCompression=True)
    sitk.WriteImage(img_crop, str(out_img), useCompression=True)

    print(f"[OK] {lbl_path.name}  ->  start(x,y,z)={start_xyz}, size={size_xyz}  | saved: {out_img.name}, {out_lbl.name}")
    return True


def main():
    ensure_dirs()
    label_files = list_label_files()
    if not label_files:
        print(f"No label files found in: {LABELS_DIR}")
        sys.exit(1)

    processed = 0
    skipped = 0
    errors = 0

    for lbl in label_files:
        try:
            ok = process_pair(lbl)
            if ok:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] {lbl.name}: {e}")

    print("\nSummary:")
    print(f"  Processed pairs: {processed}")
    print(f"  Skipped (no matching image): {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
