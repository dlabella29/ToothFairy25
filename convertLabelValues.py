#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib

def remap_labels(arr: np.ndarray) -> np.ndarray:
    """
    1) Apply the original ToothFairy3→Grand-Challenge mapping:
       103→51, 104→52, 105→53, 111–148→50
    2) Then apply the post-processing remaps:
       53 → 19
       52 → 20
       51 → 29
       50 → 30
       48 → 39
       47 → 40
    """
    out = arr.copy()

    # — original GC mapping — pulp classes 111–148 → 50
    pulp_mask = (arr >= 111) & (arr <= 148)
    out[pulp_mask] = 50

    # single-value → challenge IDs
    out[arr == 103] = 51
    out[arr == 104] = 52
    out[arr == 105] = 53

    # — new post-processing remaps — 
    out[out == 53] = 19
    out[out == 52] = 20
    out[out == 51] = 29
    out[out == 50] = 30
    out[out == 48] = 39
    out[out == 47] = 40

    return out

def main():
    src_dir = "/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/labelsTr"
    dst_dir = "/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/labelsTrChallengeFixed0-46"
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".nii.gz"):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        # load original labels
        img  = nib.load(src_path)
        data = img.get_fdata().astype(np.int16)

        # remap labels
        new_data = remap_labels(data)

        # save with original affine & header
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        nib.save(new_img, dst_path)

        print(f"Converted and saved: {fname}")

if __name__ == "__main__":
    main()
