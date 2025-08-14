#!/usr/bin/env python
# inference_ToothFairy_ensemble.py (Aug-2025 – iSTAPLE + clean-up, 18-neigh. touching)

import os, sys, json, yaml, shutil, warnings, glob
import SimpleITK as sitk, torch
import numpy as np

# ─────────────── PATHS (edit if your tree changes) ──────────────────────────
BASE     = "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/ToothFairy"
INPUT    = os.path.join(BASE, "input",  "images", "cbct")
BUNDLES  = os.path.join(BASE, "AutoToothWorkDirChallenge")            # segresnet_0 … 5
TMP      = os.path.join(BASE, "tmp")
CONV     = os.path.join(TMP,  "converted")
OUTIMG   = os.path.join(BASE, "output", "images", "oral-pharyngeal-segmentation")
num_folds = 5

shutil.rmtree(TMP, ignore_errors=True)  # clean tmp/ completely
for d in (TMP, CONV, OUTIMG): os.makedirs(d, exist_ok=True)

# ─────────────── HOUSE-KEEPING ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning,
                        message="A NumPy version .* is required for this")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻  inference device: {device}\n")

# Clean old prediction_testing folders so that only fresh masks exist
for pred_dir in glob.glob(os.path.join(BUNDLES, "segresnet_*", "prediction_testing")):
    if os.path.isdir(pred_dir):
        print(f"🧹 removing stale predictions in {pred_dir}")
        shutil.rmtree(pred_dir)           # delete dir and its contents
        os.makedirs(pred_dir, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from segmenter import run_segmenter                           # MONAI bundle API

# ─────────────── 1. CONVERT FIRST .mha ──────────────────────────────────────
mha_files = sorted(f for f in os.listdir(INPUT) if f.lower().endswith(".mha"))
if not mha_files:
    raise FileNotFoundError(f"No .mha files found in {INPUT}")
mha_fp = os.path.join(INPUT, mha_files[0])
nii_fp = os.path.join(CONV,  mha_files[0].replace(".mha", ".nii.gz"))
print(f"➊  reading   : {mha_fp}")
sitk.WriteImage(sitk.ReadImage(mha_fp), nii_fp, useCompression=True)
print(f"    written  : {nii_fp}\n")

# ─────────────── 2. BUILD TEST LIST JSON ────────────────────────────────────
json_fp = os.path.join(TMP, "ToothFairy_test_data.json")
json.dump({"testing": [{"image": [os.path.basename(nii_fp)]}]},
          open(json_fp, "w"), indent=2)
print(f"➋  JSON      : {json_fp}\n")

# ─────────────── 3. FOLD CONFIG PATHS ───────────────────────────────────────
CFG = [os.path.join(BUNDLES, f"segresnet_{i}", "configs", "hyper_parameters.yaml")
       for i in range(num_folds)]

def infer_fold(i: int) -> str:
    """Run bundle *i*, copy its mask into tmp/, return that path."""
    cfg = CFG[i]
    assert os.path.isfile(cfg), f"Config missing: {cfg}"
    bundle_root = os.path.dirname(os.path.dirname(cfg))

    override = {
        "bundle_root":         bundle_root,
        "data_file_base_dir":  CONV,
        "data_list_file_path": json_fp,
        "infer#enabled":       True,
    }

    print(f"▶ fold {i}: {bundle_root}")
    run_segmenter(config_file=cfg, **override)

    legacy_dir = os.path.join(bundle_root, "prediction_testing")
    tmp_dir    = os.path.join(TMP, f"fold_{i}", "prediction_testing")
    os.makedirs(tmp_dir, exist_ok=True)

    preds = [f for f in os.listdir(legacy_dir) if f.endswith(".nii.gz")]
    if not preds:
        raise RuntimeError(f"Fold {i}: no .nii.gz in {legacy_dir}")
    src = os.path.join(legacy_dir, preds[0])
    dst = os.path.join(tmp_dir,    preds[0])
    shutil.copy2(src, dst)
    return dst

# ─────────────── 4. RUN ALL FOLDS ───────────────────────────────────────────
pred_paths = [infer_fold(i) for i in range(num_folds)]
seg_images = [sitk.ReadImage(p) for p in pred_paths]

# ─────────────── 5. LABEL FUSION WITH MultiLabel-STAPLE ─────────────────────
staple = sitk.MultiLabelSTAPLEImageFilter()
staple.SetMaximumNumberOfIterations(3)
staple.SetTerminationUpdateThreshold(1e-4)

print("\n🧮  running MultiLabel-STAPLE …")
fused = staple.Execute(seg_images)
fused = sitk.Cast(fused, sitk.sitkUInt8)          # 0–46 classes → uint8
fused.CopyInformation(seg_images[0])

# ─────────────── 5a. CLEAN-UP: convert any component touching large label-7 → 7 ─
# Any component (of any label) that touches a label-7 (pharynx) component with >1000 voxels
# becomes 7. "Touching" tested with an 18-neighborhood (faces+edges; no corners).
THRESH_7_VOXELS = 1000
arr = sitk.GetArrayFromImage(fused)               # z-y-x

print("Starting cleanup of label-7 components (if any) …")
mask7 = (arr == 7)
if mask7.any():
    mask7_img = sitk.Cast(sitk.GetImageFromArray(mask7.astype(np.uint8)), sitk.sitkUInt8)
    mask7_img.CopyInformation(fused)

    # 6- vs 26-connected CC is all SimpleITK exposes; use 6-connected for speed
    cc7f = sitk.ConnectedComponentImageFilter()
    cc7f.SetFullyConnected(False)  # 6-connected
    cc7_img = cc7f.Execute(mask7_img)

    stats7 = sitk.LabelShapeStatisticsImageFilter()
    stats7.Execute(cc7_img)
    big7_labels = [l for l in stats7.GetLabels() if stats7.GetNumberOfPixels(l) > THRESH_7_VOXELS]

    if big7_labels:
        print(f"🧼  large label-7 components (> {THRESH_7_VOXELS} voxels): {len(big7_labels)}")
        cc7_arr = sitk.GetArrayFromImage(cc7_img)
        big7_mask = (np.isin(cc7_arr, big7_labels)).astype(np.uint8)

        big7_img = sitk.Cast(sitk.GetImageFromArray(big7_mask), sitk.sitkUInt8)
        big7_img.CopyInformation(fused)

        # 18-neighborhood dilation = union of three 2D box dilations (XY, XZ, YZ)
        d_xy = sitk.BinaryDilate(big7_img, [1, 1, 0], sitk.sitkBox, 1)
        d_xz = sitk.BinaryDilate(big7_img, [1, 0, 1], sitk.sitkBox, 1)
        d_yz = sitk.BinaryDilate(big7_img, [0, 1, 1], sitk.sitkBox, 1)
        dilated_big7 = sitk.Or(sitk.Or(d_xy, d_xz), d_yz)

        dilated_big7_arr = sitk.GetArrayFromImage(dilated_big7).astype(bool)

        unique_labels = np.unique(arr)
        ccf = sitk.ConnectedComponentImageFilter()
        ccf.SetFullyConnected(False)  # 6-connected CC within each label

        changed_components = 0
        for L in [int(x) for x in unique_labels if x not in (0, 7)]:
            maskL = (arr == L)
            if not maskL.any():
                continue

            maskL_img = sitk.Cast(sitk.GetImageFromArray(maskL.astype(np.uint8)), sitk.sitkUInt8)
            maskL_img.CopyInformation(fused)
            ccL_img = ccf.Execute(maskL_img)
            ccL_arr = sitk.GetArrayFromImage(ccL_img)

            # IDs of label-L components that touch the (18-neigh.) large-7 region
            touch_ids = np.unique(ccL_arr[dilated_big7_arr])
            touch_ids = touch_ids[touch_ids != 0]
            if touch_ids.size == 0:
                continue

            arr[np.isin(ccL_arr, touch_ids)] = 7
            changed_components += touch_ids.size

        if changed_components:
            print(f"🔁  converted {changed_components} touching components to label 7")
        else:
            print("ℹ️  no non-7 components were touching large label-7 regions")

        fused = sitk.GetImageFromArray(arr.astype(np.uint8))
        fused.CopyInformation(seg_images[0])
    else:
        print(f"ℹ️  no label-7 components > {THRESH_7_VOXELS} voxels — cleanup skipped")
else:
    print("ℹ️  no label-7 voxels present — cleanup skipped")

# ─────────────── 5b. RELABEL BEFORE SAVING  ────────────────────────────────
REMAP = {19: 53, 20: 52, 29: 51, 30: 50, 39: 48, 40: 47}
arr = sitk.GetArrayFromImage(fused)
for old, new in REMAP.items():
    arr[arr == old] = new
fused = sitk.GetImageFromArray(arr.astype(np.uint8))
fused.CopyInformation(seg_images[0])

# ─────────────── 5c. REMOVE SMALL MANDIBLE INSTANCES (label = 1) ────────────
# Any standalone mandible component (label==1) with < 20,000 voxels is removed (set to 0).
MIN_MANDIBLE_VOXELS = 200000
MANDIBLE_LABEL = 1

mand_arr = sitk.GetArrayFromImage(fused)  # z-y-x
mand_mask = (mand_arr == MANDIBLE_LABEL)
if mand_mask.any():
    mand_img = sitk.Cast(sitk.GetImageFromArray(mand_mask.astype(np.uint8)), sitk.sitkUInt8)
    mand_img.CopyInformation(fused)

    ccf = sitk.ConnectedComponentImageFilter()
    ccf.SetFullyConnected(False)  # 6-connected
    cc_img = ccf.Execute(mand_img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)

    small_ids = [lab for lab in stats.GetLabels() if stats.GetNumberOfPixels(lab) < MIN_MANDIBLE_VOXELS]
    if small_ids:
        cc_arr = sitk.GetArrayFromImage(cc_img)
        drop = np.isin(cc_arr, small_ids)
        mand_arr[drop] = 0  # background
        fused = sitk.GetImageFromArray(mand_arr.astype(np.uint8))
        fused.CopyInformation(seg_images[0])
        print(f"🪓  removed {len(small_ids)} small mandible components (< {MIN_MANDIBLE_VOXELS} voxels)")
    else:
        print("ℹ️  no small mandible components to remove")
else:
    print("ℹ️  no mandible voxels present — removal step skipped")

# ─────────────── 6. WRITE OUTPUT  ───────────────────────────────────────────
out_fp = os.path.join(OUTIMG, "ensembled_segmentation.mha")
if os.path.exists(out_fp):
    os.remove(out_fp)
sitk.WriteImage(fused, out_fp, useCompression=True)

print(f"\n✅  relabelled + iSTAPLE fusion mask → {out_fp}\n"
      f"   per-fold fresh outputs are under {TMP}/fold_*/prediction_testing/")
