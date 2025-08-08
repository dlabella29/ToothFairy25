#!/usr/bin/env python
# inference_ToothFairy_ensemble.py           (Aug‑2025 – iSTAPLE + clean‑up)

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

# ─────────────── HOUSE‑KEEPING ──────────────────────────────────────────────
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
       for i in range(6)]

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

# ─────────────── 5. LABEL FUSION WITH MultiLabel‑STAPLE ─────────────────────
staple = sitk.MultiLabelSTAPLEImageFilter()
staple.SetMaximumNumberOfIterations(3)
staple.SetTerminationUpdateThreshold(1e-4)

print("\n🧮  running MultiLabel‑STAPLE …")
fused = staple.Execute(seg_images)
fused = sitk.Cast(fused, sitk.sitkUInt8)          # 0–46 classes → uint8
fused.CopyInformation(seg_images[0])

# ─────────────── 5b. RELABEL BEFORE SAVING  ────────────────────────────────
# mapping: old → new
REMAP = {19: 53, 20: 52, 29: 51, 30: 50, 39: 48, 40: 47}

arr = sitk.GetArrayFromImage(fused)               # z‑y‑x numpy array
for old, new in REMAP.items():
    arr[arr == old] = new

# optionally verify that all old labels are gone
# assert not np.isin(list(REMAP.keys()), arr)

fused = sitk.GetImageFromArray(arr)
fused.CopyInformation(seg_images[0])              # preserve origin/spacing

# ─────────────── 6. WRITE OUTPUT  ───────────────────────────────────────────
out_fp = os.path.join(OUTIMG, "ensembled_segmentation.mha")
if os.path.exists(out_fp):                        # avoid FileNotFoundError
    os.remove(out_fp)

sitk.WriteImage(fused, out_fp, useCompression=True)

print(f"\n✅  relabelled + iSTAPLE fusion mask → {out_fp}\n"
      f"   per‑fold fresh outputs are under {TMP}/fold_*/prediction_testing/")

