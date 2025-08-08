#!/usr/bin/env python3
import os
import json

def create_toothfairy_autoseg_datalist():
    # ——— Data dirs ———
    images_dir = "/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/imagesTr"
    labels_dir = "/media/dlabella29/Extreme Pro/Grand Challenge Data/ToothFairy3/labelsTrChallengeFixed0-46"
    ext = ".nii.gz"
    suffix = "_0000" + ext

    # ——— Enumerate and sort cases ———
    all_images = sorted(f for f in os.listdir(images_dir) if f.endswith(suffix))
    cases = [fname[:-len(suffix)] for fname in all_images]

    # ——— Select test cases by segment ———
    first20 = cases[:20]
    middle  = cases[80:300]
    last10  = cases[-10:]

    test_cases = []
    #test_cases += first20[:3]    # 3 from first 20
    #test_cases += middle[:7]      # 7 from cases[80:300]
    #test_cases += last10[-2:]     # 2 from last 10

    # ——— Remaining are training ———
    train_cases = [c for c in cases if c not in test_cases]

    # ——— Build datalist ———
    datalist = {"training": [], "testing": []}

    # Training entries with fold assignment 0–7 :contentReference[oaicite:2]{index=2}
    for idx, case in enumerate(train_cases):
        fold = idx % 5
        img_path = os.path.join(images_dir, f"{case}_0000{ext}")
        lbl_path = os.path.join(labels_dir,  f"{case}{ext}")
        datalist["training"].append({
            "fold":  fold,
            "image": [img_path],
            "label": lbl_path
        })

    # Testing entries now include labels too (no fold) :contentReference[oaicite:3]{index=3}
    for case in test_cases:
        img_path = os.path.join(images_dir, f"{case}_0000{ext}")
        lbl_path = os.path.join(labels_dir,  f"{case}{ext}")
        datalist["testing"].append({
            "image": [img_path],
            "label": lbl_path
        })

    # ——— Write out JSON ———
    out_dir  = "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/ToothFairy"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ToothFairyJsonAutoSegSubmit.json")
    with open(out_path, "w") as f:
        json.dump(datalist, f, indent=4)

    print(f"Wrote datalist with {len(train_cases)} training and "
          f"{len(test_cases)} testing cases to:\n  {out_path}")

if __name__ == "__main__":
    create_toothfairy_autoseg_datalist()
