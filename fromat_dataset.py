import os
import shutil
import json
from pathlib import Path

# ----- CONFIG -----
SOURCE_DIR = Path("MSLesSeg_Dataset")  
TARGET_DIR = Path("nnUNet_raw/Dataset001_MSLesSeg")

IMAGES_TR = TARGET_DIR / "imagesTr"
LABELS_TR = TARGET_DIR / "labelsTr"
IMAGES_TS = TARGET_DIR / "imagesTs"

# Clean target folders
for d in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


# ----- MODALITY DETECTOR (robust for ALL your filenames) -----
def detect_modality(filename: str):
    n = filename.upper()

    if "FLAIR" in n:
        return "FLAIR"

    # Modality T1 (NOT timepoint)
    if n.endswith("_T1.NII.GZ") or "_T1_T1" in n:
        return "T1"

    # Modality T2
    if n.endswith("_T2.NII.GZ") or "_T1_T2" in n:
        return "T2"

    if "MASK" in n:
        return "MASK"

    return None


# ----- DATASET CONVERSION -----
def convert_split(split_name, is_training=True):
    split_dir = SOURCE_DIR / split_name

    for patient_dir in sorted(split_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        print(f"\nProcessing {patient_id} ...")

        # TRAIN uses timepoint T1/
        if is_training:
            src = patient_dir / "T1"
            if not src.exists():
                print(f"⚠️ No T1 folder for {patient_id}, skipping.")
                continue

        # TEST uses files directly in Pxx/
        else:
            src = patient_dir

        # detect files
        flair = t1 = t2 = mask = None

        for f in src.glob("*.nii.gz"):
            mod = detect_modality(f.name)
            if mod == "FLAIR":
                flair = f
            elif mod == "T1":
                t1 = f
            elif mod == "T2":
                t2 = f
            elif mod == "MASK":
                mask = f

        # -------- TRAIN --------
        if is_training:
            if mask is None:
                print(f"⚠️ No MASK found for {patient_id}, skipping.")
                continue

            if flair: shutil.copy(flair, IMAGES_TR / f"{patient_id}_0000.nii.gz")
            if t1: shutil.copy(t1, IMAGES_TR / f"{patient_id}_0001.nii.gz")
            if t2: shutil.copy(t2, IMAGES_TR / f"{patient_id}_0002.nii.gz")
            shutil.copy(mask, LABELS_TR / f"{patient_id}.nii.gz")

        # -------- TEST --------
        else:
            # test has NO masks
            if flair: shutil.copy(flair, IMAGES_TS / f"{patient_id}_0000.nii.gz")
            if t1: shutil.copy(t1, IMAGES_TS / f"{patient_id}_0001.nii.gz")
            if t2: shutil.copy(t2, IMAGES_TS / f"{patient_id}_0002.nii.gz")


# ----- EXECUTION -----
convert_split("train", True)
convert_split("test", False)

# ----- dataset.json -----
dataset_json = {
    "name": "MSLesSeg",
    "description": "MS lesion dataset using only timepoint T1 and all modalities",
    "tensorImageSize": "4D",
    "modality": {
        "0": "FLAIR",
        "1": "T1",
        "2": "T2"
    },
    "labels": {
        "0": "background",
        "1": "lesion"
    },
    "numTraining": len(list(LABELS_TR.glob("*.nii.gz"))),
    "numTest": len(list(IMAGES_TS.glob("*_0000.nii.gz"))),
    "file_ending": ".nii.gz"
}

with open(TARGET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("\n\n✅ DONE — dataset successfully converted.")
