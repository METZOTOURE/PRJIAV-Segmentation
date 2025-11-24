import os
import shutil
import json
from dotenv import load_dotenv
from pathlib import Path

# ----- CONFIG -----
load_dotenv()
SOURCE_DIR = Path(os.getenv("PREPROCESSED_DATA_DIR", "MSLesSeg_Dataset"))
TARGET_DIR = Path("Dataset001_MSLesSeg")

TRAIN_DIR = Path(os.getenv("TRAIN_DATA_DIR", "MSLesSeg_Dataset/train"))
TEST_DIR = Path(os.getenv("TEST_DATA_DIR", "MSLesSeg_Dataset/test"))

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
def convert_split(split_dir, is_training=True):
    for patient_dir in sorted(split_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        print(f"\nProcessing {patient_id} ...")

        # TRAIN: Process all timepoints T1, T2, T3, T4
        if is_training:
            timepoints = ["T1", "T2", "T3", "T4"]
            
            for timepoint in timepoints:
                src = patient_dir / timepoint
                
                if not src.exists():
                    print(f"⚠️ No {timepoint} folder for {patient_id}, skipping {timepoint}.")
                    continue

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

                if mask is None:
                    print(f"⚠️ No MASK found for {patient_id}/{timepoint}, skipping.")
                    continue

                if flair: shutil.copy(flair, IMAGES_TR / f"{patient_id}_{timepoint}_0000.nii.gz")
                if t1: shutil.copy(t1, IMAGES_TR / f"{patient_id}_{timepoint}_0001.nii.gz")
                if t2: shutil.copy(t2, IMAGES_TR / f"{patient_id}_{timepoint}_0002.nii.gz")
                shutil.copy(mask, LABELS_TR / f"{patient_id}_{timepoint}.nii.gz")

        # TEST: Files directly in patient folder (no timepoint subfolders)
        else:
            src = patient_dir
            
            # detect files
            flair = t1 = t2 = None

            for f in src.glob("*.nii.gz"):
                mod = detect_modality(f.name)
                if mod == "FLAIR":
                    flair = f
                elif mod == "T1":
                    t1 = f
                elif mod == "T2":
                    t2 = f

            # Copy with simple patient ID (no timepoint suffix for test)
            if flair: shutil.copy(flair, IMAGES_TS / f"{patient_id}_0000.nii.gz")
            if t1: shutil.copy(t1, IMAGES_TS / f"{patient_id}_0001.nii.gz")
            if t2: shutil.copy(t2, IMAGES_TS / f"{patient_id}_0002.nii.gz")


# ----- EXECUTION -----
convert_split(TRAIN_DIR, True)
convert_split(TEST_DIR, False)

# ----- dataset.json -----
dataset_json = {
    "channel_names": {
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
