import shutil
import json
from pathlib import Path
import random
import os
from dotenv import load_dotenv

# ----- CONFIG -----
load_dotenv()
SOURCE_DIR = Path(os.getenv("MNI_DATASET","MNI"))  # Dossier contenant les patients (patient09, etc.)
TARGET_DIR = Path("Dataset003_MNI")

IMAGES_TR = TARGET_DIR / "imagesTr"
LABELS_TR = TARGET_DIR / "labelsTr"
IMAGES_TS = TARGET_DIR / "imagesTs"
LABELS_TS = TARGET_DIR / "labelsTs"

# Split ratio
TRAIN_RATIO = 0.8
RANDOM_SEED = 42  # Pour reproductibilité

# Clean target folders
for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FORMATAGE DATASET MNI POUR nnU-Net (avec split 80/20)")
print("=" * 70)

# ----- COLLECTER TOUS LES PATIENTS -----
all_patients = []

for patient_dir in sorted(SOURCE_DIR.iterdir()):
    if not patient_dir.is_dir():
        continue
    
    patient_id = patient_dir.name
    
    # Chercher les fichiers FLAIR et GOLD_STANDARD
    flair_file = None
    gold_file = None
    
    for f in patient_dir.glob("*.nii.gz"):
        if f.name.startswith("._"):
            continue
        
        filename_upper = f.name.upper()

        if f.name.startswith("FLAIR"):
            flair_file = f
            print(f.name)
        
        if "GOLD_STANDARD" in filename_upper and not f.name.startswith("_"):
            gold_file = f
    
    # Vérifier qu'on a les deux fichiers
    if flair_file and gold_file:
        all_patients.append({
            'id': patient_id,
            'flair': flair_file,
            'gold': gold_file
        })

print(f"\n📊 Total patients trouvés: {len(all_patients)}")

# ----- SPLIT TRAIN/TEST -----
random.seed(RANDOM_SEED)
random.shuffle(all_patients)

split_idx = int(len(all_patients) * TRAIN_RATIO)
train_patients = all_patients[:split_idx]
test_patients = all_patients[split_idx:]

print(f"✅ Train: {len(train_patients)} patients ({len(train_patients)/len(all_patients)*100:.1f}%)")
print(f"✅ Test: {len(test_patients)} patients ({len(test_patients)/len(all_patients)*100:.1f}%)")

# ----- CONVERSION TRAIN -----
print(f"\n{'='*70}")
print("TRAITEMENT DU TRAIN SET")
print(f"{'='*70}")

for patient in train_patients:
    patient_id = patient['id']
    print(f"\n  Processing {patient_id}...")
    
    shutil.copy(patient['flair'], IMAGES_TR / f"{patient_id}_0000.nii.gz")
    shutil.copy(patient['gold'], LABELS_TR / f"{patient_id}.nii.gz")
    
    print(f"    ✅ {patient_id} - FLAIR: {patient['flair'].name}")
    print(f"       {patient_id} - GOLD: {patient['gold'].name}")

# ----- CONVERSION TEST -----
print(f"\n{'='*70}")
print("TRAITEMENT DU TEST SET")
print(f"{'='*70}")

for patient in test_patients:
    patient_id = patient['id']
    print(f"\n  Processing {patient_id}...")
    
    shutil.copy(patient['flair'], IMAGES_TS / f"{patient_id}_0000.nii.gz")
    shutil.copy(patient['gold'], LABELS_TS / f"{patient_id}.nii.gz")
    
    print(f"    ✅ {patient_id} - FLAIR: {patient['flair'].name}")
    print(f"       {patient_id} - GOLD: {patient['gold'].name}")

# ----- dataset.json -----
dataset_json = {
    "channel_names": {
        "0": "FLAIR"
    },
    "labels": {
        "background": 0,
        "lesion": 1
    },
    "numTraining": len(list(LABELS_TR.glob("*.nii.gz"))),
    "numTest": len(list(LABELS_TS.glob("*.nii.gz"))),
    "file_ending": ".nii.gz"
}

with open(TARGET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# Sauvegarder la liste des splits pour référence
splits_info = {
    "train": [p['id'] for p in train_patients],
    "test": [p['id'] for p in test_patients],
    "train_ratio": TRAIN_RATIO,
    "random_seed": RANDOM_SEED
}

with open(TARGET_DIR / "splits_info.json", "w") as f:
    json.dump(splits_info, f, indent=4)

print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print(f"✅ Total patients: {len(all_patients)}")
print(f"✅ Train: {len(train_patients)} patients")
print(f"✅ Test: {len(test_patients)} patients")
print(f"\n📂 TRAIN SET:")
print(f"   - Images: {len(list(IMAGES_TR.glob('*.nii.gz')))}")
print(f"   - Labels: {len(list(LABELS_TR.glob('*.nii.gz')))}")
print(f"\n📂 TEST SET:")
print(f"   - Images: {len(list(IMAGES_TS.glob('*.nii.gz')))}")
print(f"   - Labels: {len(list(LABELS_TS.glob('*.nii.gz')))}")
print(f"\n✅ Dataset JSON créé: {TARGET_DIR / 'dataset.json'}")
print(f"✅ Splits info sauvegardé: {TARGET_DIR / 'splits_info.json'}")
print("\nStructure créée:")
print(f"  {TARGET_DIR}/")
print(f"    ├── imagesTr/")
print(f"    ├── labelsTr/")
print(f"    ├── imagesTs/")
print(f"    ├── labelsTs/")
print(f"    ├── dataset.json")
print(f"    └── splits_info.json")
print("\n✅ DONE — Dataset MNI formaté avec split 80/20")
print("=" * 70)
