import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paramètres
FOLD_ID = 4
JSON_PATH = f"C:/Users/agmau/OneDrive/Documents/INSA_Lyon/5A/PRJIAV/work_dir/nnUNet_inference/inference_before_finetuning/output_inference_MSLesSeg_Tr_per_fold/fold_{FOLD_ID}/summary.json"
CSV_PATH = "evaluation_results/MSLesSeg_full/full_evaluation_results.csv"
OUTPUT_JSON = f"C:/Users/agmau/OneDrive/Documents/INSA_Lyon/5A/PRJIAV/work_dir/nnUNet_inference/inference_before_finetuning/output_inference_MSLesSeg_Tr_per_fold/fold_{FOLD_ID}/summary_completed.json"

LABEL_KEY = "1"  # classe de segmentation


# Chargement des fichiers
with open(JSON_PATH, "r") as f:
    summary = json.load(f)

df = pd.read_csv(CSV_PATH)
df_fold = df[df["fold"] == FOLD_ID]

# Indexation rapide par patient
df_fold = df_fold.set_index("case_id")


# Complétion des métriques par images
all_metrics_values = defaultdict(list)

for case in summary["metric_per_case"]:
    pred_path = Path(case["prediction_file"])
    patient_id = pred_path.stem.replace(".nii", "")

    if patient_id not in df_fold.index:
        continue  # sécurité : on ne crée aucune nouvelle image

    csv_row = df_fold.loc[patient_id]

    case_metrics = case["metrics"].setdefault(LABEL_KEY, {})

    for csv_key, json_key in {
        "dice": "Dice",
        "iou": "IoU",
        "lesion_f1": "lesion_f1",
        "lesion_precision": "lesion_precision",
        "lesion_recall": "lesion_recall",
        "hd95": "HD95",
        "assd": "ASSD",
    }.items():

        if json_key not in case_metrics and pd.notna(csv_row[csv_key]):
            case_metrics[json_key] = float(csv_row[csv_key])

    # Collecte pour le recalcul global
    for k, v in case_metrics.items():
        if isinstance(v, (int, float)):
            all_metrics_values[k].append(v)


# Recalcul des métriques globales
def compute_mean(metrics_dict):
    return {k: sum(v) / len(v) for k, v in metrics_dict.items() if len(v) > 0}

mean_metrics = compute_mean(all_metrics_values)

summary["mean"][LABEL_KEY] = mean_metrics
summary["foreground_mean"] = mean_metrics.copy()


# Sauvegarde
with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=4)

print(f"✅ JSON complété et sauvegardé dans : {OUTPUT_JSON}")
