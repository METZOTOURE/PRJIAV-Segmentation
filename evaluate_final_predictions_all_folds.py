import os
import glob
import re
import nibabel as nib
import numpy as np
import math
from medpy.metric.binary import dc
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_erosion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


gt_base_dir = Path("C:/Users/agmau/OneDrive/Documents/INSA_Lyon/5A/PRJIAV/work_dir/nnUNet_raw_data_base/Dataset002_MSLesSeg_FLAIR/labelsTs")
predictions_base_dir = Path("C:/Users/agmau/OneDrive/Documents/INSA_Lyon/5A/PRJIAV/work_dir/nnUNet_inference/inference_before_finetuning/output_inference_MSLesSeg_Ts_per_fold")
output_dir = Path("./evaluation_results/MSLesSeg_Ts")


def get_surface_distances(gt_bin, pred_bin, voxel_spacing=None):
    """Calcule les distances de surface entre deux masques binaires.
    Retourne (dists_gt_to_pred, dists_pred_to_gt) ou (None, None) si un masque est vide.
    """
    def surface_voxels(mask):
        eroded = binary_erosion(mask)
        return np.array(np.where(np.logical_xor(mask, eroded))).T

    coords_gt = surface_voxels(gt_bin)
    coords_pred = surface_voxels(pred_bin)

    # Modification: Retourne np.inf pour HD95/ASSD si un seul masque est vide
    if len(coords_gt) == 0 and len(coords_pred) == 0:
        return np.array([]), np.array([])  # Pour que min(axis=1) ne plante pas, mais le calcul final sera 0
    if len(coords_gt) == 0 or len(coords_pred) == 0:
        return None, None  # Indique un cas où la métrique doit être infinie

    if voxel_spacing is not None:
        voxel_spacing_arr = np.array(voxel_spacing)
        coords_gt = coords_gt * voxel_spacing_arr
        coords_pred = coords_pred * voxel_spacing_arr

    dists_gt_to_pred = cdist(coords_gt, coords_pred, metric='euclidean').min(axis=1)
    dists_pred_to_gt = cdist(coords_pred, coords_gt, metric='euclidean').min(axis=1)
    return dists_gt_to_pred, dists_pred_to_gt

def assd_numpy(gt_bin, pred_bin, voxel_spacing=None):
    """Calcul de l'Average Symmetric Surface Distance (ASSD) en mm."""
    dists_gt_to_pred, dists_pred_to_gt = get_surface_distances(gt_bin, pred_bin, voxel_spacing)
    
    if dists_gt_to_pred is None:  # Cas où un seul masque est vide
        return np.inf
    if len(dists_gt_to_pred) == 0 and len(dists_pred_to_gt) == 0:  # Cas où les deux masques sont vides
        return 0.0

    assd = (np.mean(dists_gt_to_pred) + np.mean(dists_pred_to_gt)) / 2.0
    return assd


def hd95_numpy(gt_bin, pred_bin, voxel_spacing=None):
    """Calcul du 95ème percentile de la distance de Hausdorff (HD95) en mm."""
    dists_gt_to_pred, dists_pred_to_gt = get_surface_distances(gt_bin, pred_bin, voxel_spacing)
    
    if dists_gt_to_pred is None:  # Cas où un seul masque est vide
        return np.inf
    if len(dists_gt_to_pred) == 0 and len(dists_pred_to_gt) == 0:  # Cas où les deux masques sont vides
        return 0.0

    hd95 = max(np.percentile(dists_gt_to_pred, 95), np.percentile(dists_pred_to_gt, 95))
    return hd95

def load_nifti_data_and_spacing(path):
    """Charge les données d'un fichier NIFTI et l'espacement des voxels."""
    if not os.path.exists(path):
        print(f"AVERTISSEMENT: Fichier non trouvé: {path}")
        return None, None
    try:
        img = nib.load(path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        return data, spacing
    except Exception as e:
        print(f"ERREUR: Impossible de charger {path}: {e}")
        return None, None

def compute_iou(gt_mask_bin, pred_mask_bin):
    """Calcule l'Intersection over Union (IoU) pour des masques binaires."""
    intersection = np.logical_and(gt_mask_bin, pred_mask_bin).sum()
    union = np.logical_or(gt_mask_bin, pred_mask_bin).sum()
    if union == 0: 
        return 1.0
    return intersection / union

def compute_lesion_f1_precision_recall(gt_bin, pred_bin, iou_threshold=0.5):
    """Calcule le F1-score, la précision et le rappel au niveau des lésions.
    Version améliorée du Code 2 pour une gestion claire des cas sans lésions.
    """
    labeled_gt, num_gt_lesions = label(gt_bin)
    labeled_pred, num_pred_lesions = label(pred_bin)

    if num_gt_lesions == 0 and num_pred_lesions == 0:
        return 1.0, 1.0, 1.0  # Cas parfait : aucune lésion attendue, aucune prédite
    if num_gt_lesions == 0:
        # Aucune lésion attendue, mais certaines ont été prédites (FP)
        return 0.0, 0.0, 1.0  # F1=0, Précision=0, Rappel=1 (car toutes les 0 lésions GT ont été "trouvées")
    if num_pred_lesions == 0:
        # Des lésions étaient attendues, mais aucune n'a été prédite (FN)
        return 0.0, 1.0, 0.0  # F1=0, Précision=1 (car aucune des prédictions n'est fausse), Rappel=0

    tp_count = 0
    gt_matched = [False] * num_gt_lesions

    for i in range(1, num_pred_lesions + 1):
        pred_lesion_mask = (labeled_pred == i)
        # Trouver les lésions GT qui chevauchent cette prédiction
        overlapping_gt_labels = np.unique(labeled_gt[pred_lesion_mask])

        best_iou = 0
        best_gt_idx = -1
        for gt_label in overlapping_gt_labels:
            if gt_label == 0: 
                continue  # Ignorer l'arrière-plan

            gt_lesion_mask = (labeled_gt == gt_label)
            current_iou = compute_iou(gt_lesion_mask, pred_lesion_mask)

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = gt_label - 1

        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp_count += 1
            gt_matched[best_gt_idx] = True

    fp_count = num_pred_lesions - tp_count
    fn_count = num_gt_lesions - tp_count

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score, precision, recall

def compute_all_metrics(gt_data, pred_data, voxel_spacing):
    """Calcule toutes les métriques pour une paire de masques."""
    gt_bin = (gt_data > 0).astype(np.uint8)
    pred_bin = (pred_data > 0).astype(np.uint8)
    metrics = {k: np.nan for k in ['dice', 'hd95', 'assd', 'iou', 'lesion_f1', 'lesion_precision', 'lesion_recall']}
    gt_is_empty, pred_is_empty = not np.any(gt_bin), not np.any(pred_bin)

    if gt_is_empty and pred_is_empty:
        metrics.update({'dice': 1.0, 'iou': 1.0, 'hd95': 0.0, 'assd': 0.0})
    elif gt_is_empty or pred_is_empty:
        metrics.update({'dice': 0.0, 'iou': 0.0, 'hd95': np.inf, 'assd': np.inf})
    else:
        try: 
            metrics['dice'] = dc(pred_bin, gt_bin)
        except: 
            print(f"  Erreur Dice")
        try: 
            metrics['iou'] = compute_iou(gt_bin, pred_bin)
        except: 
            print(f"  Erreur IoU")
        try: 
            metrics['hd95'] = hd95_numpy(gt_bin, pred_bin, voxel_spacing)
        except: 
            print(f"  Erreur HD95")
        try: 
            metrics['assd'] = assd_numpy(gt_bin, pred_bin, voxel_spacing)
        except: 
            print(f"  Erreur ASSD")

    try:
        lf1, lprec, lrec = compute_lesion_f1_precision_recall(gt_bin, pred_bin)
        metrics.update({'lesion_f1': lf1, 'lesion_precision': lprec, 'lesion_recall': lrec})
    except: 
        print(f"  Erreur Lesion F1")
    return metrics


def main():

    all_individual_results = []
    num_folds = 5

    for fold in range(num_folds):
        print(f"\n--- Traitement du Fold {fold} ---")
        validation_pred_dir = predictions_base_dir / f"fold_{fold}" #/ "validation"
        if not validation_pred_dir.is_dir():
            print(f"  Dossier de prédictions non trouvé pour le fold {fold} à '{validation_pred_dir}'. Passage au suivant.")
            continue
        prediction_files = sorted(list(validation_pred_dir.glob("*.nii.gz")))
        if not prediction_files:
            print(f"  Aucun fichier de prédiction (.nii.gz) trouvé dans {validation_pred_dir}. Passage au suivant.")
            continue

        for pred_path in prediction_files:
            filename = os.path.basename(pred_path)
            case_id = filename.replace(".nii.gz", "")
            print(f"  Traitement du cas: {case_id}")

            gt_path = gt_base_dir / f"{case_id}.nii.gz"
            if not gt_path.exists():
                print(f"    Vérité terrain manquante pour {case_id}. Cas ignoré.")
                continue

            gt_data, gt_spacing = load_nifti_data_and_spacing(gt_path)
            pred_data, pred_spacing = load_nifti_data_and_spacing(pred_path)

            if gt_data is None or pred_data is None: 
                continue

            if gt_data.shape != pred_data.shape:
                print(f"    AVERTISSEMENT: Dimensions différentes pour {case_id}. Cas ignoré.")
                continue

            metrics = compute_all_metrics(gt_data, pred_data, gt_spacing)

            result_entry = {'fold': fold, 'case_id': case_id, **metrics}
            all_individual_results.append(result_entry)

    if not all_individual_results:
        print("\nAnalyse terminée, mais aucun cas n'a pu être traité.")
        return

    results_df = pd.DataFrame(all_individual_results)

    print("\n\n--- Moyennes par Fold ---")
    numeric_cols = results_df.select_dtypes(include=np.number).columns.drop(['fold'], errors='ignore')
    fold_means = results_df.groupby('fold')[numeric_cols].mean()
    print(fold_means.to_string(float_format="%.4f"))

    print("\n--- Moyennes Globales sur tous les cas ---")
    global_means = results_df[numeric_cols].mean()
    print(global_means.to_string(float_format="%.4f"))


    all_results = fold_means.to_dict('list')
    print("\n--- Métriques d'inférence (Moyenne ± SEM) ---")
    for metric_name, values in all_results.items():
        if values:
            # np.nanmean et np.nanstd ignorent les valeurs NaN dans les calculs
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values, ddof=1)
            sem_val = std_val/math.sqrt(5)
            print(f"  - {metric_name.capitalize():<20}: {mean_val:.4f} ± {sem_val:.4f} ({mean_val-sem_val*1.96:.4f} -- {mean_val+sem_val*1.96:.4f})")

    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "full_evaluation_results.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"\nRésultats détaillés sauvegardés dans : {csv_path}")


if __name__ == "__main__":
    main()


