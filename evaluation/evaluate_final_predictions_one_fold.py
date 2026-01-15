import os
from dotenv import load_dotenv
import nibabel as nib
import numpy as np
import math
from medpy.metric.binary import dc
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_erosion
# Correction : Importer la fonction de rééchantillonnage appropriée de nibabel
from nibabel.processing import resample_from_to

# --- CHEMINS À ADAPTER ---
# Assurez-vous que ces chemins sont corrects pour votre environnement
load_dotenv()
fold_id = 0
gt_dir = os.getenv("GT_PATH")
pred_dir = os.getenv("PREDICTIONS_PATH") + f"/fold_{fold_id}"
# --- FIN DE LA CONFIGURATION ---

def load_nifti_image(path):
    """
    Charge une image NIfTI et retourne l'objet image complet (pas seulement les données).
    Ceci est nécessaire pour accéder à l'affine et au header.
    """
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return None
    try:
        return nib.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def get_surface_coords(mask_bin, voxel_spacing):
    """Calcule les coordonnées de la surface d'un masque binaire."""
    if not mask_bin.any():
        return np.array([])
    eroded = binary_erosion(mask_bin)
    surface = np.logical_xor(mask_bin, eroded)
    coords = np.array(np.where(surface)).T
    return coords * np.array(voxel_spacing)

def hd95_numpy(gt_coords, pred_coords):
    """Calcule la distance de Hausdorff au 95ème percentile."""
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return np.inf
    dists_gt_to_pred = cdist(gt_coords, pred_coords, metric='euclidean').min(axis=1)
    dists_pred_to_gt = cdist(pred_coords, gt_coords, metric='euclidean').min(axis=1)
    return max(np.percentile(dists_gt_to_pred, 95), np.percentile(dists_pred_to_gt, 95))

def aasd_numpy(gt_coords, pred_coords):
    """Calcule la distance de surface symétrique moyenne (Average Symmetric Surface Distance)."""
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return np.inf
    dists_gt_to_pred = cdist(gt_coords, pred_coords, metric='euclidean').min(axis=1)
    dists_pred_to_gt = cdist(pred_coords, gt_coords, metric='euclidean').min(axis=1)
    return (np.mean(dists_gt_to_pred) + np.mean(dists_pred_to_gt)) / 2.0

def compute_iou(gt_mask_bin, pred_mask_bin):
    """Calcule l'Intersection sur l'Union (IoU)."""
    intersection = np.logical_and(gt_mask_bin, pred_mask_bin).sum()
    union = np.logical_or(gt_mask_bin, pred_mask_bin).sum()
    return 1.0 if union == 0 else intersection / union

def compute_lesion_metrics(gt_mask_bin, pred_mask_bin):
    """Calcule le F1-score, la précision et le rappel au niveau des lésions."""
    gt_labels, num_gt = label(gt_mask_bin)
    pred_labels, num_pred = label(pred_mask_bin)

    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0  # F1, Precision, Recall

    tp = 0
    fn_lesions = list(range(1, num_gt + 1))

    # Trouver les Vrais Positifs (TP) et Faux Négatifs (FN)
    for i in range(1, num_gt + 1):
        gt_lesion_mask = (gt_labels == i)
        if np.any(pred_mask_bin[gt_lesion_mask]):
            tp += 1
            if i in fn_lesions:
                fn_lesions.remove(i)

    # Trouver les Faux Positifs (FP)
    fp = 0
    for i in range(1, num_pred + 1):
        pred_lesion_mask = (pred_labels == i)
        if not np.any(gt_mask_bin[pred_lesion_mask]):
            fp += 1

    fn = len(fn_lesions)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall

def compute_all_metrics(gt_data, pred_data, spacing):
    """Calcule l'ensemble des métriques pour une paire de masques."""
    if not gt_data.any() and not pred_data.any():
        return {'dice': 1.0, 'hd95': 0.0, 'aasd': 0.0, 'iou': 1.0, 'lesion_f1': 1.0, 'lesion_precision': 1.0, 'lesion_recall': 1.0}

    # Métriques de surface (HD95, AASD)
    gt_coords = get_surface_coords(gt_data, spacing)
    pred_coords = get_surface_coords(pred_data, spacing)
    hd95 = hd95_numpy(gt_coords, pred_coords)
    aasd = aasd_numpy(gt_coords, pred_coords)

    # Métriques de chevauchement (Dice, IoU)
    dice_score = dc(pred_data, gt_data)
    iou_score = compute_iou(gt_data, pred_data)

    # Métriques au niveau des lésions
    lesion_f1, lesion_precision, lesion_recall = compute_lesion_metrics(gt_data, pred_data)

    return {
        'dice': dice_score,
        'hd95': hd95,
        'aasd': aasd,
        'iou': iou_score,
        'lesion_f1': lesion_f1,
        'lesion_precision': lesion_precision,
        'lesion_recall': lesion_recall
    }

def main():
    """Fonction principale pour charger les données, les traiter et calculer les statistiques."""
    all_results = {'dice': [], 'hd95': [], 'aasd': [], 'iou': [], 'lesion_f1': [], 'lesion_precision': [], 'lesion_recall': []}
    try:
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    except FileNotFoundError:
        print(f"ERREUR : Le dossier des prédictions n'a pas été trouvé : {pred_dir}")
        return

    print(f"Évaluation de {len(pred_files)} cas trouvés dans {pred_dir}")

    for pred_filename in pred_files:
        case_id = pred_filename.replace('.nii.gz', '')
        print(f"--- Traitement du cas : {case_id} ---")

        pred_path = os.path.join(pred_dir, pred_filename)
        gt_filename = pred_filename
        gt_path = os.path.join(gt_dir, gt_filename)

        # Charger les objets images NIfTI complets
        pred_img = load_nifti_image(pred_path)
        gt_img = load_nifti_image(gt_path)

        if gt_img is None or pred_img is None:
            print("  Données manquantes pour ce cas, cas ignoré.")
            continue
        
        # --- CORRECTION APPLIQUÉE ICI ---
        # Rééchantillonner l'image de prédiction pour qu'elle corresponde à l'espace
        # (orientation, espacement, dimensions) de l'image de vérité terrain.
        # 'order=0' spécifie une interpolation au plus proche voisin, ce qui est
        # essentiel pour les masques de segmentation pour éviter de créer des
        # valeurs non binaires.
        print("  Alignement de la prédiction sur l'espace de la vérité terrain...")
        pred_img_resampled = resample_from_to(pred_img, gt_img, order=0)
        
        # Extraire les données en tant que tableaux numpy booléens APRÈS le rééchantillonnage
        pred_data = pred_img_resampled.get_fdata().astype(np.bool_)
        gt_data = gt_img.get_fdata().astype(np.bool_)
        
        # L'espacement est maintenant garanti d'être le même.
        # Nous utilisons celui de l'image de référence (vérité terrain).
        gt_spacing = gt_img.header.get_zooms()[:3]

        metrics = compute_all_metrics(gt_data, pred_data, gt_spacing)
        print(f"    Métrique -> Dice: {metrics['dice']:.4f}, HD95(mm): {metrics['hd95']:.2f}, AASD(mm): {metrics['aasd']:.2f}, Lesion F1: {metrics['lesion_f1']:.4f}")

        for key, value in metrics.items():
            all_results[key].append(value)

    print("\n\n--- STATISTIQUES FINALES DE L'ENSEMBLE DE TEST ---")
    # Remplacer les infinis (cas où une surface est vide) par NaN pour un calcul de moyenne robuste
    for metric_name in ['hd95', 'aasd']:
        all_results[metric_name] = [v if v != np.inf else np.nan for v in all_results[metric_name]]

    print(f"Nombre total de cas évalués : {len(all_results['dice'])}")
    for metric_name, values in all_results.items():
        if values:
            # np.nanmean et np.nanstd ignorent les valeurs NaN dans les calculs
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values, ddof=1)
            sem_val = std_val/math.sqrt(5)
            print(f"  - {metric_name.capitalize():<20}: {mean_val:.4f} ± {sem_val:.4f} ({mean_val-sem_val*1.96:.4f} -- {mean_val+sem_val*1.96:.4f})")
        else:
            print(f"  - {metric_name.capitalize():<20}: N/A (pas de données)")

if __name__ == "__main__":
    main()
