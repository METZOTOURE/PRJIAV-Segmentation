#!/bin/bash

################################################################################
# Script de fine-tuning nnUNet pour Dataset002_MSLesSeg_FLAIR
# Utilise les poids pré-entraînés de Dataset004_MyTest avec transfer learning
# Version optimisée pour machine avec 80GB VRAM (utilise ResEncL)
################################################################################

set -e  # Arrêt en cas d'erreur

################################################################################
# PARAMÈTRES CONFIGURABLES
################################################################################

# Nombre d'epochs à entraîner (défaut: 1000 pour nnU-Net)
# Vous pouvez réduire ce nombre pour des tests rapides (ex: 100)
NUM_EPOCHS="${NUM_EPOCHS:-1000}"

# Liste des folds à entraîner (défaut: tous les folds 0-4)
# Exemples d'utilisation:
#   - Entraîner seulement le fold 0: FOLDS_TO_TRAIN="0"
#   - Entraîner les folds 0 et 1: FOLDS_TO_TRAIN="0 1"
#   - Entraîner tous les folds: FOLDS_TO_TRAIN="0 1 2 3 4"
FOLDS_TO_TRAIN="${FOLDS_TO_TRAIN:-0 1 2 3 4}"

################################################################################

echo "=========================================="
echo "Fine-tuning nnUNet - Dataset002_MSLesSeg_FLAIR"
echo "Avec transfer learning depuis Dataset004"
echo "=========================================="
echo ""
echo "Configuration de l'entraînement:"
echo "  - Nombre d'epochs: $NUM_EPOCHS"
echo "  - Folds à entraîner: $FOLDS_TO_TRAIN"
echo "=========================================="

# 1. Configuration de l'environnement
echo ""
echo "[1/5] Configuration de l'environnement..."

echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"

# Chemin vers les poids pré-entraînés de Dataset004
PRETRAINED_WEIGHTS="$nnUNet_results/Dataset004_MyTest/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
echo "  Poids pré-entraînés: $PRETRAINED_WEIGHTS"

# 2. Vérification des poids pré-entraînés
echo ""
echo "[2/5] Vérification des poids pré-entraînés de Dataset004..."
if [ ! -d "$PRETRAINED_WEIGHTS" ]; then
    echo "  ⚠ ERREUR: Les poids pré-entraînés n'ont pas été trouvés!"
    echo "  Chemin attendu: $PRETRAINED_WEIGHTS"
    echo ""
    echo "  Veuillez d'abord entraîner Dataset004 ou ajuster la variable PRETRAINED_WEIGHTS"
    exit 1
fi

# Vérifier qu'il y a au moins un checkpoint
if ! ls "$PRETRAINED_WEIGHTS"/fold_*/checkpoint_final.pth 1> /dev/null 2>&1; then
    echo "  ⚠ ERREUR: Aucun checkpoint_final.pth trouvé dans $PRETRAINED_WEIGHTS"
    exit 1
fi

echo "  ✓ Poids pré-entraînés trouvés!"
ls "$PRETRAINED_WEIGHTS"/fold_*/checkpoint_final.pth | head -n 5

# 3. Vérification et prétraitement du dataset
echo ""
echo "[3/5] Vérification et prétraitement de Dataset002..."
echo "  Note: Cette étape peut prendre plusieurs minutes"
echo "  Utilisation de ResEncL (~24GB VRAM) - Compatible avec votre GPU 80GB"

# Vérifier si le prétraitement a déjà été fait avec ResEncL
if [ -f "$nnUNet_preprocessed/Dataset002_MSLesSeg_FLAIR/nnUNetResEncUNetLPlans.json" ]; then
    echo "  Dataset002 déjà prétraité avec ResEncL. Voulez-vous le refaire? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "  Lancement du prétraitement avec ResEncL..."
        nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncL --verify_dataset_integrity
    else
        echo "  Prétraitement existant conservé."
    fi
else
    echo "  Lancement du prétraitement avec ResEncL..."
    nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncL --verify_dataset_integrity
fi

# 4. Fine-tuning avec transfer learning (ResEncL + poids de Dataset004)
echo ""
echo "[4/5] Fine-tuning de Dataset002 avec transfer learning..."
echo "  ✓ Utilisation de ResEncL avec poids pré-entraînés de Dataset004"
echo "  ✓ Compatible avec votre GPU 80GB VRAM"
echo "  ✓ Nombre d'epochs: $NUM_EPOCHS"
echo "  ✓ Folds à entraîner: $FOLDS_TO_TRAIN"
echo ""
echo "  Note: Le transfer learning devrait accélérer la convergence"

# Compter le nombre de folds à entraîner
FOLD_ARRAY=($FOLDS_TO_TRAIN)
NUM_FOLDS=${#FOLD_ARRAY[@]}
CURRENT_FOLD=0

for fold in $FOLDS_TO_TRAIN; do
    CURRENT_FOLD=$((CURRENT_FOLD + 1))
    echo ""
    echo "  === Entraînement fold $fold ($CURRENT_FOLD/$NUM_FOLDS) avec transfer learning ==="

    # Trouver le checkpoint correspondant au fold
    CHECKPOINT="$PRETRAINED_WEIGHTS/fold_$fold/checkpoint_final.pth"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  ⚠ Checkpoint non trouvé pour fold $fold: $CHECKPOINT"
        echo "  → Entraînement depuis zéro pour ce fold"
        nnUNetv2_train 002 3d_fullres $fold \
            -p nnUNetResEncUNetLPlans \
            --npz
    else
        echo "  ✓ Utilisation du checkpoint: $CHECKPOINT"
        nnUNetv2_train 002 3d_fullres $fold \
            -p nnUNetResEncUNetLPlans \
            -pretrained_weights "$CHECKPOINT" \
            --npz
    fi

    echo "  ✓ Fold $fold terminé!"
done

echo ""
echo "  ✓ Tous les folds sélectionnés ont été entraînés ($NUM_FOLDS/$NUM_FOLDS)"

# 5. Sélection de la meilleure configuration
echo ""
echo "[5/5] Sélection de la meilleure configuration..."
nnUNetv2_find_best_configuration 002 -c 3d_fullres -tr nnUNetTrainerResEncUNetL -p nnUNetResEncUNetLPlans

echo ""
echo "=========================================="
echo "Fine-tuning terminé avec succès!"
echo "=========================================="
echo ""
echo "Résumé:"
echo "  ✓ Architecture utilisée: ResEncL (~24GB VRAM)"
echo "  ✓ Transfer learning activé depuis Dataset004"
echo "  ✓ 5 folds entraînés"
echo ""
echo "Les modèles entraînés sont dans:"
echo "  $nnUNet_results/Dataset002_MSLesSeg_FLAIR/nnUNetTrainerResEncUNetL__nnUNetResEncUNetLPlans__3d_fullres/"
echo ""
echo "Pour lancer l'inférence, utilisez le script: inference_dataset002.sh"
echo "  (Pensez à adapter le script pour utiliser ResEncL si nécessaire)"
echo ""

