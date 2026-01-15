#!/bin/bash
#SBATCH --job-name=nnunet_finetune_002
#SBATCH --partition=mesonet
#SBATCH --account=m25172
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/home/almaratrat/prjiav/logs/finetune_002_%j.out
#SBATCH --error=/home/almaratrat/prjiav/logs/finetune_002_%j.err

################################################################################
# Script SBATCH pour fine-tuning nnUNet Dataset002 sur MesoNET/Juliet
# Conforme aux recommandations du guide PRJIAV
################################################################################

echo "=========================================="
echo "Job SLURM - Fine-tuning nnUNet Dataset002"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Définir PROJ_HOME
export PROJ_HOME="/home/almaratrat/prjiav"

# Vérifier que le conteneur existe
CONTAINER="$PROJ_HOME/containers/pytorch_25.09-py3.sif"
if [ ! -f "$CONTAINER" ]; then
    echo "ERREUR: Conteneur Apptainer non trouvé: $CONTAINER"
    exit 1
fi

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$PROJ_HOME/logs"

# Vérifier la disponibilité du GPU
echo "Vérification du GPU disponible:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Lancer le training dans le conteneur Apptainer
echo "Lancement du training dans le conteneur Apptainer..."
echo ""

apptainer exec \
    --bind $PROJ_HOME:$PROJ_HOME \
    --nv \
    $CONTAINER \
    bash $PROJ_HOME/data/scripts/finetune_dataset002.sh

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job terminé"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
