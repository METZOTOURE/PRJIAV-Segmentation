#!/bin/bash
################################################################################
# Wrapper pour lancer nnUNet de manière sécurisée sur MesoNET/Juliet
# Évite les segmentation faults liés au multiprocessing
################################################################################

set -e

echo "=========================================="
echo "Configuration de l'environnement pour nnUNet sur MesoNET"
echo "=========================================="

export PYTHONWARNINGS="ignore"

# CRITIQUE: Limiter le multiprocessing des dataloaders pour éviter les segfaults
# On utilise 1 worker: permet le préchargement sans surcharger le système
export nnUNet_n_proc_DA=0
echo "✓ nnUNet_n_proc_DA=$nnUNet_n_proc_DA (1 worker pour préchargement)"

# CRITIQUE: Désactiver la compilation JIT de nnUNet (problèmes avec Triton/CUDA)
export nnUNet_compile=false
echo "✓ nnUNet_compile=$nnUNet_compile (désactive compilation JIT)"

# Variables d'environnement pour éviter les conflits UCX/InfiniBand
export UCX_TLS=tcp,self
export UCX_NET_DEVICES=all
echo "✓ UCX_TLS=$UCX_TLS (évite conflits InfiniBand)"

# Configurer OpenMP pour utiliser tous les CPUs disponibles
# Si SLURM_CPUS_PER_TASK est défini, on l'utilise, sinon on prend 16 par défaut
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "✓ OMP_NUM_THREADS=$OMP_NUM_THREADS (CPU threads pour calculs parallèles)"

# CRITIQUE: Forcer PyTorch à utiliser 'spawn' au lieu de 'fork' pour le multiprocessing
# Cela évite les segfaults avec CUDA sur les clusters HPC
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

# Forcer l'affichage de la progression en temps réel
export TQDM_MININTERVAL=1
export PYTHONIOENCODING=utf-8

# Variables pour stabiliser le multiprocessing
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "✓ Configuration multiprocessing sécurisée activée"

echo "=========================================="
echo ""

# Lancer la commande nnUNet passée en paramètres
echo "Lancement de: $@"
echo ""

# Appliquer le patch multiprocessing directement dans Python
python3 -c "
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import sys
import subprocess
sys.exit(subprocess.call(sys.argv[1:]))
" "$@"
