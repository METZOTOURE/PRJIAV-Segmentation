#!/usr/bin/env python3
"""
Patch pour forcer le multiprocessing à utiliser 'spawn' au lieu de 'fork'
Cela évite les segfaults avec CUDA sur les clusters HPC
"""
import multiprocessing as mp

# Forcer le mode 'spawn' pour éviter les problèmes avec CUDA
try:
    mp.set_start_method('spawn', force=True)
    print("✓ Multiprocessing set to 'spawn' mode (safe for CUDA)")
except RuntimeError:
    # Déjà défini, ignorer
    pass
