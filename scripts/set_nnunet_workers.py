#!/usr/bin/env python3
"""
Script pour définir le nombre de workers pour nnUNet
Fixe le problème de segmentation fault sur MesoNET/Juliet
"""
import os

# Réduire drastiquement le nombre de workers pour éviter les segfaults
# 0 = pas de multiprocessing, tout dans le process principal
os.environ['nnUNet_n_proc_DA'] = '0'

print("Configuration nnUNet pour MesoNET:")
print(f"  nnUNet_n_proc_DA = {os.environ.get('nnUNet_n_proc_DA', 'non défini')}")
print("\nCette configuration évite les segmentation faults liés au multiprocessing")
