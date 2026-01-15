# PRJIAV-Segmentation

Projet de segmentation de lésions de sclérose en plaques (MS) utilisant nnU-Net.

## 📋 Prérequis

- Python 3.11+
- Au moins 8 GB RAM (16 GB recommandé)
- Espace disque : ~50 GB pour les données et modèles

## 🔧 Installation

### 1. Créer l'environnement virtuel

```powershell
# Créer le venv
python -m venv .venv

# Activer (PowerShell)
.\.venv\Scripts\Activate.ps1

# Ou (CMD)
.venv\Scripts\activate.bat
```

### 2. Installer les dépendances

Packages python nécessaires au bon fonctionnement du code présent dans ce repo Git : 
- matplotlib
- monai
- MedPy
- nibabel
- numpy
- pandas
- scikit-learn
- scipy
- seaborn
- simpleitk
- torch
- torchvision
- tqdm

```powershell
# Installation complète
pip install -r requirements.txt
```

Pour l'installation de nnU-Net, se référer au repo Git associé au projet.

## 📁 Structure du repo Git

### Fichier `.env` nécessaire

Certains fichiers de code utilisent des chemins présents dans un fichier `.env`. Ce fichier doit contenir au moins les chemins suivants : 
- `RAW_MSLESSEG_DATASET` : chemin vers les données brutes du dataset MSLesSeg
- `MSLESSEG_DATASET` : chemin vers les données pré-traitées du dataset MSLesSeg
- `TRAIN_DATA_DIR` : chemin vers les données d'entraînement pré-traitées du dataset MSLesSeg
- `TEST_DATA_DIR` : chemin vers les données de test pré-traitées du dataset MSLesSeg
- `MNI_DATASET` : chemin vers les données du dataset OpenMS
- `MSSEG1_DATASET`: chemin vers les données du dataset MSSEG 1 (utilisé par Zaineb pour le premier entraînement de nnU-Net)
- `nnUNet_preprocessed` : chemin vers les données pré-traitées par nnU-Net
- `GT_PATH` : chemin vers les véritables labels
- `PREDICTIONS_PATH` : chemin vers les labels prédis (sans le dossier fold)
- `OUTPUT_PATH` : chemin vers l'endroit où doivent être enregistrés les métriques d'évalution et boxplots associées

### 📁 Dossier `evaluation`

Ce dossier comprend tout le code nécessaire pour réaliser l'évaluation des prédictions. 

**Pensez à modifier les chemins vers les différents dossiers dans le code avant de les exécuter.**

- `complete_summary.py` : complète les fichiers `summary.json` généré par nnU-Net lors de l'évaluation initiale des prédictions avec des métriques supplémentaires
- `evaluate_final_predictions_all_folds.py` : évalue les prédictions réalisées par nnU-Net sur les 5 folds de validation croisée de nnU-Net, regroupe les métriques dans un fichier .csv et calcule les résultats moyens sur les 5 folds 
- `evaluate_final_predictions_one_fold.py`: évalue les prédictions réalisées par nnU-Net sur les 1 fold de validation croisée de nnU-Net
- `generate_boxplots.py` : à partir d'un fichier csv comprenant les métriques pour les 5 folds de validation croisée de nnU-Net, génère des boxplots pour chaque métrique
- `metric_computation.py` : à partir d'un fichier csv comprenant les métriques pour les 5 folds de validation croisée de nnU-Net, calcule les résultats moyens sur les 5 folds

### 📁 Dossier `evaluation_results`

Ce dossier contient tous les métriques d'évaluation après les prédictions réalisées par nnU-Net, à savoir pour chaque dataset :
- un fichier .csv avec les métriques pour chaque prédiction pour chaque fold
- des boxplots pour chaque métrique

### 📁 Dossier `format_datasets`

Ce dossier contient tout le code nécessaire au formatage des données pour ensuite pouvoir les utiliser avec nnU-Net. 

- `format_MNI.py` : reformate le nom des images et labels du dataset OpenMS pour les adapter au format accepté par nnU-Net
- `format_MSLesSeg_FLAIR_only.py` : reformate le nom des images FLAIR et labels du dataset MSLesSeg pour les adapter au format accepté par nnU-Net
- `format_MSLesSeg.py` : reformate le nom des images FLAIR, T1 et T2 et labels du dataset MSLesSeg pour les adapter au format accepté par nnU-Net

### 📁 Dossier `scripts`

à compléter

### Autres fichiers
- `.gitignore`
- `inspect_volumes.ipynb` : inspecte et print les spécificités des volumes des différents datasets utilisés
- `read_preprocessed_data.py` : explore le contenu du dossier comprenant les données pré-traitées par nnU-Net
- `unet_seg AAT.ipynb` et `unet_seg.ipynb` : notebooks comprenant le code nécessaire pour entraîner un modèle U-Net 3D de zéro (n'a pas été utilisé pendant le projet)
- `utils.py` : contient les fonctions nécessaires au bon fonctionnement d'autres codes