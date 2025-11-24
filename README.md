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

```powershell
# Installation minimale (pour formatage et vérification)
pip install nibabel

# Installation complète (pour entraînement)
pip install -r requirements.txt
```

## 📁 Structure du Dataset
