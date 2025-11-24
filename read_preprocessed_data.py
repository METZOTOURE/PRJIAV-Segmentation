import os
import pickle
import numpy as np
from pathlib import Path

# Si vous avez des fichiers .b2nd (nnU-Net v2.6+)
try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False
    print("⚠️ blosc2 non installé. Les fichiers .b2nd ne peuvent pas être lus.")
    print("   Installez avec: pip install blosc2")

# Configuration
preprocessed_dir = Path(os.getenv("nnUNet_preprocessed", "D:/nnUNet_preprocessed"))
dataset_name = "Dataset001_MSLesSeg"
configuration = "3d_fullres"  # ou "2d", "3d_lowres"

# Chemin vers les données
data_path = preprocessed_dir / dataset_name / f"nnUNetPlans_3d_fullres"

print("=" * 70)
print("LECTURE DES DONNEES PREPROCESSEES nnU-Net")
print("=" * 70)

if not data_path.exists():
    print(f"\n❌ Dossier non trouvé: {data_path}")
    print("   Exécutez d'abord: nnUNetv2_plan_and_preprocess -d 001")
    exit(1)

# Lister les fichiers disponibles
npy_files = sorted(data_path.glob("*.npy"))
pkl_files = sorted(data_path.glob("*.pkl"))
b2nd_files = sorted(data_path.glob("*.b2nd"))

print(f"\n✅ Dossier: {data_path}")
print(f"📊 Fichiers .npy trouvés: {len(npy_files)}")
print(f"📊 Fichiers .pkl trouvés: {len(pkl_files)}")
print(f"📊 Fichiers .b2nd trouvés: {len(b2nd_files)}")

# ========================================
# LECTURE D'UN FICHIER .npy
# ========================================
if npy_files:
    example_npy = npy_files[0]
    print(f"\n{'='*70}")
    print(f"📖 EXEMPLE 1: Lecture d'un fichier .npy")
    print(f"Fichier: {example_npy.name}")
    print(f"{'='*70}")
    
    # Charger le fichier
    data = np.load(example_npy)
    
    print(f"\n✅ Données chargées:")
    print(f"   - Shape: {data.shape}")
    print(f"   - Dtype: {data.dtype}")
    print(f"   - Taille mémoire: {data.nbytes / (1024**2):.2f} MB")
    print(f"   - Min: {data.min():.4f}")
    print(f"   - Max: {data.max():.4f}")
    print(f"   - Mean: {data.mean():.4f}")
    print(f"   - Std: {data.std():.4f}")
    
    # Interpréter la shape
    if len(data.shape) == 4:  # 3D avec modalités
        num_modalities, depth, height, width = data.shape
        print(f"\n📐 Structure des données:")
        print(f"   - Format: 3D multi-modalités")
        print(f"   - Nombre de modalités: {num_modalities}")
        print(f"   - Dimensions spatiales: {depth} x {height} x {width}")
        
        modality_names = ["FLAIR", "T1", "T2"]
        for i in range(num_modalities):
            if i < len(modality_names):
                mod_data = data[i]
                print(f"   - Modalité {i} ({modality_names[i]}):")
                print(f"       Range: [{mod_data.min():.4f}, {mod_data.max():.4f}]")
                print(f"       Mean: {mod_data.mean():.4f}")
    
    elif len(data.shape) == 3:  # 2D ou 3D sans modalités
        print(f"\n📐 Structure: {data.shape}")

# ========================================
# LECTURE D'UN FICHIER .pkl
# ========================================
if pkl_files:
    example_pkl = pkl_files[0]
    print(f"\n{'='*70}")
    print(f"📖 EXEMPLE 2: Lecture d'un fichier .pkl (métadonnées)")
    print(f"Fichier: {example_pkl.name}")
    print(f"{'='*70}")
    
    # Charger les métadonnées
    with open(example_pkl, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\n✅ Métadonnées chargées:")
    print(f"   Type: {type(metadata)}")
    
    if isinstance(metadata, dict):
        print(f"\n📋 Contenu du dictionnaire:")
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                print(f"   - {key}: array shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"   - {key}: {type(value).__name__} (length {len(value)})")
            else:
                print(f"   - {key}: {value}")
    
    # Clés typiques dans les métadonnées nnU-Net
    print(f"\n🔑 Informations importantes:")
    if 'spacing' in metadata:
        print(f"   - Spacing (résolution): {metadata['spacing']}")
    if 'origin' in metadata:
        print(f"   - Origin: {metadata['origin']}")
    if 'direction' in metadata:
        print(f"   - Direction: {metadata['direction']}")
    if 'crop_bbox' in metadata:
        print(f"   - Crop bbox: {metadata['crop_bbox']}")

# ========================================
# LECTURE D'UN FICHIER .b2nd (Blosc2)
# ========================================
if b2nd_files and HAS_BLOSC2:
    example_b2nd = b2nd_files[0]
    print(f"\n{'='*70}")
    print(f"📖 EXEMPLE 3: Lecture d'un fichier .b2nd (compressé)")
    print(f"Fichier: {example_b2nd.name}")
    print(f"{'='*70}")
    
    try:
        # Ouvrir avec blosc2
        b2_array = blosc2.open(example_b2nd)
        
        # Convertir en NumPy array
        data = b2_array[:]
        
        print(f"\n✅ Données décompressées:")
        print(f"   - Shape: {data.shape}")
        print(f"   - Dtype: {data.dtype}")
        print(f"   - Taille fichier: {example_b2nd.stat().st_size / (1024**2):.2f} MB")
        print(f"   - Taille décompressée: {data.nbytes / (1024**2):.2f} MB")
        print(f"   - Ratio compression: {data.nbytes / example_b2nd.stat().st_size:.2f}x")
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture: {e}")

elif b2nd_files and not HAS_BLOSC2:
    print(f"\n⚠️  {len(b2nd_files)} fichiers .b2nd trouvés mais blosc2 non installé")
    print("   Installez avec: pip install blosc2")

# ========================================
# STATISTIQUES GLOBALES
# ========================================
print(f"\n{'='*70}")
print(f"📊 STATISTIQUES SUR PLUSIEURS CAS")
print(f"{'='*70}")

files_to_analyze = npy_files[:5] if npy_files else b2nd_files[:5]

for i, file_path in enumerate(files_to_analyze, 1):
    if file_path.suffix == '.npy':
        data = np.load(file_path)
    elif file_path.suffix == '.b2nd' and HAS_BLOSC2:
        data = blosc2.open(file_path)[:]
    else:
        continue
    
    print(f"\n{i}. {file_path.stem}")
    print(f"   Shape: {data.shape}")
    print(f"   Range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"   Memory: {data.nbytes / (1024**2):.2f} MB")

# ========================================
# GUIDE D'UTILISATION
# ========================================
print(f"\n{'='*70}")
print("💡 GUIDE D'UTILISATION")
print(f"{'='*70}")
print("""
1. Lire un fichier .npy:
   import numpy as np
   data = np.load("P001_T1.npy")
   # data.shape = (3, 182, 218, 182)  # 3 modalités
   flair = data[0]  # Première modalité
   t1 = data[1]     # Deuxième modalité
   t2 = data[2]     # Troisième modalité

2. Lire les métadonnées .pkl:
   import pickle
   with open("P001_T1.pkl", "rb") as f:
       properties = pickle.load(f)
   spacing = properties['spacing']
   origin = properties['origin']

3. Lire un fichier .b2nd (compressé):
   import blosc2
   b2_array = blosc2.open("P001_T1.b2nd")
   data = b2_array[:]  # Convertir en NumPy array

4. Installation de blosc2:
   pip install blosc2

5. Normalisation:
   Les données sont déjà normalisées par nnU-Net selon les statistiques
   du dataset (z-score normalization ou autre selon la configuration).
""")

print("=" * 70)
