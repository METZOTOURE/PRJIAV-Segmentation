import SimpleITK as sitk
import os

def inspect_volume(filepath: str):
    """
    Lit un volume .nii.gz et affiche ses propriétés principales.
    """
    img = sitk.ReadImage(filepath)

    print(f"\n=== Volume : {os.path.basename(filepath)} ===")
    print("Dimensions       :", img.GetSize())         # (x, y, z)
    print("Spacing          :", img.GetSpacing())      # taille des voxels
    print("Origin           :", img.GetOrigin())       # origine du volume
    print("Direction        :", img.GetDirection())    # matrice de direction
    print("Pixel Type       :", img.GetPixelIDTypeAsString())
    print("Nombre de voxels :", img.GetWidth() * img.GetHeight() * img.GetDepth())