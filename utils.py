import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple

def subdirs(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> list[str]:
    """Get a list of subdirectories in a folder.

    Args:
        folder: The path to the folder.
        join: Whether to join the folder path with subdirectory names. Defaults to True.
        prefix: Filter subdirectories by prefix. Defaults to None.
        suffix: Filter subdirectories by suffix. Defaults to None.
        sort: Whether to sort the resulting list. Defaults to True.

    Returns:
        A list of subdirectory names in the given folder.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def datalist(data_dir: str) -> list[dict]:
    """List all data for segmentation as dict {volume, mask}.

    Args:
        data_dir: Path to directory "MSLesSeg_Dataset/train" or "MSLesSeg_Dataset/test".

    Returns:
        A list of dict containing path to volume and its mask.
    
    """
    keys = subdirs(data_dir, prefix="P", join=False)
    sub_keys = []
    for k in keys:
        sub_key = subdirs(os.path.join(data_dir, k), prefix="T", join=False)
        sub_keys.append(sub_key)

    views_sequences = ['FLAIR']

    datalist = []
    for i, key in enumerate(keys):
        if sub_keys[i]:
            for sub_key in sub_keys[i]:
                for view in views_sequences:
                    datalist.append({
                        "volume": str(data_dir +"/"+ key +"/"+ sub_key +"/"+ f"{key}_{sub_key}_{view}.nii.gz"),
                        "mask": str(data_dir +"/"+ key +"/"+ sub_key +"/"+ f"{key}_{sub_key}_MASK.nii.gz"),
                    })
        else:
            for view in views_sequences:
                datalist.append({
                    "volume": str(data_dir +"/"+ key +"/"+ f"{key}_{view}.nii.gz"),
                    "mask": str(data_dir +"/"+ key +"/"+ f"{key}_MASK.nii.gz"),
                })
    
    return datalist


def inspect_volume(filepath: str) -> None:
    """Read volume .nii.gz and print main properties.

    Args:
        filepath: Path to file.
    """
    img = sitk.ReadImage(filepath)

    print(f"\n=== Volume : {os.path.basename(filepath)} ===")
    print("Dimensions       :", img.GetSize())         # (x, y, z)
    print("Spacing          :", img.GetSpacing())      # taille des voxels
    print("Origin           :", img.GetOrigin())       # origine du volume
    print("Direction        :", img.GetDirection())    # matrice de direction
    print("Pixel Type       :", img.GetPixelIDTypeAsString())
    print("Number of voxels :", img.GetWidth() * img.GetHeight() * img.GetDepth())


def show_3d_views(
    img: Tensor, mask: Tensor=None, pred: Tensor=None, slice_idx: Tuple[int, int, int]=None, title=""
) -> None:
    """Plot a slice of a volume and the corresponding segmentations.
    Args:
        img: Tensor (1, D, H, W). Volume.
        mask: Tensor (1, D, H, W). Segmentation mask. Default to None.
        slice_idx: (d, h, w). Default to None (center slice chosen).
        title: Plot's title.
    """
    if pred is not None and mask is None:
        raise ValueError("'mask' can not be set to None if a 'pred' was given.")

    # Convertir torch → numpy et enlever le channel
    img_np = img.squeeze().cpu().numpy()     # (D, H, W)
    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy()
    if pred is not None:
        pred_np = pred.squeeze().cpu().numpy()

    D, H, W = img_np.shape

    # Définir les indices de coupe
    if slice_idx is None:
        slice_idx = (D // 2, H // 2, W // 2)

    d, h, w = slice_idx

    # Préparer les trois vues :
    axial_img     = img_np[d]          # (H, W)
    sagittal_img  = img_np[:, :, w]    # (D, H)
    coronal_img   = img_np[:, h, :]    # (D, W)

    # Même pour le masque :
    if mask is not None:
        axial_mask     = mask_np[d]
        coronal_mask   = mask_np[:, h, :]
        sagittal_mask  = mask_np[:, :, w]
    else:
        axial_mask = sagittal_mask = coronal_mask = None

    # Même pour la prediction :
    if pred is not None:
        axial_pred     = pred_np[d]
        coronal_pred   = pred_np[:, h, :]
        sagittal_pred  = pred_np[:, :, w]
    else:
        axial_pred = sagittal_pred = coronal_pred = None


    if pred is not None:
        plt.figure(figsize=(14, 12))
        plt.suptitle(title, fontsize=16)

        # AXIAL
        plt.subplot(2, 3, 1)
        plt.imshow(axial_img, cmap="gray")
        plt.imshow(axial_mask, alpha=0.4, cmap="Reds")
        plt.title(f"Axial (slice {d}) with true mask")
        plt.axis("off")

        # CORONAL
        plt.subplot(2, 3, 2)
        plt.imshow(coronal_img.T, cmap="gray", origin="lower")
        plt.imshow(coronal_mask.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Coronal (slice {h}) with true mask")
        plt.axis("off")

        # SAGITTAL
        plt.subplot(2, 3, 3)
        plt.imshow(sagittal_img.T, cmap="gray", origin="lower")
        plt.imshow(sagittal_mask.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Sagittal (slice {w}) with true mask")
        plt.axis("off")

        # AXIAL
        plt.subplot(2, 3, 4)
        plt.imshow(axial_img, cmap="gray")
        plt.imshow(axial_pred, alpha=0.4, cmap="Reds")
        plt.title(f"Axial (slice {d}) with predicted mask")
        plt.axis("off")

        # CORONAL
        plt.subplot(2, 3, 5)
        plt.imshow(coronal_img.T, cmap="gray", origin="lower")
        plt.imshow(coronal_pred.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Coronal (slice {h}) with predicted mask")
        plt.axis("off")

        # SAGITTAL
        plt.subplot(2, 3, 6)
        plt.imshow(sagittal_img.T, cmap="gray", origin="lower")
        plt.imshow(sagittal_pred.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Sagittal (slice {w}) with predicted mask")
        plt.axis("off")
    else:
        plt.figure(figsize=(14, 6))
        plt.suptitle(title, fontsize=16)

        # AXIAL
        plt.subplot(1, 3, 1)
        plt.imshow(axial_img, cmap="gray")
        if mask is not None:
            plt.imshow(axial_mask, alpha=0.4, cmap="Reds")
        plt.title(f"Axial (slice {d})")
        plt.axis("off")

        # CORONAL
        plt.subplot(1, 3, 2)
        plt.imshow(coronal_img.T, cmap="gray", origin="lower")
        if mask is not None:
            plt.imshow(coronal_mask.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Coronal (slice {h})")
        plt.axis("off")

        # SAGITTAL
        plt.subplot(1, 3, 3)
        plt.imshow(sagittal_img.T, cmap="gray", origin="lower")
        if mask is not None:
            plt.imshow(sagittal_mask.T, alpha=0.4, cmap="Reds", origin="lower")
        plt.title(f"Sagittal (slice {w})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()