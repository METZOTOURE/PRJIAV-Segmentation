import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
import numpy as np

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
    img: Tensor, 
    mask: Tensor = None, 
    pred: Tensor = None, 
    slice_idx: Tuple[int, int, int] = None, 
    title: str = ""
) -> None:
    """Plot a slice of a volume and the corresponding segmentations.
    Args:
        img: Tensor (1, D, H, W). Volume.
        mask: Tensor (1, D, H, W). Segmentation mask. Default to None.
        pred: Tensor (1, D, H, W). Segmentation prediction mask. Default to None.
        slice_idx: (d, h, w). Default to None (center slice chosen).
        title: Plot's title.
    """

    if pred is not None and mask is None:
        raise ValueError("'mask' cannot be None if 'pred' is provided.")

    img_np = img.squeeze().cpu().numpy()          # (D, H, W)
    mask_np = mask.squeeze().cpu().numpy() if mask is not None else None
    pred_np = pred.squeeze().cpu().numpy() if pred is not None else None

    D, H, W = img_np.shape

    # --- Slice indices ---
    if slice_idx is None:
        slice_idx = (D // 2, H // 2, W // 2)

    d, h, w = slice_idx

    # --- Extract the 3 views ---
    views = {
        "Axial":     (img_np[d],            mask_np[d] if mask_np is not None else None,
                                      pred_np[d] if pred_np is not None else None),
        "Coronal":   (img_np[:, h, :],      mask_np[:, h, :] if mask_np is not None else None,
                                      pred_np[:, h, :] if pred_np is not None else None),
        "Sagittal":  (img_np[:, :, w],      mask_np[:, :, w] if mask_np is not None else None,
                                      pred_np[:, :, w] if pred_np is not None else None),
    }

    # --- Define layout ---
    if pred is not None:
        rows, cols = 2, 3
    else:
        rows, cols = 1, 3

    plt.figure(figsize=(14, 6 if pred is None else 12))
    plt.suptitle(title, fontsize=16)

    # --- Plot loop ---
    for i, (name, (img_view, mask_view, pred_view)) in enumerate(views.items()):
        # First row → ground truth (mask)
        ax = plt.subplot(rows, cols, i + 1)
        oriented_img = np.rot90(img_view.T, k=1) if name != "Sagittal" else np.rot90(img_view, k=2)
        plt.imshow(oriented_img, cmap="gray")

        if mask_view is not None:
            oriented_mask = np.rot90(mask_view.T, k=1) if name != "Sagittal" else np.rot90(mask_view, k=2)
            plt.imshow(oriented_mask, cmap="Reds", alpha=0.4)

        plt.title(f"{name} (slice {slice_idx[i]})")
        plt.axis("off")

        # Second row → prediction
        if pred is not None:
            ax = plt.subplot(rows, cols, i + 1 + cols)
            plt.imshow(oriented_img, cmap="gray")

            if pred_view is not None:
                oriented_pred = np.rot90(pred_view.T, k=1) if name != "Sagittal" else np.rot90(pred_view, k=2)
                plt.imshow(oriented_pred, cmap="Reds", alpha=0.4)

            plt.title(f"{name} prediction")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def MSLesSeg_to_MSSEG(ref_path: str, img_path: str, out_path: str):

    ref = sitk.ReadImage(ref_path)
    img = sitk.ReadImage(img_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(img.GetPixelID())

    img_matched = resampler.Execute(img)


    sitk.WriteImage(img_matched, out_path)

    print("Corrected image saved in :", out_path)