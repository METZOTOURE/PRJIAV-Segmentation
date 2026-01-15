import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("./evaluation_results/MSLesSeg_full/full_evaluation_results.csv")
output_dir = Path("./evaluation_results/MSLesSeg_full")


def generate_and_save_boxplots(results_df, output_plot_dir):
    """Génère et sauvegarde des boîtes à moustaches pour chaque métrique, comparant les folds."""
    if results_df.empty:
        print("DataFrame vide, impossible de générer les graphiques.")
        return

    output_plot_dir.mkdir(exist_ok=True)
    print(f"\n--- Génération des Boîtes à Moustaches (sauvegardées dans '{output_plot_dir}') ---")
    metric_names = ['dice', 'hd95', 'assd', 'iou', 'lesion_f1', 'lesion_precision', 'lesion_recall']

    for metric in metric_names:
        plot_df = results_df[['fold', metric]].copy()
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        plot_df.dropna(inplace=True)
        if plot_df.empty:
            print(f"  Pas de données valides pour '{metric}', graphique ignoré.")
            continue

        plt.figure(figsize=(10, 7))
        sns.boxplot(x='fold', y=metric, data=plot_df, palette="viridis")
        sns.stripplot(x='fold', y=metric, data=plot_df, color='0.25', size=4, jitter=True, alpha=0.6)
        plt.title(f"Distribution de '{metric.upper()}' par Fold", fontsize=16)
        plt.xlabel("Fold", fontsize=12)
        plt.ylabel(f"Valeur de {metric.upper()}", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(output_plot_dir / f"boxplot_{metric}_MSLesSeg_IAMB2026.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Graphique pour '{metric}' sauvegardé.")


def main():
    results_df = pd.read_csv(csv_path)
    generate_and_save_boxplots(results_df, output_dir / "plots")

if __name__ == "__main__":
    main()