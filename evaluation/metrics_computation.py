import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import math
from pathlib import Path

load_dotenv()
csv_file = Path(os.getenv("OUTPUT_PATH")) / "full_evaluation_results.csv"
results_df = pd.read_csv(csv_file)

print("\n\n--- Moyennes par Fold ---")
numeric_cols = results_df.select_dtypes(include=np.number).columns.drop(['fold'], errors='ignore')
fold_means = results_df.groupby('fold')[numeric_cols].mean()
print(fold_means.to_string(float_format="%.4f"))

print("\n--- Moyennes Globales sur tous les cas ---")
global_means = results_df[numeric_cols].mean()
print(global_means.to_string(float_format="%.4f"))

all_results = fold_means.to_dict('list')
print("\n--- Métriques d'inférence (Moyenne ± STD) ---")
for i, (metric_name, values) in enumerate(all_results.items()):
        if values:
            mean_val = np.nanmean(values)
            std_val = np.std(values)
            sem_val = std_val/math.sqrt(5)
            print(f"  - {metric_name.capitalize():<20}: {mean_val:.4f} ± {std_val:.4f} ({mean_val-sem_val*1.96:.4f} -- {mean_val+sem_val*1.96:.4f})")
