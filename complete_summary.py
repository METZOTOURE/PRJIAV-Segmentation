import json
from dotenv import load_dotenv
import os

def compute_extra_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def complete_summary_json(summary_path, output_path):
    with open(summary_path) as f:
        data = json.load(f)

    fg = data["foreground_mean"]
    precision, recall, f1 = compute_extra_metrics(fg["TP"], fg["FP"], fg["FN"])

    fg["precision"] = precision
    fg["recall"] = recall
    fg["f1"] = f1

    for cls, metrics in data["mean"].items():
        precision, recall, f1 = compute_extra_metrics(metrics["TP"], metrics["FP"], metrics["FN"])
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    for case in data["metric_per_case"]:
        for cls, metrics in case["metrics"].items():
            precision, recall, f1 = compute_extra_metrics(metrics["TP"], metrics["FP"], metrics["FN"])
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    summary_path = os.getenv("SUMMARY_FILE")
    output_path = os.path.join(os.path.dirname(summary_path), "summary_completed.json")

    complete_summary_json(summary_path, output_path)
    print(f"summary.json completed\n    saved in {output_path}")