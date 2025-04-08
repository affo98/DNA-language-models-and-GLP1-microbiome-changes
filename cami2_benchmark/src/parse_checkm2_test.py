import os
import json

import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join("cami2_benchmark", "model_results")

COMPLETENESS_BINS = [90, 80, 70, 60, 50]


def parse_quality_report(file_path):
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < 5]
    return df["Completeness"].values


def process_all_reports(results_dir):
    """Walks through the RESULTS_DIR to collect all models and datasets."""
    data = {}

    for dataset in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset, "checkm2")
        if not os.path.isdir(dataset_path):
            continue

        for model in os.listdir(dataset_path):
            report_path = os.path.join(dataset_path, model, "quality_report.tsv")
            if not os.path.isfile(report_path):
                continue

            completeness_values = parse_quality_report(report_path)
            bin_counts = [
                int(np.sum(completeness_values >= b)) for b in COMPLETENESS_BINS
            ]

            if dataset not in data:
                data[dataset] = {}
            data[dataset][model] = bin_counts

    return data


if __name__ == "__main__":
    results = process_all_reports(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, "parsed_checkm2_results.json"), "w") as f:
        json.dump(results, f, indent=4)
