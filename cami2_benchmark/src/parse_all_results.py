import os
import json
import glob
from collections import defaultdict
import re


import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join("cami2_benchmark", "model_results")
OUTPUT_DIR = os.path.join("cami2_benchmark", "model_results", "parsed_results")

COMPLETENESS_BINS = [90, 80, 70, 60, 50]

MODELS_NOT_INCLUDE = ["vamb", "taxvamb", "comebin"]


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


def parse_knn_histograms(results_dir):
    """
    Collects similarity histogram JSON files across datasets and models, and includes
    'k' and 'p' as fields in the data instead of using them as keys.

    Args:
        results_dir (str): The base directory containing datasets.

    Returns:
        dict: Nested dict structure:
              histograms[dataset][model] = list of dicts with keys: k, p, data...
    """
    histograms = defaultdict(lambda: defaultdict(list))

    for dataset_dir in glob.glob(os.path.join(results_dir, "*")):
        dataset_name = os.path.basename(dataset_dir)
        print(dataset_name)

        for model_dir in glob.glob(os.path.join(dataset_dir, "*_output")):
            model_name = os.path.basename(model_dir)
            if model_name in MODELS_NOT_INCLUDE:
                continue
            print(model_name)

            hist_file = glob.glob(
                os.path.join(model_dir, "test", "k*_p*_similarity_histogram.json")
            )[0]
            print(hist_file)
            filename = os.path.basename(hist_file)

            # Extract k and p from filename using regex
            match = re.match(r"k(\d+)_p([\d.]+)_similarity_histogram\.json", filename)
            if not match:
                ValueError(f"Filename {filename} does not match expected pattern.")

            k = int(match.group(1))
            p = int(match.group(2))

            with open(hist_file, "r") as f:
                data = json.load(f)

            data["k"] = k
            data["p"] = p
            histograms[dataset_name][model_name] = data

    return histograms


def parse_contig_lengths(results_dir):
    pass


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cami2_results = process_all_reports(RESULTS_DIR)
    with open(os.path.join(OUTPUT_DIR, "parsed_checkm2_results.json"), "w") as f:
        json.dump(cami2_results, f, indent=4)

    knn_histograms = parse_knn_histograms(RESULTS_DIR)
    with open(os.path.join(OUTPUT_DIR, "parsed_knn_histograms.json"), "w") as f:
        json.dump(knn_histograms, f, indent=4)
