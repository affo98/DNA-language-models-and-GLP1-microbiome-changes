import os
import json
from glob import glob
from collections import defaultdict
import re
import gzip

from Bio import SeqIO
from tqdm import tqdm
from datetime import datetime

import pandas as pd
import numpy as np


MODEL_RESULTS_DIR = os.path.join("cami2_benchmark", "model_results")
LOG_DIR = os.path.join("cami2_benchmark", "logs")
BASE_DIR = "cami2_benchmark"
PROCESSED_DATA_DIR = os.path.join("cami2_benchmark", "processed_data")
OUTPUT_DIR = os.path.join("cami2_benchmark", "model_results", "parsed_results")

COMPLETENESS_BINS = [90, 80, 70, 60, 50]

MODELS_NOT_INCLUDE = ["vamb", "taxvamb", "comebin"]


def parse_quality_report(file_path):
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < 5]
    return df["Completeness"].values


def process_all_reports(model_results_dir):
    """Walks through the model_results_dir to collect all models and datasets."""
    data = {}

    for dataset in os.listdir(model_results_dir):
        dataset_path = os.path.join(model_results_dir, dataset, "checkm2")
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


def parse_knn_histograms(model_results_dir):
    """
    Collects similarity histogram JSON files across datasets and models, and includes
    'k' and 'p' as fields in the data instead of using them as keys.

    Args:
        model_results_dir (str): The base directory containing datasets.

    Returns:
        dict: Nested dict structure:
              histograms[dataset][model] = list of dicts with keys: k, p, data...
    """
    histograms = defaultdict(lambda: defaultdict(list))

    for dataset_dir in glob(os.path.join(model_results_dir, "*")):
        dataset_name = os.path.basename(dataset_dir)

        for model_dir in glob(os.path.join(dataset_dir, "*_output")):
            model_name = os.path.basename(model_dir).split("_output")[0]
            if model_name in MODELS_NOT_INCLUDE:
                continue

            hist_file = glob(
                os.path.join(model_dir, "test", "k*_p*_similarity_histogram.json")
            )[0]
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


def parse_contig_lengths(processed_data_dir):
    """Reads in contigs from each dataset and saves their length in a list."""

    contigs_lengths = defaultdict(list)
    for dataset_dir in tqdm(
        glob(os.path.join(processed_data_dir, "*")), desc="Parsing contig lengths"
    ):
        dataset_name = os.path.basename(dataset_dir)

        contigs_file = glob(os.path.join(dataset_dir, "catalogue.fna.gz"))[0]

        lengths = []

        try:
            with gzip.open(contigs_file, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    lengths.append(len(record.seq))

        except Exception:
            with open(contigs_file, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    lengths.append(len(record.seq))

        contigs_lengths[dataset_name] = lengths

    return contigs_lengths


def parse_runtimes(base_dir: str) -> pd.DataFrame:
    results = []

    models_binning = [
        "tnf",
        "tnfkernel",
        "dna2vec",
        "dnabert2",
        "dnabert2random",
        "dnaberts",
    ]
    binning_pattern = re.compile(r"(\w+?)_(test|val)_binning\.log")

    for fname in os.listdir(base_dir):
        match = binning_pattern.match(fname)
        print(match)
        if match:
            model, dataset = match.groups()
            if model in models_binning:
                with open(os.path.join(base_dir, fname), "r") as f:
                    content = f.read()
                    runtime_match = re.search(
                        r"Binning of .* ran in ([\d.]+) Seconds", content
                    )
                    if runtime_match:
                        runtime_min = float(runtime_match.group(1)) / 60
                        results.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "runtime_minutes": runtime_min,
                            }
                        )
    print(results)

    # # Handle vamb and taxvamb
    # for model in ['vamb', 'taxvamb']:
    #     log_path = os.path.join(base_dir, f'{model}_output/log.txt')
    #     if os.path.isfile(log_path):
    #         with open(log_path, 'r') as f:
    #             content = f.read()
    #             match = re.search(r'Completed Vamb in ([\d.]+) seconds', content)
    #             if match:
    #                 runtime_min = float(match.group(1)) / 60
    #                 dataset = 'unknown'  # modify as needed
    #                 results.append({'dataset': dataset, 'model': model, 'runtime_minutes': runtime_min})

    # # Handle comebin
    # start_path = os.path.join(base_dir, 'comebin_output/data_sugmentation/comebin.log')
    # end_path = os.path.join(base_dir, 'comebin_output/comebin_res/comebin.log')

    # def parse_timestamp(line):
    #     return datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f')

    # try:
    #     with open(start_path, 'r') as f:
    #         for line in f:
    #             if 'generate_aug_data' in line:
    #                 start_time = parse_timestamp(line)
    #                 break
    #     with open(end_path, 'r') as f:
    #         lines = f.readlines()
    #         for line in reversed(lines):
    #             if 'Reading Map:' in line:
    #                 end_time = parse_timestamp(line)
    #                 break
    #     runtime_min = (end_time - start_time).total_seconds() / 60
    #     results.append({'dataset': 'unknown', 'model': 'comebin', 'runtime_minutes': runtime_min})
    # except Exception as e:
    #     print(f"Error processing comebin logs: {e}")

    # return pd.DataFrame(results)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # cami2_results = process_all_reports(MODEL_RESULTS_DIR)
    # with open(os.path.join(OUTPUT_DIR, "parsed_checkm2_results.json"), "w") as f:
    #     json.dump(cami2_results, f, indent=4)

    # knn_histograms = parse_knn_histograms(MODEL_RESULTS_DIR)
    # with open(os.path.join(OUTPUT_DIR, "parsed_knn_histograms.json"), "w") as f:
    #     json.dump(knn_histograms, f, indent=4)

    # contig_lengths = parse_contig_lengths(PROCESSED_DATA_DIR)
    # with open(os.path.join(OUTPUT_DIR, "parsed_contig_lengths.json"), "w") as f:
    #     json.dump(contig_lengths, f, indent=4)

    runtimes = parse_runtimes(BASE_DIR)
