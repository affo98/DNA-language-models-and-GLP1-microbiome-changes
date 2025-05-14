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


MODEL_RESULTS_DIR = os.path.join("phenotype_mil", "binning_results")
LOG_DIR = os.path.join("phenotype_mil", "logs")
BASE_DIR = os.path.join("phenotype_mil")
PROCESSED_DATA_DIR = os.path.join("ngs_pipeline")
OUTPUT_DIR = os.path.join("phenotype_mil", "binning_results", "parsed_results")


COMPLETENESS_BINS = [90, 80, 70, 60, 50]
CONTAMINATION = 5

DATASETS = ["T2D-EW", "UNSEEN_BIO"]

OTHER_MODELS = [
    "dnaberts",
    # "dnaberth_400k",
    # "dnaberth_2mv1",
    # "dnaberth_2mv2",
]


def parse_quality_report(file_path):
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < CONTAMINATION]
    return df["Completeness"].values


def get_bin_counts(model_results_dir):
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

    with open(os.path.join(OUTPUT_DIR, "parsed_bincounts_results.json"), "w") as f:
        json.dump(data, f, indent=4)

    return data


def get_cluster_results(model_results_dir):
    """Walks through the model_results_dir to collect all models and datasets."""

    for dataset in os.listdir(model_results_dir):
        dataset_path = os.path.join(model_results_dir, dataset, "checkm2")
        if not os.path.isdir(dataset_path):
            continue

        for model in os.listdir(dataset_path):
            report_path = os.path.join(dataset_path, model, "quality_report.tsv")
            if not os.path.isfile(report_path):
                continue

            df = pd.read_csv(report_path, sep="\t")
            df = df[["Name", "Completeness", "Contamination"]]
            df.to_csv(os.path.join(OUTPUT_DIR, f"{model}_{dataset}"))

    return


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
        if dataset_name not in DATASETS:
            continue
        print(dataset_name)

        for model_dir in glob(os.path.join(dataset_dir, "*_output")):
            model_name = os.path.basename(model_dir).split("_output")[0]
            print(model_name)

            hist_file = glob(
                os.path.join(model_dir, "test", "k*_p*_similarity_histogram.json")
            )
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

    with open(os.path.join(OUTPUT_DIR, "parsed_knn_histograms.json"), "w") as f:
        json.dump(histograms, f, indent=4)

    return histograms


def parse_contig_lengths(processed_data_dir):
    """Reads in contigs from each dataset and saves their length in a list."""

    contigs_summary = []
    contigs_lengths = defaultdict(list)
    for dataset_dir in tqdm(
        glob(os.path.join(processed_data_dir, "*")), desc="Parsing contig lengths"
    ):
        dataset_name = os.path.basename(dataset_dir)
        if dataset_name not in DATASETS:
            continue
        contigs_file = glob(
            os.path.join(dataset_dir, "global_contig_catalogue.fna.gz")
        )[0]

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

        lengths_array = np.array(lengths)

        if len(lengths_array) > 0:
            contigs_summary.append(
                {
                    "dataset": dataset_name,
                    "num_contigs": len(lengths_array),
                    "total_length": int(lengths_array.sum()),
                    "mean_length": float(lengths_array.mean()),
                    "median_length": float(np.median(lengths_array)),
                    "min_length": int(lengths_array.min()),
                    "max_length": int(lengths_array.max()),
                    "25_percentile": float(np.percentile(lengths_array, 25)),
                    "75_percentile": float(np.percentile(lengths_array, 75)),
                }
            )

    with open(os.path.join(OUTPUT_DIR, "parsed_contig_lengths_mil.json"), "w") as f:
        json.dump(contigs_lengths, f, indent=4)

    contigs_summary_df = pd.DataFrame(contigs_summary)
    contigs_summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "parsed_contig_lengths_mil.csv"), index=False
    )

    return contigs_summary, contigs_lengths


def parse_runtimes(base_dir: str) -> pd.DataFrame:

    # ----- other models -----
    other_model_result = defaultdict(lambda: {"runtime_minutes": 0})
    binning_pattern = re.compile(r"(\w+?)_(test|val)_binning\.log")
    for root, _, files in os.walk(base_dir):
        for fname in files:
            match = binning_pattern.match(fname)
            if match:
                model, _ = match.groups()
                if model in OTHER_MODELS:
                    dataset = next((ds for ds in DATASETS if ds in root), None)
                    if dataset:
                        full_path = os.path.join(root, fname)
                        with open(full_path, "r") as f:
                            content = f.read()
                            runtime_match = re.search(
                                r"Binning of .* ran in ([\d.]+) Seconds", content
                            )
                            if runtime_match:
                                runtime_min = float(runtime_match.group(1)) / 60
                                key = (dataset, model)
                                other_model_result[key][
                                    "runtime_minutes"
                                ] += runtime_min

    other_models_df = pd.DataFrame(
        [
            {"dataset": k[0], "model": k[1], "runtime_minutes": v["runtime_minutes"]}
            for k, v in other_model_result.items()
        ]
    )

    other_models_df.to_csv(os.path.join(OUTPUT_DIR, "runtimes.csv"), index=False)
    return other_models_df


def parse_bin_postprocess(logdir: str) -> pd.DataFrame:
    results = []
    for dataset in DATASETS:
        for model in OTHER_MODELS:
            log_path = os.path.join(
                logdir, dataset, f"{model}_test_bin_postprocessing.log"
            )

            if os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    content = f.read()
                    pre_clusters = re.search(
                        r"Total clusters before filtering: (\d+)", content
                    )
                    post_clusters = re.search(
                        r"Total clusters after filtering: (\d+)", content
                    )
                    removed_clusters = re.search(
                        r"Number of clusters removed: (\d+)", content
                    )

                    results.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "clusters_before": (
                                int(pre_clusters.group(1)) if pre_clusters else None
                            ),
                            "clusters_after": (
                                int(post_clusters.group(1)) if post_clusters else None
                            ),
                            "clusters_removed": (
                                int(removed_clusters.group(1))
                                if removed_clusters
                                else None
                            ),
                        }
                    )
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "bin_postprocess.csv"), index=False)


def parse_nvaltest(model_results_dir: str) -> pd.DataFrame:
    results = []
    for dataset in DATASETS:
        for model in OTHER_MODELS:
            filepath = os.path.join(
                model_results_dir,
                dataset,
                f"{model}_output",
                "test",
                "n_total_val_test.json",
            )
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "n_total": data.get("n_total"),
                        "n_val": data.get("n_val"),
                        "n_test": data.get("n_test"),
                    }
                )

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "n_val_test.csv"), index=False)
    return pd.DataFrame(results)


def parse_heatmaps(model_results_dir: str) -> pd.DataFrame:
    results = []
    for dataset in DATASETS:
        for model in OTHER_MODELS:
            filepath = os.path.join(
                model_results_dir,
                dataset,
                f"{model}_output",
                "checkm2_validation_results",
                "heatmap_data.json",
            )
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "heatmap": data,
                    }
                )

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "heatmaps.csv"), index=False)
    return pd.DataFrame(results)


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bin_counts = get_bin_counts(MODEL_RESULTS_DIR)

    get_cluster_results(MODEL_RESULTS_DIR)

    knn_histograms = parse_knn_histograms(MODEL_RESULTS_DIR)

    runtimes = parse_runtimes(BASE_DIR)

    bin_postprocess = parse_bin_postprocess(LOG_DIR)

    n_valtest = parse_nvaltest(MODEL_RESULTS_DIR)

    heatmaps = parse_heatmaps(MODEL_RESULTS_DIR)

    # contig_summary, contig_lengths = parse_contig_lengths(PROCESSED_DATA_DIR)
