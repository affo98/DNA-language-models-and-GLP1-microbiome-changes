import os
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json

COMPLETENESS_BINS = [90, 80, 70, 60, 50]
CONTAMINATION_THRESHOLD = 10


def parse_quality_report(file_path) -> pd.DataFrame:
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < CONTAMINATION_THRESHOLD]
    return df["Completeness"].values


def process_all_reports(results_dir) -> dict:
    """Walks through the results directory and structures data for heatmap plotting."""
    data = {}

    for k_p_combination in os.listdir(results_dir):
        report_path = os.path.join(results_dir, k_p_combination, "quality_report.tsv")
        if not os.path.isfile(report_path):
            continue

        try:
            k_value, p_value = k_p_combination.split("_")
            k_value = k_value[1:]
            p_value = p_value[1:]
        except ValueError:
            continue

        completeness_values = parse_quality_report(report_path)
        bin_counts = [np.sum(completeness_values >= b) for b in COMPLETENESS_BINS]

        if k_value not in data:
            data[k_value] = {}

        data[k_value][p_value] = bin_counts[-1]  # Store highest completeness count

    return data


def select_best_combination(data, output_dir) -> dict:
    """Find the highest value and its corresponding (k, p) combination, then store it in a dictionary and save it to a file."""
    max_value = -1
    best_k, best_p = None, None

    for k, p_values in data.items():
        for p, value in p_values.items():
            if value > max_value:
                max_value = value
                best_k, best_p = k, p

    result = {"best_k": int(best_k), "best_p": int(best_p), "max_value": int(max_value)}

    with open(os.path.join(output_dir, "best_combination.json"), "w") as f:
        json.dump(result, f, indent=4)

    return result


def plot_results(data, output_dir) -> None:
    plt.figure(figsize=(8, 6))
    df = pd.DataFrame(data).T  # Transpose to align k values on y-axis

    sns.heatmap(df, annot=True, fmt="d", cmap="Oranges", linewidths=0.5)

    # Labels
    plt.xlabel("P values")
    plt.ylabel("K values")

    file_path = os.path.join(
        output_dir,
        f"heatmap.png",
    )
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return


def main(args):

    data = process_all_reports(args.input_dir)
    print(data)

    best_combination = select_best_combination(data, args.output_dir)
    print(best_combination)

    plot_results(data, args.output_dir)


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        help="Input directory with checkm2 results",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Output dir for best model",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = add_arguments()

    main(args)
