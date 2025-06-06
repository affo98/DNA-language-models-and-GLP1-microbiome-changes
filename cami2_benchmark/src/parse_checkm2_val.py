import os
from argparse import ArgumentParser
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import math


COMPLETENESS_BINS = [90, 80, 70, 60, 50]


CONTAMINATION_THRESHOLDS = list(range(5, 101, 5))  # [5, 10, 15, ..., 100]
WEIGHTS = [1 / (2**i) for i in range(20)]  # [1, 0,5, 0.25, 0.125...]


def plot_results(data, output_dir) -> None:
    plt.figure(figsize=(8, 3))

    df = pd.DataFrame(data)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    df = df.sort_index(axis=0, ascending=False)
    df = df.sort_index(axis=1)

    sns.heatmap(
        df.values,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        linewidths=0.5,
        xticklabels=df.columns,
        yticklabels=df.index,
    )

    plt.xlabel("K")
    plt.ylabel("Percentile")

    file_path = os.path.join(
        output_dir,
        f"heatmap.png",
    )
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return


def parse_quality_report(file_path, contamination) -> pd.DataFrame:
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < contamination]
    return df["Completeness"].values


def process_all_reports(
    results_dir, contamination, weight, weighted_count_dict: dict
) -> dict:
    """Walks through the results directory and structures data for heatmap plotting."""

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

        completeness_values = parse_quality_report(report_path, contamination)
        bin_counts = [np.sum(completeness_values >= b) for b in COMPLETENESS_BINS]

        if k_value not in weighted_count_dict:
            weighted_count_dict[k_value] = {}

        weighted_count_dict[k_value][p_value] = (
            int(bin_counts[-1]) * weight
        )  # n bins above 50

    return weighted_count_dict


def select_best_combination(data, n_val: int, log) -> dict:
    """Find the highest value and its corresponding (k, p) combination.
    If there is a tie, select the combination closest to N.
    """
    max_value = -1
    best_k, best_p = None, None
    candidates = []

    for k, p_values in data.items():
        for p, value in p_values.items():
            if value > max_value:
                max_value = value
                candidates = [(k, p)]
            elif value == max_value:
                candidates.append((k, p))

    # If there's a tie, select the closest (k, p) to N
    if len(candidates) > 1:
        best_k, best_p = min(
            candidates, key=lambda kp: abs(int(kp[0]) - math.sqrt(n_val))
        )
        log.append(
            f"Tie between {candidates}\nSelecting {best_k, best_p} using n_val {n_val}; sqrt: {math.sqrt(n_val)}"
        )
    else:
        best_k, best_p = candidates[0]

    result = {
        "best_k": int(best_k),
        "best_p": int(best_p),
        "max_weighted_sum": float(max_value),
    }
    return result


def main(args, log):

    weighted_count_dict = {}
    for contamination, weight in zip(CONTAMINATION_THRESHOLDS[:4], WEIGHTS[:4]):
        weighted_count_dict = process_all_reports(
            args.input_dir, contamination, weight, weighted_count_dict
        )
    print(weighted_count_dict)
    best_combination = select_best_combination(weighted_count_dict, args.n_val, log)
    best_combination["contamination_values"] = CONTAMINATION_THRESHOLDS[:4]

    # if no results are found with contamination < 20, then try all contamination values.
    if best_combination["max_weighted_sum"] <= 0:
        log.append(
            "No valid combination found in the first pass, trying with full list."
        )
        weighted_count_dict = {}

        for contamination, weight in zip(CONTAMINATION_THRESHOLDS, WEIGHTS):
            weighted_count_dict = process_all_reports(
                args.input_dir, contamination, weight, weighted_count_dict
            )
        print(weighted_count_dict)
        best_combination = select_best_combination(weighted_count_dict, args.n_val, log)
        best_combination["contamination_values"] = CONTAMINATION_THRESHOLDS

    # result found
    with open(os.path.join(args.output_dir, "heatmap_data.json"), "w") as f:
        json.dump(weighted_count_dict, f, indent=4)

    with open(os.path.join(args.output_dir, "best_combination.json"), "w") as f:
        json.dump(best_combination, f, indent=4)
    plot_results(weighted_count_dict, args.output_dir)


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
    parser.add_argument(
        "--n_val",
        "-n",
        type=int,
        help="Number of validation contigs",
    )
    parser.add_argument(
        "--log",
        "-l",
        help="Path to save logfile",
    )

    args = parser.parse_args()

    return args


class Logger:
    def __init__(self, log_path: str):
        """Initialize the Logger with a path to the log file.

        Args:
            log_path (str): The path to the log file.
        """
        self.log_path = log_path

    def append(self, message: str) -> None:
        """Print and append a log message to the log file.

        Args:
            message (str): The message to log.
        """
        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write(message + "\n")


if __name__ == "__main__":

    args = add_arguments()
    log = Logger(args.log)

    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    main(args, log)
