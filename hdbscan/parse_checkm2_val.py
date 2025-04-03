from argparse import ArgumentParser
import json

import pandas as pd
import numpy as np


def parse_quality_report(file_path: str) -> None:
    """Parses a CheckM2 quality report and writes completeness & contamination results to a file.

    Args:
        file_path (str): path to checkm2 report
    """

    contamination_thresholds = [5, 10, 15]
    completeness_bins = [90, 80, 70, 60, 50]

    df = pd.read_csv(file_path, sep="\t")

    results_dict = {"contamination_thresholds": {}}

    for contamination_threshold in contamination_thresholds:

        df = df[df["Contamination"] < contamination_threshold]

        completeness_values = df["Completeness"].values

        bin_counts = [np.sum(completeness_values >= b) for b in completeness_bins]
        bin_counts = [int(i) for i in bin_counts]
        results_dict["contamination_thresholds"][contamination_threshold] = bin_counts

    with open("hdbscan_results.txt", "w") as json_file:
        json.dump(results_dict, json_file)

    return


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_report",
        "-i",
        help="Path to checkm2 tsv report",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = add_arguments()

    parse_quality_report(args.input_report)
