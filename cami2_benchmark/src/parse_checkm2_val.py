import os
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COMPLETENESS_BINS = [90, 80, 70, 60, 50]
CONTAMINATION_THRESHOLD = 5


def parse_quality_report(file_path):
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < 5]
    return df["Completeness"].values


def process_all_reports(results_dir):
    """Walks through the RESULTS_DIR to collect all results."""
    data = {}

    for k_p_combination in os.listdir(results_dir):
        report_path = os.path.join(results_dir, k_p_combination, "quality_report.tsv")
        if not os.path.isfile(report_path):
            continue

        completeness_values = parse_quality_report(report_path)
        bin_counts = [np.sum(completeness_values >= b) for b in COMPLETENESS_BINS]

        data[k_p_combination] = bin_counts

    return data


def main(args):

    data = process_all_reports(args.input_dir)
    print(data)


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
