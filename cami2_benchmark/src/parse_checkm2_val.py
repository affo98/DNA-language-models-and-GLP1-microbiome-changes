import os
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COMPLETENESS_BINS = [90, 80, 70, 60, 50]
CONTAMINATION_THRESHOLD = 10


def parse_quality_report(file_path) -> pd.DataFrame:
    """Parses a CheckM2 quality report and extracts completeness & contamination."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Contamination"] < CONTAMINATION_THRESHOLD]
    return df["Completeness"].values


def process_all_reports(results_dir) -> dict:
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


# def plot_results(data)->None:
#             plt.figure(figsize=(8, 6))

#         if knn:
#             plt.axvline(
#                 self.knn_threshold,
#                 color="g",
#                 linestyle="--",
#                 label=f"KNN Threshold: {self.knn_threshold} (k={self.knn_k}, p={self.knn_p})",
#             )

#         plt.plot(
#             self.pairsim_vector,
#             self.bin_vector,
#             color="skyblue",
#             linestyle="-",
#             linewidth=2,
#         )

#         plt.xlabel("Similarity Bins")
#         plt.ylabel("Frequency")
#         plt.title(f"Similarity Histogram {self.model_name}")

#         plt.legend()

#         file_path = os.path.join(
#             self.save_path,
#             f"k{self.knn_k}_p{self.knn_p}_similarity_histogram.png",
#         )
#         plt.tight_layout()
#         plt.savefig(file_path)
#         plt.close()
#         self.log.append(f"Plot saved at: {file_path}")

#         return


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
