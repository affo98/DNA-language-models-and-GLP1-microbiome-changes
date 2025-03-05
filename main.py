import os
import csv
import argparse

import numpy as np

from src.clustering import KMediod
from src.get_embeddings import calculate_tnf

csv.field_size_limit(2**30)


def setup_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "CAMI2_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2"),
    }

    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        os.environ[var_name] = path

    return


def main():

    parser = argparse.ArgumentParser(
        description="Calculate or load tetranucleotide frequencies (TNFs)."
    )
    parser.add_argument(
        "--tnf",
        "-t",
        action="store_true",
        help="Set this flag if TNFs are already computed and saved in a .npz file.",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        help="dataset cami2 to include",
    )
    args = parser.parse_args()

    setup_paths()

    contigs_file = os.path.join(
        os.environ["CAMI2_OUTPUT_PATH"], f"{args.dataset_name}.csv"
    )

    with open(contigs_file) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
        contigs = [line[12] for line in data[1:]]

    if not args.tnf:
        embeddings = calculate_tnf(contigs)
        np.savez("tnf_embeddings.npz", embeddings=embeddings)

    else:
        d = np.load("tnf_embeddings.npz")
        embeddings = d["embeddings"]

    kmediod = KMediod(
        embeddings,
        min_similarity=0.007,
        min_bin_size=10,
        num_steps=3,
        max_iter=1000,
        normalized=False,
    )
    predictions = kmediod.fit()


if __name__ == "__main__":
    main()
