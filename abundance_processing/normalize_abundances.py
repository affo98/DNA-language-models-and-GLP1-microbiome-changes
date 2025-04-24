import os
import sys

import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
from itertools import tee  # copies the iterator into n objects.


def log(msg: str, path: str = "log.txt"):
    """Handles writing string (msg) to a log file.

    Args:
        msg (str): The string to be logged
        path (str): Path to the log file. Defaults to "log.txt".
    """
    if os.path.exists("log.txt"):
        os.remove("log.txt")
    with open(path, "a") as f:
        f.write(msg + "\n")


def process_abundance(
    abundance_path: str, num_contigs: int, chunk_size: int = 100000
) -> tuple[np.ndarray, int]:
    """Loads all the abundances and extracts the per sample sum of abundances used in downstream normalization.

    Args:
        abundance_path (str): Path to all abundances.tsv
        num_contigs (int): Total number of contigs
        chunk_size (int, optional): Number of rows in the tsv file to be processed at a time. Defaults to 100000.

    Returns:
        tuple(np.ndarray, int): Sum of sample abundances with shape (1 x n_samples), n_samples
    """

    log("#" * 100)
    log(f"\t\tProcessing {abundance_path} containing n_contigs = {num_contigs}")
    log(f"\t\tIterating through the contig catalogue with a chunk size of {chunk_size}")
    ### 1. Compute sample_depths_sum (total abundance per sample) in a first pass ###
    sample_depths_sum = None
    chunk_iter, chunk_iter_copy = tee(
        pd.read_csv(abundance_path, sep="\t", index_col=0, chunksize=chunk_size)
    )

    log(f"Number of chunks: {len(list(chunk_iter_copy))}")

    samples = None

    # incremental load using pandas where the tsv file has the header format: contigs, sample_1.tsv, sample_2.tsv..., sample_N.tsv
    for chunk in tqdm(
        chunk_iter,
        total=num_contigs // chunk_size,
        desc="Retrieving sum of abundances per sample",
    ):

        samples = (
            samples if samples else list(chunk.columns)
        )  # extract list of sample ids.

        data = chunk.astype(np.float64)  # (chunck_size,n_samples) -> data = (10.000x96)

        abundance = data.to_numpy()

        sample_abund_sums = np.sum(
            abundance, axis=0
        )  # column (sample) wise sum abundances for the i'th chunck of <chunck_size> contig abundances -> sample_abund_sums = (1x96)
        if sample_depths_sum is None:
            sample_depths_sum = sample_abund_sums  # None -> [1x96] -> [1x96]
        else:
            sample_depths_sum += sample_abund_sums

    n_samples = len(sample_depths_sum)
    log(f"Number of samples: {n_samples} with sample ids {samples}")

    return sample_depths_sum, n_samples


def normalize_by_sample_abundance(
    abundance_path: str,
    sample_depths_sum: np.ndarray,
    num_contigs: int,
    n_samples: int,
    chunk_size: int = 100000,
) -> np.ndarray:
    """Loads all the abundances and extracts the per contig sum of abundances used in downstream normalization.

    Args:
        abundance_path (str): Path to all abundances.tsv
        sample_depths_sum (np.ndarray): Sum of per sample abundances (1 x n_samples)
        num_contigs (int): Total number of contigs
        n_samples (int): Total number of samples
        chunk_size (int, optional): Number of rows in the tsv file to be processed at a time. Defaults to 100000.

    Returns:
        np.ndarray: Sum of contig abundances of shape (n_contigs x 1)
    """
    total_contig_abundance = None
    chunk_iter = pd.read_csv(
        abundance_path, sep="\t", index_col=0, chunksize=chunk_size
    )

    for chunk in tqdm(
        chunk_iter,
        total=num_contigs // chunk_size,
        desc="Normalizing contig abundances by sample abundances",
    ):
        data = chunk.astype(np.float64)

        abundance = data.to_numpy()

        abundance *= (
            1_000_000 / sample_depths_sum
        )  # Normalize by sample abundances (RKPM)

        total_contig_abund_sums = np.sum(
            abundance, axis=1
        )  # row (contig) wise sums for the i'th chunck of <chunck_size> contig abundances -> contig_abund_sums = (100000x1)

        zero_total_abundance_in_chunck = (
            total_contig_abund_sums == 0
        )  # check for contigs with zero abundances

        if zero_total_abundance_in_chunck.sum() > 1:
            abundance[zero_total_abundance_in_chunck] = (
                1 / n_samples
            )  # set abundances to 1/n_samples

            total_contig_abund_sums[zero_total_abundance_in_chunck] = 1

        if total_contig_abundance is None:
            total_contig_abundance = total_contig_abund_sums
        else:
            total_contig_abundance = np.hstack(
                (total_contig_abundance, total_contig_abund_sums)
            )

    return total_contig_abundance


def normalize_by_global_contig_abundances(
    abundance_path: str,
    output_abundance_path: str,
    sample_depths_sum: np.ndarray,
    total_contig_abundance: np.ndarray,
    num_contigs: int,
    n_samples: int,
    chunk_size: int = 100000,
) -> None:
    """Normalize the abundances and write them to a new abundance.tsv file

    Args:
        abundance_path (str): Path to all abundances.tsv
        output_abundance_path (str): Path to normalized abundances.tsv
        sample_depths_sum (np.ndarray): Sum of per sample abundances (1 x n_samples)
        total_contig_abundance (np.ndarray): Sum of per contig abundances (n_contigs x 1)
        num_contigs (int): Total number of contigs
        n_samples (int): Total number of samples
        chunk_size (int, optional): Number of rows in the tsv file to be processed at a time.. Defaults to 100000.

    Raises:
        ValueError: If array shapes in division are mismatching.

    Returns:
        None:
    """
    first_chunk = True
    chunk_iter = pd.read_csv(
        abundance_path, sep="\t", index_col=0, chunksize=chunk_size
    )

    for i, chunk in tqdm(
        enumerate(chunk_iter),
        total=num_contigs // chunk_size,
        desc="Normalizing abundances by contig global abundances",
    ):
        start = i * chunk_size
        end = start + chunk_size

        data = chunk.astype(np.float64)

        abundance = data.to_numpy()

        abundance *= 1_000_000 / sample_depths_sum

        #########################################################################################################################
        ### Have to recalculate total contig abundance step to get inputed abundances before normalizing and writing to file. ###
        #########################################################################################################################

        total_contig_abund_sums = np.sum(
            abundance, axis=1
        )  # row (contig) wise sums for the i'th chunck of <chunck_size> contig abundances -> contig_abund_sums = (100000x1)

        zero_total_abundance_in_chunck = (
            total_contig_abund_sums == 0
        )  # check for contigs with zero abundances

        if zero_total_abundance_in_chunck.sum() > 1:

            abundance[zero_total_abundance_in_chunck] = (
                1 / n_samples
            )  # set abundances to 1/n_samples

        try:
            abundance /= total_contig_abundance[start:end].reshape(-1, 1)
        except ValueError as e:
            log(e)
            raise ValueError

        # Convert back to DataFrame for saving
        normalized_chunk = pd.DataFrame(
            abundance, columns=chunk.columns, index=chunk.index
        )
        # Append to file
        normalized_chunk.to_csv(
            output_abundance_path,
            sep="\t",
            mode="w" if first_chunk else "a",  # Write header only once
            header=first_chunk,
            index=False,
        )
        first_chunk = False
    log(
        "Finished normalizing the abundances and wrote them to path: normalized_abundances.tsv"
    )
    return None


def vamb_abundances(abundance, output_abundance_path):
    abundance = abundance.copy()
    sample_depths_sum = abundance.sum(axis=0)
    print("#" * 50)
    print(f"\t\tSAMPLE DEPTHS SUM VAMB:\n{sample_depths_sum}")
    print("#" * 50, "\n")
    # normalize OG abundances
    abundance *= 1_000_000 / sample_depths_sum
    # print(f"same sahpes as OG abundances 3x2:{abundance.shape}")

    total_abundance = abundance.sum(axis=1)
    # print(f"total abundance -> sum all contig abundances across samples -> 3x1: {total_abundance.shape}")

    # Normalize abundance to sum to 1
    n_samples = abundance.shape[1]
    # print(f"number of samples -> 2: {n_samples}")

    zero_total_abundance = total_abundance == 0
    # print(f"Masking array with any contigs with zero abundances, should be third entry {zero_total_abundance.shape}, {zero_total_abundance}")

    abundance[zero_total_abundance] = 1 / n_samples
    # print(f"Setting the zero abundances to be 1/N_samples: 0.5 {abundance[zero_total_abundance]}")

    nonzero_total_abundance = total_abundance.copy()
    # print(f"copying total abundances of all contigs with nonzero abundances: {nonzero_total_abundance.shape}")

    nonzero_total_abundance[zero_total_abundance] = 1.0
    # print(f"setting sum index of zero abundant entries = 1 as 1/n * n = 1: {nonzero_total_abundance[zero_total_abundance]}")
    abundance /= nonzero_total_abundance.reshape((-1, 1))
    # print(f"(3,0) numpy to actualy vector -> reshaped to : {nonzero_total_abundance.reshape((-1, 1)).shape}")
    # print(f"Normalizing abundances by contigs abundances across samples")

    # print(f"Final Normalized abundances:\n {abundance}")

    with open(output_abundance_path, "w") as file:
        header = (
            "contig\t" + "\t".join(str(i) for i in range(abundance.shape[1])) + "\n"
        )
        file.write(header)
        for i in range(abundance.shape[0]):
            line = f"{i}\t" + "\t".join(str(val) for val in abundance[i, :]) + "\n"
            file.write(line)
    return None


if __name__ == "__main__":

    path_to_abundances = sys.argv[0]  # "../../../../../Downloads/abundances.tsv"
    print(path_to_abundances)
    path_to_normalized_abundances = sys.argv[1]
    print(path_to_normalized_abundances)
    run_vamb = sys.argv[2]
    print(run_vamb, type(run_vamb))
    if run_vamb is "True":
        abundance_df = pd.read_csv(path_to_abundances)
        vamb_abundances(abundance_df.to_numpy(), path_to_normalized_abundances)
    else:
        chunk_size = 1000000

        stdout_lines = subprocess.run(
            ["wc", "-l", "<", f"{path_to_abundances}"], capture_output=True, text=True
        )
        num_contigs = int(stdout_lines.stdout.strip().split()[0]) - 1

        sample_depths_sum, n_samples = process_abundance(
            path_to_abundances, num_contigs, chunk_size
        )
        total_contig_abundance = normalize_by_sample_abundance(
            path_to_abundances, sample_depths_sum, n_samples, num_contigs, chunk_size
        )
        normalize_by_global_contig_abundances(
            path_to_abundances,
            path_to_normalized_abundances,
            sample_depths_sum,
            total_contig_abundance,
            num_contigs,
            n_samples,
            chunk_size,
        )
