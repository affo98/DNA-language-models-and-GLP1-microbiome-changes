import csv
import os

import torch
import numpy as np
import pandas as pd


CLUSTERS_HEADER = "clustername\tcontigname"
SAMPLE_HEADER = "sample_id\tlabel"


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


def read_cluster_abundances(
    abundance_path: str, sample_ids: np.array, log: Logger
) -> pd.DataFrame:

    cluster_abundances = pd.read_csv(
        os.path.join(abundance_path, "cluster_abundances.tsv"), sep="\t"
    )
    cluster_abundances.columns = cluster_abundances.columns.str.replace(".tsv", "")

    cluster_abundances.set_index("cluster_id", inplace=True)
    cluster_abundances = cluster_abundances.T
    cluster_abundances.reset_index(inplace=True)
    cluster_abundances.rename(columns={"index": "sample"}, inplace=True)

    cluster_abundances["sample"] = cluster_abundances["sample"].astype(str)
    cluster_abundances["sample"] = pd.Categorical(
        cluster_abundances["sample"], categories=sample_ids.astype(str), ordered=True
    )
    cluster_abundances = cluster_abundances.sort_values("sample")
    cluster_abundances.reset_index(drop=True, inplace=True)
    print(cluster_abundances)
    log.append(
        f"Abundances shape {cluster_abundances.shape}, Sample IDs: {cluster_abundances['sample'].values.tolist()}"
    )

    return cluster_abundances


def read_clusters(clusters_path: str, log: Logger) -> dict[str, set[str]]:
    """Read cluster file into a dict[clusterid, set[contigids]]."""
    with open(clusters_path, "r") as filehandle:
        clusters_dict = {}
        lines = iter(filehandle)

        header = next(lines)
        if header.rstrip(" \n") != CLUSTERS_HEADER:
            raise ValueError(
                f'Expected cluster TSV file to start with header: "{CLUSTERS_HEADER}"'
            )

        for line in lines:
            stripped = line.strip()

            if not stripped or stripped[0] == "#":
                continue

            clustername, contigname = stripped.split("\t")
            if clustername not in clusters_dict:
                clusters_dict[clustername] = set()
            clusters_dict[clustername].add(contigname)

    log.append(f"Read {len(clusters_dict)} clusters from {clusters_path}")
    return clusters_dict


def read_sample_labels(
    sample_labels_path: str, log: Logger, split_train_test: bool
) -> tuple[np.array, np.array]:
    sample_ids = []
    labels = []

    with open(sample_labels_path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        first_row = next(reader, None)  # Skip the header
        if (
            first_row[0] == SAMPLE_HEADER.split("\t")[0]
            and first_row[1] == SAMPLE_HEADER.split("\t")[1]
        ):
            log.append(f"Header Skipped in sample labels file: {first_row}")
        else:
            sample_ids.append(str(first_row[0].replace("SAMEA", "")))
            labels.append(int(first_row[1]))
        for row in reader:
            sample_ids.append(str(row[0].replace("SAMEA", "")))
            labels.append(int(row[1]))

    if not split_train_test:
        assert len(sample_ids) == len(labels)
        log.append(f"Sample Ids: {sample_ids} \n Labels: {labels}")
        return np.array(sample_ids), np.array(labels)

    sample_ids_train, sample_ids_test, labels_train, labels_test = (
        split_samples_traintest(sample_ids, labels, proportion=0.3)
    )
    return (
        np.array(sample_ids_train),
        np.array(sample_ids_test),
        np.array(labels_train),
        np.array(labels_test),
    )


def split_samples_traintest(
    sample_ids: list[str],
    labels: list[str],
    proportion: int,
    log: Logger,
) -> tuple[list[str], list[str], list[str], list[str]]:

    assert len(sample_ids) == len(
        labels
    ), f"Len of contigs {len(sample_ids)} and contig names {len(labels)} dont match."

    n_total = len(sample_ids)
    n_test = int(proportion * n_total)

    rng = np.random.default_rng(seed=42)
    test_indices = rng.choice(n_total, size=n_test, replace=False)

    mask = np.ones(n_total, dtype=bool)
    mask[test_indices] = False

    sample_ids_train = [sample_ids[i] for i in range(n_total) if mask[i]]
    sample_ids_test = [sample_ids[i] for i in test_indices]

    labels_train = [labels[i] for i in range(n_total) if mask[i]]
    labels_test = [labels[i] for i in test_indices]

    # Verify splits
    assert (
        len(sample_ids_train) + len(sample_ids_test)
        == len(labels_train) + len(labels_test)
        == n_total
    ), "Mismatch between total counts"
    assert (
        len(sample_ids_train) == len(labels_train) == (n_total - n_test)
    ), "Mismatch between test counts"
    assert (
        len(sample_ids_test) == len(labels_test) == n_test
    ), "Mismatch between validation counts"

    log.append(
        f"Splitting into validation/test using proportion={proportion}\n"
        f"N total: {n_total}, N val: {n_test}, N test: {n_total - n_test}"
    )

    return sample_ids_train, sample_ids_test, labels_train, labels_test


def get_available_device() -> tuple[torch.device, int]:
    """
    Returns the best available device for PyTorch computations.
    - If CUDA (GPU) is available, it returns 'cuda' and the number of GPUs.
    - If MPS (Apple GPU) is available, it returns 'mps' and 1 (for batch size handling).
    - Otherwise, it returns 'cpu' and 1.

    Returns:
        tuple[torch.device, int]: A tuple containing the device and the number of available devices.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        return device, gpu_count
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, 1
    else:
        device = torch.device("cpu")
        return device, 1
