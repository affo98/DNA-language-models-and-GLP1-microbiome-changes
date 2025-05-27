import gzip
from Bio import SeqIO
import csv
import json
import os
import sys
import subprocess

import numpy as np
import torch

csv.field_size_limit(2**30)


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


def get_gpu_mem() -> int | None:
    """Return current GPU memory usage in MiB (as int)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        # if multiple GPUs, take the first line
        used = int(out.strip().split("\n")[0])
        return used
    except Exception as e:
        print("Warning: could not query nvidia-smi:", e, file=sys.stderr)
        return None


def read_embeddings(
    input_path: str, model_name, log: Logger
) -> tuple[np.memmap | np.ndarray, list[str]]:
    """Read embeddings from file and return as numpy array and list of contig names."""

    with open(os.path.join(input_path, "n_total_val_test.json")) as f:
        n_val_test_data = json.load(f)
    n_test = n_val_test_data["n_test"]
    log.append(f"Number of test contigs: {n_test}")

    try:
        embeddings = np.memmap(
            os.path.join(input_path, "embeddings", "embeddings.npy"),
            dtype="float32",
            mode="r",
            shape=(n_test, 768),
        )  # embeddings_array = np.array(embeddings)
        contig_names = np.load(
            os.path.join(input_path, "embeddings", "contignames.npy"), allow_pickle=True
        )
    except:
        embedding_data = np.load(
            os.path.join(input_path, "embeddings", f"{model_name}.npz")
        )
        embeddings = embedding_data["embeddings"]
        contig_names = embedding_data["contig_names"]

    log.append(f"Read {embeddings.shape[0]} embeddings from {input_path}")

    return embeddings, contig_names


def read_contigs(
    contigs_file: str, filter_len: int, log: Logger
) -> tuple[list[str], list[str]]:
    """Read in contigs from a fasta file. Either Gzip or normal file.
    Filters contigs larger than filter_len"""

    contigs, contig_names, total_contigs = [], [], 0

    try:
        with gzip.open(contigs_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                total_contigs += 1
                if len(record.seq) < filter_len:
                    contigs.append(str(record.seq))
                    contig_names.append(str(record.id))

    except Exception:
        with open(contigs_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                total_contigs += 1
                if len(record.seq) < filter_len:
                    contigs.append(str(record.seq))
                    contig_names.append(str(record.id))

    log.append(
        f"Removing contigs above {filter_len} base-pairs\nTotal Contigs before filtering: {total_contigs}"
    )
    log.append(f"Total Contigs after filtering: {len(contigs)}")

    assert len(contigs) == len(
        contig_names
    ), f"Len of contigs {len(contigs)} and contig names {len(contig_names)} dont match."

    return contigs, contig_names


def split_contigs_valtest(
    contigs: list[str],
    contig_names: list[str],
    log: Logger,
    save_path=None,
    proportion: int = 0.1,
) -> tuple[list[str], list[str], list[str], list[str]]:

    assert len(contigs) == len(
        contig_names
    ), f"Len of contigs {len(contigs)} and contig names {len(contig_names)} dont match."

    n_total = len(contigs)
    n_val = int(proportion * n_total)

    rng = np.random.default_rng(seed=42)
    val_indices = rng.choice(n_total, size=n_val, replace=False)

    mask = np.ones(n_total, dtype=bool)
    mask[val_indices] = False

    contigs_test = [contigs[i] for i in range(n_total) if mask[i]]
    contigs_val = [contigs[i] for i in val_indices]

    contigs_names_test = [contig_names[i] for i in range(n_total) if mask[i]]
    contigs_names_val = [contig_names[i] for i in val_indices]

    # Verify splits
    assert (
        len(contigs_test) + len(contigs_val)
        == len(contigs_names_test) + len(contigs_names_val)
        == n_total
    ), "Mismatch between total counts"
    assert (
        len(contigs_test) == len(contigs_names_test) == (n_total - n_val)
    ), "Mismatch between test counts"
    assert (
        len(contigs_val) == len(contigs_names_val) == n_val
    ), "Mismatch between validation counts"

    log.append(
        f"Splitting into validation/test using proportion={proportion}\n"
        f"N total: {n_total}, N val: {n_val}, N test: {n_total - n_val}"
    )
    if save_path:
        save_dict = {"n_total": n_total, "n_val": n_val, "n_test": n_total - n_val}
        with open(os.path.join(save_path, "n_total_val_test.json"), "w") as f:
            json.dump(save_dict, f, indent=4)

    return contigs_test, contigs_val, contigs_names_test, contigs_names_val


def sort_sequences(dna_sequences: list[str]) -> tuple[list, np.array]:
    """Sorting sequences by length and returning sorted sequences and indices

    Args:
        data (list): List ID, DNA Sequence, Label from loaded CSV

    Returns:
        tuple[list, list]: Sorted DNA Sequences and corresponding indices
    """
    lengths = [len(seq) for seq in dna_sequences]
    idx_asc = np.argsort(lengths)
    idx_desc = idx_asc[::-1]
    dna_sequences = [dna_sequences[i] for i in idx_desc]

    return dna_sequences, idx_desc


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


def validate_input_array(array: np.ndarray) -> np.ndarray:
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    if not array.flags["OWNDATA"]:
        array = array.copy()

    assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]
    return array
