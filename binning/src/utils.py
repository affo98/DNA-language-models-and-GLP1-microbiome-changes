import gzip
from Bio import SeqIO
import csv

import numpy as np
import torch

csv.field_size_limit(2**30)


def read_contigs(contigs_file: str) -> list[str]:
    """Read in contigs from a fasta file. Either Gzip or normal file."""

    contigs = []
    try:
        with gzip.open(contigs_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                contigs.append(str(record.seq))
    except Exception:
        with open(contigs_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                contigs.append(str(record.seq))
    return contigs


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
