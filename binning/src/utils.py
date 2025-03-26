import gzip
from Bio import SeqIO
import csv

import numpy as np
import torch

csv.field_size_limit(2**30)


def read_contigs(contigs_file: str) -> list[str]:
    """Read in contigs from a fasta file."""

    contigs = []
    with gzip.open(contigs_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            contigs.append(str(record.seq))
    return contigs[0:10000]


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
