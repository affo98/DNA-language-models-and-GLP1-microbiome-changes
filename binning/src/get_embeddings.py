import os

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import normalize


def get_embeddings(
    dna_sequences: list[str],
    batch_size: list[int],
    model_name: str,
    model_path: str,
    save_path: str,
    normalize: bool,
) -> np.array:
    """
    Generate or load embeddings for a given set of DNA sequences using the specified model.

    This function generates embeddings using different models based on the specified `model_name`.
    If embeddings have already been computed and saved, it loads them from the provided path.
    Otherwise, it calculates new embeddings, saves them, and returns the result.
    This function calls other functions to calculate embeddings.

    Args:dings will be generated.
    model_name (str): Name of the model to use for embedding generation. Models can be found in congis/model.yml.
    model_path (str): Path to the pretrained model file or directory
    dna_sequences (list of str): List of DNA sequences for which embed required for specific models.
    save_path (str): Path to save the computed embeddings or to load existing ones.

    Returns:
    np.array: Array of embeddings with shape (num_sequences, embedding_dimension).

    """

    if os.path.exists(save_path):
        print(f"Load embeddings from file {save_path}\n")
        embeddings = np.load(save_path)

        if embeddings.shape[0] == len(dna_sequences):
            return embeddings
        else:
            print(
                f"Mismatch in number of embeddings from {save_path} and DNA sequences.\nRecalculating embeddings."
            )
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if model_name == "tnf":
        embeddings = calculate_tnf(dna_sequences, model_path)
    elif model_name == "tnf_kernel":
        embeddings = calculate_tnf(dna_sequences, model_path, use_kernel=True)

    if normalize:
        embeddings = normalize(embeddings)

    with open(save_path, "wb") as f:
        np.save(f, embeddings)

    return embeddings


def validate_input_array(array: np.ndarray) -> np.ndarray:
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    if not array.flags["OWNDATA"]:
        array = array.copy()

    assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]
    return array


def calculate_tnf(
    dna_sequences: list[str], model_path: str, use_kernel: bool = False
) -> np.ndarray:
    """Calculates tetranucleotide frequencies in a list of DNA sequences.

    This function computes the frequencies of all possible tetranucleotides (sequences of four nucleotides)
    within each DNA sequence in `dna_sequences`. Corresponds to calculate 4-mers. There are 4^4=256 combinations.

    Args:
        dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list of nucleotide characters (A, T, C, or G) or possibly other letters such as N.
        kernel (bool): Whether to use VAMB-kernel to downproject TNFs into 103-dimensions. We download the the pre-computed kernel from VAMB (https://github.com/RasmussenLab/vamb/blob/master/vamb/kernel.npz) and save it in helper/.
        See more at: https://github.com/RasmussenLab/vamb/blob/master/src/create_kernel.py

    Returns:
            - embeddings (np.ndarray): A 2D numpy array of shape (n, 256), where `n` is the number of DNA sequences,
              and each row contains the tetranucleotide frequency vector for a sequence.

    """

    nucleotides = ["A", "T", "C", "G"]
    tetra_nucleotides = [
        a + b + c + d
        for a in nucleotides
        for b in nucleotides
        for c in nucleotides
        for d in nucleotides
    ]

    # Build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}

    # Build embeddings by counting TNFs
    embeddings = np.zeros((len(dna_sequences), len(tetra_nucleotides)))

    for j, seq in tqdm(
        enumerate(dna_sequences), total=len(embeddings), desc="Calculating TNFs"
    ):
        for i in range(len(seq) - 3):
            try:
                tetra_nuc = seq[i : i + 4]
                embeddings[j, tnf_index[tetra_nuc]] += 1
            except KeyError:  # there exist nucleotide N which will give error
                continue

        if embeddings[j, :].sum() == 0:
            raise ValueError(
                f"TNF value of contig at index {j} is all zeros. "
                + "This implies that the sequence contained no 4-mers of A, C, G, T or U, "
                + "making this sequence uninformative. This is probably a mistake. "
                + "Verify that the sequence contains usable information (e.g. is not all N's)"
            )

    total_counts = np.sum(embeddings, axis=1)
    total_counts[total_counts == 0] = 1e-10  # avoid division by 0
    embeddings = embeddings / total_counts[:, None]

    if use_kernel:
        kernel_raw = np.load(model_path)
        kernel = validate_input_array(kernel_raw["arr_0"])

        tnf_embeddings += -(1 / 256)

        embeddings = np.dot(embeddings, kernel)

    return embeddings
