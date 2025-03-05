import numpy as np
from tqdm import tqdm


def calculate_tnf(dna_sequences: list[str]) -> np.ndarray:
    """Calculates tetranucleotide frequencies in a list of DNA sequences.

    This function computes the frequencies of all possible tetranucleotides (sequences of four nucleotides)
    within each DNA sequence in `dna_sequences`. Corresponds to calculate 4-mers. There are 4^4=256 combinations.

    Args:
        dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list of nucleotide characters (A, T, C, or G).

    Returns:
            - embeddings (np.ndarray): A 2D numpy array of shape (n, 256), where `n` is the number of DNA sequences,
              and each row contains the tetranucleotide frequency vector for a sequence.

    TO-DO: UPDATE TNF TO USE 103-DIMENSIONAL - SEE VAMB
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
    no_missing_tns = 0
    for j, seq in tqdm(
        enumerate(dna_sequences), total=len(embeddings), desc="Calculating TNFs"
    ):
        count_N = 0
        for i in range(len(seq) - 3):
            try:
                tetra_nuc = seq[i : i + 4]
                embeddings[j, tnf_index[tetra_nuc]] += 1
            except KeyError:  # there exist nucleotide N which will give error
                count_N += 1 / 4
        if len(seq) > 0:
            no_missing_tns += count_N / len(seq)

    print(f"Average Number of Missing Nucleotide (N): {count_N/len(dna_sequences)}")
    # Convert counts to frequencies
    total_counts = np.sum(embeddings, axis=1)
    embeddings = embeddings / total_counts[:, None]

    return embeddings
