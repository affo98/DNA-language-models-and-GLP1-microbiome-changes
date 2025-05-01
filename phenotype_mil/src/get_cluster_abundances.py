import sys
from tqdm import tqdm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def create_cluster_mappings(
    path_to_clusters: str,
) -> tuple[dict[str, int], dict[int, int], dict[int, int]]:
    """Loads the filtered_clusters and creates 3 dictionaries used for mapping and lookups:
    -------
        1) contig_to_cluster: large dict with each contig mapping to their cluster_id |
            {"S1C1010":1}
        2) cluster_id_to_enumerate: maps cluster_ids to indexes used in downstream numpy arrays
        3) enumerate_to_cluster_id: maps numpy arrays indices to cluster_ids

    Args:
        path_to_clusters (str): path to filtered clusters (output from binning pipeline)

    Returns:
    -------
        tuple[dict[str, int], dict[int, int], dict[int, int]]:
            1) contig_to_cluster\n
            2) cluster_id_to_enumerate\n
            3) enumerate_to_cluster_id
    """
    clusters_df = pd.read_csv(path_to_clusters, sep="\t")
    # plotting distribution
    plotting_cluster_contig_distribution(clusters_df)
    contig_to_cluster = clusters_df.set_index("contigname")["clustername"].to_dict()

    cluster_id_to_enumerate = {}
    enumerate_to_cluster_id = {}
    for i, cluster_id in enumerate(clusters_df["clustername"].unique()):
        cluster_id_to_enumerate[cluster_id] = i
        enumerate_to_cluster_id[i] = cluster_id

    return contig_to_cluster, cluster_id_to_enumerate, enumerate_to_cluster_id


def plotting_cluster_contig_distribution(clusters_df: pd.DataFrame) -> None:
    """Plotting the number of contigs per cluster in a bar plot with log y scale
    and saving as png

    Args:
        clusters_df (pd.DataFrame): filtered cluster results

    Returns:
        None:
    """
    # Only used for plotting

    clusters_contigs_sets = dict()
    for cluster in clusters_df["clustername"].unique():
        clusters_contigs_sets[cluster] = set()

    for cluster, contig_name in zip(
        clusters_df["clustername"].to_list(), clusters_df["contigname"].to_list()
    ):
        clusters_contigs_sets[cluster].add(contig_name)

    n_clusters = len(clusters_df["clustername"].unique())
    print(f"Number of clusters: {n_clusters}")

    contig_distribution = sorted(
        [len(clusters_contigs_sets[i]) for i in clusters_contigs_sets.keys()],
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n_clusters), contig_distribution, width=0.5)
    ax.set_ylabel("n_contigs in cluster")
    ax.set_xlabel("clusters")
    ax.set_yscale("log")
    plt.savefig("./cluster_contig_abundance_distribution.png")

    return None


def extract_samples_and_contigs_from_tsv(
    path_to_abundances: str,
) -> tuple[pd.Index, list[str]]:
    """Extract the sample ids and all contigs from the abundance file, without loading
    the full dataframe into memory.

    Args:
        path_to_abundances (str): Path to the normalized abundances.

    Returns:
        tuple[pd.Index, list[str]]:
        ------
        pd.Index: iterable of contig names\n
        list[str]: list of sample names from the header file
    """

    with open(path_to_abundances) as f:
        header = f.readline().strip().split("\t")

    contigs_col, sample_cols = header[0], header[1:]  # exclude index column

    df = pd.read_csv(
        path_to_abundances, sep="\t", usecols=[contigs_col], index_col=contigs_col
    )

    contigs = df.index

    return contigs, sample_cols


def read_abundances_from_sample(path_to_abundances: str, sample_col: str) -> np.ndarray:
    """Reads a the abundances from one sample from the abundances file given a sample column name.

    Args:
        path_to_abundances (str): path to normalized abundances
        sample_col (str): column name of the sample in the abundance file

    Returns:
        np.ndarray: abundances in a numpy array
    """

    df = pd.read_csv(path_to_abundances, sep="\t", usecols=[sample_col])

    return df.values


def calculate_cluster_abundances(
    contigs: pd.Index,
    contig_abundances: np.ndarray,
    contig_to_cluster: dict,
    cluster_to_array_index: dict,
) -> np.ndarray:
    """Aggregate the contig abundances per cluster and l1 normalize the abundances,
    such that the sample wise abundances sum to 1.

    Args:
        contigs (pd.Index): iterable of the contig names
        contig_abundances (np.ndarray): normalized abundances
        contig_to_cluster (dict): mapping from contig name to cluster id
        cluster_to_array_index (dict): mapping from cluster id to numpy array index

    Returns:
        np.ndarray: l1 normalized cluster abundances
    """
    cluster_abundances = np.zeros((len(cluster_to_array_index.keys()), 1))

    for contig_name, contig_abundances in tqdm(
        zip(contigs, contig_abundances), desc="Calculating cluster abundances"
    ):
        try:
            cluster_id = contig_to_cluster[contig_name]
            cluster_sum_index = cluster_to_array_index[cluster_id]
            cluster_abundances[cluster_sum_index] += contig_abundances
        except KeyError:
            continue

    cluster_abundances_normalized = normalize(cluster_abundances, axis=0, norm="l1")
    return cluster_abundances_normalized


def write_cluster_abundances(
    path_to_cluster_abundances: str,
    list_of_cluster_abundances: list[np.ndarray],
    enumerate_to_cluster_id: dict[int, int],
    sample_list: list[str],
) -> None:
    """Write the cluster abundances to a tsv file of shape: (n_clusters x n_samples)

    Args:
        path_to_cluster_abundances (str): path to normalized abundances
        list_of_cluster_abundances (list[np.ndarray]): list of l1 normalized cluster abundances
        enumerate_to_cluster_id (dict[int, int]): mapping from numpy array index to cluster id
        sample_list (list[str]): list of sample names.
    """

    cluster_abundances = np.concatenate(list_of_cluster_abundances, axis=1)

    cluster_abundances_df_index = [
        enumerate_to_cluster_id[key] for key in enumerate_to_cluster_id.keys()
    ]

    cluster_abundaces_df = pd.DataFrame(
        cluster_abundances,
        columns=sample_list,
        index=cluster_abundances_df_index,
        dtype=np.float32,
    )
    cluster_abundaces_df.index.name = "cluster_id"

    cluster_abundaces_df.to_csv(path_to_cluster_abundances, sep="\t")
    return


if __name__ == "__main__":

    """Should be executed from snakemake with:
    python get_cluster_abundances.py path/to/filtered_cluster_results\n
    path/to/normalized/abundances path/to/output.tsv

    In snakemake:
    python get_cluster_abundances.py {input.clusters_filtered} {input.norm_abundances} {output}
    """

    PATH_TO_CLUSTERS = sys.argv[1]
    PATH_TO_NORMALIZED_ABUNDANCES = sys.argv[2]
    PATH_TO_CLUSTER_ABUNDANCES = sys.argv[3]

    contigs_lookup, cluster_to_index, index_to_cluster = create_cluster_mappings(
        PATH_TO_CLUSTERS
    )

    CONTIGS, SAMPLES = extract_samples_and_contigs_from_tsv(
        PATH_TO_NORMALIZED_ABUNDANCES
    )

    cluster_abundance_list = []

    for sample in tqdm(SAMPLES, desc="Calculating cluster abundances"):
        sample_abundances = read_abundances_from_sample(
            PATH_TO_NORMALIZED_ABUNDANCES, sample
        )

        sample_cluster_abundances = calculate_cluster_abundances(
            CONTIGS, sample_abundances, contigs_lookup, cluster_to_index
        )

        cluster_abundance_list.append(sample_cluster_abundances)

    write_cluster_abundances(
        PATH_TO_CLUSTER_ABUNDANCES, cluster_abundance_list, index_to_cluster, SAMPLES
    )
