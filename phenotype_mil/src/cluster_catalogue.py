"""Creates the cluster catalogue (n_clusters x embedding_dim) from the clusters and seeds files."""

import os
import json

import numpy as np

from src.utils import Logger, read_clusters


CLUSTERS_FILENAME = "clusters_filtered.tsv"
SEEDS_FILENAME = "seeds.json"


def get_cluster_catalogue(
    input_path: str,
    log: Logger,
) -> np.array:
    """
    Create a cluster catalogue from the input path and save it to the specified path.
    """

    clusters_filtered_path = os.path.join(input_path, CLUSTERS_FILENAME)
    clusters = read_clusters(clusters_filtered_path)

    seeds_path = os.path.join(input_path, SEEDS_FILENAME)
    seeds = get_seeds(seeds_path, clusters.keys())

    cluster_catalogue = list(seeds.values())

    print(cluster_catalogue.shape)
    print(seeds.shape)

    log.append(f"Cluster catalogue shape: {cluster_catalogue.shape}")
    log.append(f"Seeds shape: {seeds.shape}")
    return cluster_catalogue


def get_seeds(seeds_path: str, clusterids) -> dict[int, np.ndarray]:
    """Loads a seeds dictionary saved in JSON format, converting seed lists back to np.arrays."""
    with open(seeds_path, "r") as filehandle:
        data = json.load(filehandle)

    clusterids = set(int(cid) for cid in clusterids)

    seeds = {
        int(label): np.array(seed)
        for label, seed in data.items()
        if int(label) in clusterids
    }

    return seeds
