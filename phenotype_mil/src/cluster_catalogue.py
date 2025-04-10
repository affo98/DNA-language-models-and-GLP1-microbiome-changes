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
    cluster_catalogue = get_seeds_centroid_catalogue(seeds_path, clusters.keys(), log)

    log.append(f"Cluster catalogue shape: {cluster_catalogue.shape}")
    return cluster_catalogue


def get_seeds_centroid_catalogue(seeds_path: str, clusterids, log) -> np.array:
    """Loads a seeds dictionary saved in JSON format, converting seed lists back to np.arrays."""
    with open(seeds_path, "r") as filehandle:
        data = json.load(filehandle)

    clusterids = set(int(cid) for cid in clusterids)

    seeds = {
        int(label): np.array(seed)
        for label, seed in data.items()
        if int(label) in clusterids
    }
    log.append(f"Using {seeds.shape} seeds from {seeds_path}")

    cluster_catalogue = list(seeds.values())

    return cluster_catalogue
