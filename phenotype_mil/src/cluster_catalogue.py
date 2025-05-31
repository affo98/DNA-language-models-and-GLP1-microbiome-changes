"""Creates the cluster catalogue (n_clusters x embedding_dim) from the clusters and seeds files."""

import os
import json

import numpy as np

from src.utils import Logger, read_clusters


CLUSTERS_FILENAME = "clusters_filtered.tsv"
SEEDS_FILENAME = "seeds.npz"


def get_cluster_catalogue(
    input_path: str,
    log: Logger,
) -> np.array:
    """
    Create a cluster catalogue from the input path and save it to the specified path.
    """

    clusters_filtered_path = os.path.join(input_path, CLUSTERS_FILENAME)
    clusters = read_clusters(clusters_filtered_path, log)

    seeds_path = os.path.join(input_path, SEEDS_FILENAME)
    cluster_catalogue_centroid = get_seeds_centroid_catalogue(
        seeds_path, clusters.keys(), log
    )

    return cluster_catalogue_centroid


def get_seeds_centroid_catalogue(seeds_path: str, clusterids, log) -> dict:
    """Loads a seed-centroid npz file and filter only the seeds that are in the clusterids."""
    seeds_npz = np.load(seeds_path, allow_pickle=True)
    seed_labels = seeds_npz["seed_labels"]
    seed_embeddings = seeds_npz["seeds"]

    clusterids = set(int(cid) for cid in clusterids)

    seeds = {
        str(label): np.array(seed)
        for label, seed in zip(seed_labels, seed_embeddings)
        if int(label) in clusterids
    }
    log.append(f"Cluster catalogue seed-centroid shape: {len(seeds)}")

    return seeds
