import os
import sys
from argparse import ArgumentParser
import json
from time import time
from tqdm import tqdm

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import Logger, get_available_device

CLUSTERS_HEADER = "clustername\tcontigname"
CLUSTERS_FILENAME = "clusters_filtered.tsv"

BLOCK_SIZE = 1000
PERCENTILE = 95.0


def read_clusters(input_path: str, log: Logger) -> dict[str, set[str]]:
    """Read cluster file into a dict[clusterid, set[contigids]]."""
    with open(input_path, "r") as filehandle:
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

    log.append(f"Read {len(clusters_dict)} clusters from {input_path}")
    return clusters_dict


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


def compute_hausdorff_distance(
    A: torch.Tensor,
    B: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    percentile: float = PERCENTILE,
) -> float:
    """
    Compute Hausdorff distance between two sets of normalized embeddings A and B using cosine distance.
    Uses percentile-based symmetric/birectional Hausdorff distance.

    Args:
        A (torch.Tensor): Tensor of shape (n_a, d)
        B (torch.Tensor): Tensor of shape (n_b, d)
        block_size (int): Block size for memory-efficient computation
        percentile (float): Percentile (0–100) of the minimum distances to use instead of max

    Returns:
        float: Percentile Hausdorff distance
    """

    def directed_percentile_hausdorff(X, Y, block_size, percentile):
        min_dists_all = []
        for i in range(0, X.size(0), block_size):
            x_block = X[i : i + block_size]  # (block_size, d)
            sim = torch.mm(x_block, Y.T)  # (block_size, n_y)
            dist = 1 - sim  # cosine distance
            min_dist, _ = dist.min(dim=1)  # (block_size,)
            min_dists_all.append(min_dist)
        min_dists_all = torch.cat(min_dists_all)
        return torch.quantile(min_dists_all, percentile / 100.0).item()

    h_ab = directed_percentile_hausdorff(A, B, block_size, percentile)
    h_ba = directed_percentile_hausdorff(B, A, block_size, percentile)
    bidirectional_distance = max(h_ab, h_ba)
    return bidirectional_distance


def compute_hausdorff_matrix(
    embeddings: np.memmap | np.ndarray,
    contig_names: list[str],
    clusters: dict[str, set[str]],
    device,
    save_path: str,
    log,
):
    """
    Computes the Hausdorff distance matrix between clusters and saves it.

    Args:
        embeddings: np.ndarray or np.memmap of shape (N, D)
        contig_names: list of contig names corresponding to embeddings
        clusters: dict[clustername, set[contigname]]
        block_size: block size for efficient distance calculation
        device: torch device
        save_path: where to save the resulting .npz file
        log: Logger instance
    """
    name_to_index = {name: i for i, name in enumerate(contig_names)}
    cluster_names = list(clusters.keys())
    n_clusters = len(cluster_names)

    log.append(f"Computing Hausdorff distances for {n_clusters} clusters...")

    cluster_embeddings = {}
    torch_embeddings = torch.tensor(embeddings, device=device)

    for cname in cluster_names:
        indices = [name_to_index[n] for n in clusters[cname] if n in name_to_index]
        if not indices:
            raise ValueError(f"No contigs from cluster {cname} found in embeddings.")
        cluster_embeddings[cname] = torch_embeddings[indices]

    distance_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    for i, name_i in enumerate(tqdm(cluster_names, desc="Hausdorff matrix")):
        A = cluster_embeddings[name_i]
        for j in range(i, n_clusters):
            name_j = cluster_names[j]
            B = cluster_embeddings[name_j]

            d = compute_hausdorff_distance(A, B)

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    cluster_array = np.array(cluster_names)

    output_file = os.path.join(save_path, "hausdorff_distances.npz")
    np.savez(output_file, distance_matrix=distance_matrix, cluster_names=cluster_array)

    log.append(f"Saved Hausdorff distance matrix to {output_file}")

    return distance_matrix, cluster_array


def main(args, log):

    device, _ = get_available_device()
    log.append(f"Using {device} for Hausdorff distance calculation")
    clusters = read_clusters(os.path.join(args.input_path, CLUSTERS_FILENAME), log)
    embeddings, contig_names = read_embeddings(args.input_path, args.model_name, log)

    distance_matrix, clusters_array = compute_hausdorff_matrix(
        embeddings,
        contig_names,
        clusters,
        device=device,
        save_path=args.save_path,
        log=log,
    )


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-mn",
        help="Name of the model to get embeddings from",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--input_path",
        "-mp",
        help="Path to the embeddings file",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to save the computed embeddings or to load existing ones",
    )
    parser.add_argument(
        "--log",
        "-l",
        help="Path to save logfile",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time()
    args = add_arguments()

    os.makedirs(args.save_path)
    log = Logger(args.log)
    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    main(args, log)

    end_time = time()
    elapsed_time = end_time - start_time
    log.append(
        f"Hausdorff distances of {args.model_name} ran in {elapsed_time:.2f} Seconds"
    )
