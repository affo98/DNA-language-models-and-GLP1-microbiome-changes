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

BLOCK_SIZE = 10000
PERCENTILE = 95.0

DATASETS = [
    "airways_short",
    "gastro_short",
    "oral_short",
    "urogenital_short",
    "skin_short",
    "plant_short",
    "marine_short",
    "metahit",
]

MODELS = ["dnaberts"]


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


def directed_percentile_hausdorff(X, Y, device, block_size, percentile):
    min_dists_all = []
    NX = X.shape[0]
    NY = Y.shape[0]
    for i in range(0, NX, block_size):
        i_end = min(i + block_size, NX)
        block_i_np = X[i:i_end]
        block_i = torch.from_numpy(block_i_np).to(device)
        min_dist_block = None

        for j in range(0, NY, block_size):
            j_end = min(j + block_size, NY)
            block_j_np = Y[j:j_end]
            block_j = torch.from_numpy(block_j_np).to(device)
            sim = torch.mm(block_i, block_j.T)
            dist = 1 - sim

            if min_dist_block is None:
                min_dist_block = dist.min(dim=1).values  # (block_size, )
            else:
                min_dist_block = torch.minimum(min_dist_block, dist.min(dim=1).values)

        min_dists_all.append(min_dist_block)  # list of (block_size, ) tensors

    min_dists_all = torch.cat(min_dists_all)
    return torch.quantile(min_dists_all, percentile / 100.0).item()


def compute_hausdorff_distance(
    A: np.ndarray,
    B: np.ndarray,
    device,
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

    h_ab = directed_percentile_hausdorff(A, B, device, block_size, percentile)
    h_ba = directed_percentile_hausdorff(B, A, device, block_size, percentile)
    bidirectional_distance = max(h_ab, h_ba)
    return bidirectional_distance


def compute_hausdorff_matrix(
    embeddings: np.memmap | np.ndarray,
    contig_names: list[str],
    clusters: dict[str, set[str]],
    device,
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

    for cname in cluster_names:
        indices = [name_to_index[n] for n in clusters[cname] if n in name_to_index]
        if not indices:
            raise ValueError(f"No contigs from cluster {cname} found in embeddings.")
        cluster_embeddings[cname] = embeddings[indices]

    distance_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    for i, name_i in tqdm(enumerate(cluster_names), desc="Hausdorff matrix"):
        A = cluster_embeddings[name_i]
        for j in range(i, n_clusters):
            name_j = cluster_names[j]
            B = cluster_embeddings[name_j]

            d = compute_hausdorff_distance(A, B, device, BLOCK_SIZE, PERCENTILE)

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    cluster_array = np.array(cluster_names)

    return distance_matrix, cluster_array


def main(model_name, dataset_name, input_path, save_path, log):

    device, _ = get_available_device()
    log.append(f"Using {device} for Hausdorff distance calculation")
    clusters = read_clusters(os.path.join(input_path, CLUSTERS_FILENAME), log)
    embeddings, contig_names = read_embeddings(input_path, model_name, log)

    distance_matrix, cluster_array = compute_hausdorff_matrix(
        embeddings,
        contig_names,
        clusters,
        device=device,
        save_path=args.save_path,
        log=log,
    )

    output_file = os.path.join(save_path, f"{model_name}_{dataset_name}.npz")
    np.savez(output_file, distance_matrix=distance_matrix, cluster_names=cluster_array)
    log.append(f"Saved Hausdorff distance matrix to {output_file}")


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-mn",
        help="Name of the model to get embeddings from",
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        help="Name of the dataset",
        default=None,
    )
    parser.add_argument(
        "--input_path",
        "-mp",
        help="Path to the embeddings file (skips model/dataset lookup)",
        default=None,
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to save the computed embeddings or to load existing ones",
    )
    parser.add_argument(
        "--log",
        "-l",
        help="Filename for logfile",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time()
    args = add_arguments()

    os.makedirs(args.save_path)
    log = Logger(os.path.join(args.save_path, args.log))
    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    # If model name is given as input, only run that model. Otherwise run all models on all datasets.
    if bool(args.model_name and args.dataset_name and args.input_path):
        log.append(f"Running Hausdorff on single model and dataset")
        main(args.model_name, args.dataset_name, args.input_path, args.save_path, log)
    else:
        log.append(f"Running Hausdorff on {MODELS} and {DATASETS}")

        for model_name in MODELS:
            for dataset_name in DATASETS:
                input_path = os.path.join(
                    "cami2_benchmark",
                    "model_results",
                    f"{dataset_name}",
                    f"{model_name}_output",
                    "test",
                )
                main(model_name, dataset_name, input_path, args.save_path, log)

    end_time = time()
    elapsed_time = end_time - start_time
    log.append(
        f"Hausdorff distances of {args.model_name} on dataset {args.dataset_name} ran in {elapsed_time:.2f} Seconds"
    )
