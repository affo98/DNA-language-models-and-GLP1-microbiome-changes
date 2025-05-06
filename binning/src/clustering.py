import os
import json

import torch
import numpy as np
from tqdm import tqdm


from src.utils import get_available_device, Logger, to_fp16_tensor


class KMediod:

    def check_params(
        self,
        embeddings: np.ndarray,
        contig_names: list[str],
        min_bin_size: int,
        num_steps: int,
        max_iter: int,
    ):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            print("Embeddings changed to dtype float32")
        if min_bin_size < 1:
            raise ValueError("Minimum bin size must be at least 1")
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1")
        if max_iter < 1:
            raise ValueError("Maximum iterations must be at least 1")
        if len(embeddings) < 1:
            raise ValueError("Matrix must have at least 1 observation.")
        assert len(contig_names) == len(
            embeddings
        ), f"Number of embeddings {len(embeddings)} does not match number of contig names {len(contig_names)}"

    def __init__(
        self,
        embeddings: np.ndarray,
        contig_names: list[str],
        save_path: str,
        log: Logger,
        log_verbose: bool,
        mode: str,
        min_bin_size: int = 10,
        num_steps: int = 3,
        max_iter: int = 1000,
        block_size: int = 1000,
    ):
        self.check_params(embeddings, contig_names, min_bin_size, num_steps, max_iter)

        device, gpu_count = get_available_device()

        embeddings = to_fp16_tensor(embeddings, device=device, log=log)
        # embeddings = torch.from_numpy(embeddings).to(device)

        self.embeddings = embeddings
        self.contig_names = contig_names
        self.save_path = save_path
        self.log = log
        self.log_verbose = log_verbose
        self.mode = mode
        self.min_bin_size = min_bin_size
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.device = device
        self.block_size = block_size

    def fit(self, min_similarity: float, knn_k: int, knn_p: float) -> np.array:
        """Runs the Iterative k-mediod algorithm, and saves the output predictions."""

        if not 0 < min_similarity < 1:
            raise (ValueError("Minimum similarity must be between 0 and 1"))
        self.log.append(
            f"Using {self.device} and threshold {min_similarity} for k-mediod clustering"
        )

        n_samples = self.embeddings.shape[0]

        density_vector = torch.zeros(n_samples, device=self.device)
        seeds = []
        seed_labels = []

        for i in range(0, n_samples, self.block_size):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(
                block_embeddings, self.embeddings.T
            )  # Shape: (block_size, n) - sim of block to all other data points

            block_density = torch.sum(
                torch.where(block_sim_matrix >= min_similarity, block_sim_matrix, 0.0),
                dim=1,
            )

            density_vector[block_start:block_end] = block_density

        predictions = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        cluster_id = 0

        self.log.append("=========================================\n")
        self.log.append(f"Running KMedoid on {n_samples} contigs\n")
        self.log.append("=========================================\n")

        progress_bar = tqdm(total=self.max_iter, desc="Clusters created K-mediod")

        while torch.any(predictions == -1):
            cluster_id += 1
            if cluster_id > self.max_iter:
                break

            # Select highest density point index
            medoid_idx = torch.argmax(density_vector).item()
            density_vector[medoid_idx] = -100  # exclude seed from density vector

            seed = self.embeddings[medoid_idx]
            available_mask = predictions == -1  # points that are still available

            for _ in range(self.num_steps):
                similarities = torch.mv(self.embeddings, seed)
                candidate_mask = (similarities >= min_similarity) & available_mask
                candidates = torch.where(candidate_mask)[0]

                if len(candidates) == 0:
                    break

                seed = torch.mean(self.embeddings[candidates], dim=0)  # update seed

            predictions[candidates] = cluster_id
            seeds.append(seed.detach().cpu().numpy())
            seed_labels.append(cluster_id)

            # Update density vector in blocks
            for i in range(0, n_samples, self.block_size):
                block_start = i
                block_end = min(i + self.block_size, n_samples)
                block_embs = self.embeddings[block_start:block_end]

                cluster_sims = torch.mm(block_embs, self.embeddings[candidates].T)
                cluster_sims = torch.where(
                    cluster_sims >= min_similarity, cluster_sims, 0.0
                )

                density_vector[block_start:block_end] -= torch.sum(cluster_sims, dim=1)

            density_vector[candidates] = -100

            progress_bar.update(1)

        self.save_seeds(seeds, seed_labels)

        # Filter small clusters
        labels, counts = torch.unique(predictions, return_counts=True)
        for label, count in zip(labels.cpu(), counts.cpu()):
            if label == -1 or count >= self.min_bin_size:
                continue
            predictions[predictions == label] = -1

        # print cluster sizes
        labels, counts = torch.unique(predictions, return_counts=True)
        if self.log_verbose:
            for i, (label, count) in enumerate(zip(labels.cpu(), counts.cpu())):
                self.log.append(f"Cluster {label}: {count} points")
                if i == 50:
                    break

        predictions = predictions.cpu().numpy()

        assert (
            len(predictions) == len(self.embeddings) == len(self.contig_names)
        ), f"Len of predictions {len(predictions)} does not match embeddings {len(self.embeddings)} and contig_names {len(self.contig_names)}"

        predictions, contig_names = self.remove_unassigned_sequences(predictions)
        self.save_clusters(knn_k, knn_p, predictions, contig_names)

        return predictions, contig_names

    def remove_unassigned_sequences(self, predictions) -> np.array:
        idx_to_keep = np.where(predictions != -1)[0]

        predictions = predictions[idx_to_keep]
        contig_names = [self.contig_names[i] for i in idx_to_keep]

        assert len(predictions) == len(
            contig_names
        ), f"Mismatch between predictions {len(predictions)} and contig names {len(contig_names)} after removing unassigned seqs."
        return predictions, contig_names

    def save_clusters(self, knn_k, knn_p, predictions, contig_names) -> None:
        """save predictions in save_path in format: clustername \\t contigname, and cluster-seeds in a separate file."""

        if self.mode == "val":
            output_file = os.path.join(
                self.save_path, f"clusters_k{knn_k}_p{knn_p}.tsv"
            )
        elif self.mode == "test":
            output_file = os.path.join(self.save_path, f"clusters.tsv")

        with open(output_file, "w") as file:
            file.write("clustername\tcontigname\n")  # header

            for cluster, contig in zip(predictions, contig_names):
                file.write(f"{cluster}\t{contig}\n")

        self.log.append(f"Predictions file written successfully to {self.save_path}")
        return

    def save_seeds(self, seeds, seed_labels) -> None:
        """Save seeds and corresponding labels in a compressed .npz file. Only saved in test-mode."""

        if self.mode == "val":
            return
        elif self.mode == "test":
            output_file = os.path.join(self.save_path, f"seeds.npz")
            np.savez(output_file, seeds=seeds, seed_labels=seed_labels)
            self.log.append(f"Seeds saved to {output_file}")
            return
