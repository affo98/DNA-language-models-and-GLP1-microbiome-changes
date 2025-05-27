import os
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from src.utils import get_available_device, Logger


class Threshold:

    def check_params(
        self,
        embeddings: np.ndarray,
        n_bins: int,
        block_size: int,
    ):
        if embeddings.dtype != np.float64:
            embeddings = embeddings.astype(np.float64)
            print("Embeddings changed to dtype float64")
        if block_size < 1:
            raise ValueError("Block size must be at least 1")
        if n_bins < 1:
            raise ValueError("Number of bins must be at least 1")

    def __init__(
        self,
        embeddings: np.ndarray,
        n_bins: int,
        block_size: int,
        save_path: str,
        model_name: str,
        log: Logger,
    ):
        self.check_params(embeddings, n_bins, block_size)

        device, gpu_count = get_available_device()
        embeddings = torch.from_numpy(embeddings).to(device)

        self.embeddings = embeddings
        self.n_bins = n_bins
        self.block_size = block_size
        self.save_path = save_path
        self.model_name = model_name
        self.log = log
        self.device = device

        self.log.append(f"Using {device} for Threshold calculations")

    def apply_mp(self, sim_matrix: torch.Tensor, n_neighbors: int) -> torch.Tensor:
        from sklearn.neighbors import kneighbors_graph
        from skhubness.utils.kneighbors_graph import check_kneighbors_graph
        from skhubness.reduction import MutualProximity

        distance_matrix = 1 - sim_matrix.cpu().numpy()  # convert to dist.
        n_samples_mp = distance_matrix.shape[0]

        # need sorted graph in this format
        knn_graph = kneighbors_graph(
            distance_matrix,
            n_neighbors=n_neighbors,
            mode="distance",
            include_self=False,
        )
        knn_graph_sorted = check_kneighbors_graph(knn_graph)

        mp = MutualProximity(method="normal", verbose=0)
        mp_graph = mp.fit_transform(knn_graph_sorted)

        # replace original distances of k neighbors with MP-distances, and keep other distances
        distance_matrix_mp = np.copy(distance_matrix)
        for i in range(n_samples_mp):
            neighbor_indices = knn_graph_sorted[i].indices
            distance_matrix_mp[i, neighbor_indices] = (
                mp_graph[i].toarray().ravel()[neighbor_indices]
            )

        sim_matrix_mp = 1 - distance_matrix_mp  # convert to sim.
        sim_matrix_mp = torch.tensor(
            sim_matrix_mp, dtype=torch.float64, device=self.device
        )

        return sim_matrix_mp

    def get_knn_threshold(self, knn_k, knn_p) -> float:

        self.knn_k = knn_k
        self.knn_p = knn_p

        n_samples = self.embeddings.shape[0]
        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

        # first, find min/max
        global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
        global_max = torch.tensor([-1], dtype=torch.float32, device=self.device)
        for i in tqdm(
            range(0, n_samples, self.block_size), desc="Calculating global min/max"
        ):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
            block_sim_matrix = self.apply_mp(block_sim_matrix, knn_k)  # APPLY MP HERE

            top_k_similarities, top_k_indices = torch.topk(
                block_sim_matrix, self.knn_k, dim=-1
            )

            top_k_embeddings = self.embeddings[
                top_k_indices
            ]  # shape: (block_size, knn_k, embedding_dim)
            centroids = top_k_embeddings.mean(
                dim=1, keepdim=True
            )  # shape: (block_size, 1, embedding_dim)

            centroids = centroids.transpose(
                1, 2
            )  # Shape: (block_size, embedding_dim, 1)

            centroid_similarities = torch.bmm(top_k_embeddings, centroids).squeeze(-1)
            centroid_similarities_flat = centroid_similarities.flatten()

            global_min = torch.min(global_min, centroid_similarities_flat.min())
            global_max = torch.max(global_max, centroid_similarities_flat.max())

        # loop through again to get histogram
        for i in tqdm(range(0, n_samples, self.block_size), desc="Calculating knns"):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
            block_sim_matrix = self.apply_mp(block_sim_matrix, knn_k)  # APPLY MP HERE

            top_k_similarities, top_k_indices = torch.topk(
                block_sim_matrix, self.knn_k, dim=-1
            )

            top_k_embeddings = self.embeddings[
                top_k_indices
            ]  # shape: (block_size, knn_k, embedding_dim)
            centroids = top_k_embeddings.mean(
                dim=1, keepdim=True
            )  # shape: (block_size, 1, embedding_dim)
            centroids = centroids.transpose(
                1, 2
            )  # Shape: (block_size, embedding_dim, 1)

            centroid_similarities = torch.bmm(top_k_embeddings, centroids).squeeze(-1)
            centroid_similarities_flat = centroid_similarities.flatten()

            bin_vector += torch.histc(
                centroid_similarities_flat,
                bins=self.n_bins,
                min=global_min.item(),
                max=global_max.item(),
            )

        bin_vector = bin_vector / bin_vector.sum()
        bin_vector = bin_vector.cpu().numpy()

        pairsim_vector = (
            torch.linspace(global_min.item(), global_max.item(), self.n_bins)
            .cpu()
            .numpy()
        )

        cumulative_sum = np.cumsum(bin_vector)
        index = np.argmax(cumulative_sum >= (self.knn_p / 100))
        knn_threshold = pairsim_vector[index]

        self.knn_threshold, self.pairsim_vector, self.bin_vector = (
            knn_threshold,
            pairsim_vector,
            bin_vector,
        )

        self.save_histogram(knn=True)
        self.save_to_json()

        return knn_threshold

    def save_to_json(self) -> None:
        """Saves the knn_threshold, pairsim_vector, and bin_vector to a JSON file."""

        data = {
            "knn_threshold": float(self.knn_threshold),
            "pairsim_vector": self.pairsim_vector.tolist(),  # Convert numpy arrays to lists
            "bin_vector": self.bin_vector.tolist(),
        }

        file_path = os.path.join(
            self.save_path, f"k{self.knn_k}_p{self.knn_p}_similarity_histogram.json"
        )

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        self.log.append(f"Threshold data saved to: {file_path}")

        return

    def save_histogram(self, knn=True) -> None:
        """Plots and saves the histogram of similarities from the provided bin_vector."""

        plt.figure(figsize=(8, 6))

        if knn:
            plt.axvline(
                self.knn_threshold,
                color="g",
                linestyle="--",
                label=f"KNN Threshold: {self.knn_threshold} (k={self.knn_k}, p={self.knn_p})",
            )

        plt.plot(
            self.pairsim_vector,
            self.bin_vector,
            color="skyblue",
            linestyle="-",
            linewidth=2,
        )

        plt.xlabel("Similarity Bins")
        plt.ylabel("Frequency")
        plt.title(f"Similarity Histogram {self.model_name}")

        plt.legend()

        file_path = os.path.join(
            self.save_path,
            f"k{self.knn_k}_p{self.knn_p}_similarity_histogram.png",
        )
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        self.log.append(f"Plot saved at: {file_path}")

        return
