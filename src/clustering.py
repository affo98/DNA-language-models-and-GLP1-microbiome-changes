import torch
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import normalize

from src.utils import get_available_device


class KMediod:

    def check_params(
        self,
        embeddings: np.ndarray,
        min_similarity: float,
        min_bin_size: int,
        num_steps: int,
        max_iter: int,
        normalized: bool,
    ):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            print("Embeddings chenged to dtype float32")
        if not normalized:
            embeddings = normalize(embeddings)
        if not 0 < min_similarity < 1:
            raise (ValueError("Minimum similarity must be between 0 and 1"))
        if min_bin_size < 1:
            raise ValueError("Minimum bin size must be at least 1")
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1")
        if max_iter < 1:
            raise ValueError("Maximum iterations must be at least 1")
        if len(embeddings) < 1:
            raise ValueError("Matrix must have at least 1 observation.")

    def __init__(
        self,
        embeddings: np.ndarray,
        min_similarity: float = 0.8,
        min_bin_size: int = 10,
        num_steps: int = 3,
        max_iter: int = 1000,
        normalized: bool = False,
    ):
        self.check_params(
            embeddings, min_similarity, min_bin_size, num_steps, max_iter, normalized
        )

        device, gpu_count = get_available_device()
        embeddings = torch.from_numpy(embeddings).to(device)
        print(f"Using {device} for k-mediod clustering")

        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self.min_bin_size = min_bin_size
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.device = device

    def fit(
        self,
    ) -> np.array:

        block_size = 30000
        n_samples = self.embeddings.shape[0]

        density_vector = torch.zeros(n_samples, device=self.device)

        for i in range(0, n_samples, block_size):
            block_start = i
            block_end = min(i + block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(
                block_embeddings, self.embeddings.T
            )  # Shape: (block_size, n) - sim of block to all other data points

            block_density = torch.sum(
                torch.where(
                    block_sim_matrix >= self.min_similarity, block_sim_matrix, 0.0
                ),
                dim=1,
            )

            density_vector[block_start:block_end] = block_density

        predictions = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        cluster_id = 0

        print("=========================================\n")
        print(f"Running KMedoid on {n_samples} contigs\n")
        print("=========================================\n")

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
                candidate_mask = (similarities >= self.min_similarity) & available_mask
                candidates = torch.where(candidate_mask)[0]

                if len(candidates) == 0:
                    break

                seed = torch.mean(self.embeddings[candidates], dim=0)  # update seed

            if len(candidates) == 0:
                cluster_id -= 1  # Rollback unused cluster ID
                continue

            predictions[candidates] = cluster_id

            # Update density vector
            cluster_embs = self.embeddings[candidates]
            print(cluster_embs.shape)
            cluster_sims = torch.mm(self.embeddings, cluster_embs.T)
            cluster_sims = torch.where(
                cluster_sims >= self.min_similarity, cluster_sims, 0.0
            )
            density_vector -= torch.sum(cluster_sims, dim=1)
            density_vector[candidates] = -100  # exclude seed from density vector

            progress_bar.update(1)
            if cluster_id % 20 == 0:
                print(f"KMediod Step {cluster_id} completed.")

        # Filter small clusters
        print(f"filter small clusters")
        labels, counts = torch.unique(predictions, return_counts=True)
        for label, count in zip(labels.cpu(), counts.cpu()):
            if label == -1 or count >= self.min_bin_size:
                continue
            predictions[predictions == label] = -1

        return predictions.cpu().numpy()
