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
        print(f"Using {device} to for k-mediod clustering")

        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self.min_bin_size = min_bin_size
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.device = device

    def fit(
        self,
        embeddings: np.ndarray,
        min_similarity=0.8,
        min_bin_size=10,
        num_steps=3,
        max_iter=1000,
    ) -> np.array:

        n_samples = embeddings.shape[0]

        sim_matrix = torch.mm(embeddings, embeddings.T)

        density_vector = torch.sum(
            torch.where(sim_matrix >= min_similarity, sim_matrix, 0.0), dim=1
        )

        predictions = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        cluster_id = 0

        print("=========================================\n")
        print(f"Running KMedoid on {n_samples} contigs\n")
        print("=========================================\n")

        progress_bar = tqdm(total=max_iter, desc="Clusters created K-mediod")

        while torch.any(predictions == -1):
            cluster_id += 1
            if cluster_id > max_iter:
                break

            # Select highest density point index
            medoid_idx = torch.argmax(density_vector).item()
            density_vector[medoid_idx] = -100  # exclude seed from density vector

            seed = embeddings[medoid_idx]
            available_mask = predictions == -1  # points that are still available

            for _ in range(num_steps):
                similarities = torch.mv(embeddings, seed)
                candidate_mask = (similarities >= min_similarity) & available_mask
                candidates = torch.where(candidate_mask)[0]

                if len(candidates) == 0:
                    break

                seed = torch.mean(embeddings[candidates], dim=0)  # update seed

            if len(candidates) == 0:
                cluster_id -= 1  # Rollback unused cluster ID
                continue

            predictions[candidates] = cluster_id

            # Update density vector
            cluster_embs = embeddings[candidates]
            cluster_sims = torch.mm(embeddings, cluster_embs.T)
            cluster_sims = torch.where(
                cluster_sims >= min_similarity, cluster_sims, 0.0
            )
            density_vector -= torch.sum(cluster_sims, dim=1)
            density_vector[candidates] = -100  # exclude seed from density vector

            progress_bar.update(1)
            if cluster_id % 20 == 0:
                print(f"KMediod Step {cluster_id} completed.")

        # Filter small clusters
        labels, counts = torch.unique(predictions, return_counts=True)
        for label, count in zip(labels.cpu(), counts.cpu()):
            if label == -1 or count >= min_bin_size:
                continue
            predictions[predictions == label] = -1

        return predictions.cpu().numpy()
