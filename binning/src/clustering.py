import os
import json

import torch
import numpy as np
from tqdm import tqdm


from src.utils import get_available_device, Logger, get_gpu_mem


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

        self.embeddings_np = embeddings
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

        self.log.append(f"Using {device} for K-medoid Clustering")

    def fit(self, min_similarity: float, knn_k: int, knn_p: float) -> np.array:
        if not 0 < min_similarity < 1:
            raise ValueError("Minimum similarity must be between 0 and 1")

        N, _ = self.embeddings_np.shape
        self.log.append("=========================================\n")
        self.log.append(f"Running KMedoid on {N} contigs\n")
        self.log.append("=========================================\n")
        self.log.append(
            f"Using {self.device} and threshold {min_similarity} for k-medoid clustering"
        )

        density_vector = torch.zeros(N, device=self.device)
        seeds = []
        seed_labels = []
        cluster_id = 0

        # compute initial density via 2D chunking
        for i in tqdm(range(0, N, self.block_size), desc="Density i-loop"):
            i_end = min(i + self.block_size, N)
            block_i_np = self.embeddings_np[i:i_end]
            block_i = torch.from_numpy(block_i_np).to(self.device)
            for j in range(0, N, self.block_size):
                j_end = min(j + self.block_size, N)
                block_j_np = self.embeddings_np[j:j_end]
                block_j = torch.from_numpy(block_j_np).to(self.device)

                sim = torch.mm(block_i, block_j.T)
                sim = torch.where(sim >= min_similarity, sim, torch.zeros_like(sim))
                density_vector[i:i_end] += sim.sum(dim=1)

                del sim, block_j
                torch.cuda.empty_cache()
            del block_i
            torch.cuda.empty_cache()

        predictions = torch.full((N,), -1, dtype=torch.long, device=self.device)
        pbar = tqdm(total=self.max_iter, desc="Clusters created K-medoid")

        while torch.any(predictions == -1) and cluster_id < self.max_iter:
            cluster_id += 1

            # select medoid
            medoid_idx = torch.argmax(density_vector).item()
            density_vector[medoid_idx] = -100  # rm seed contig
            seed_np = self.embeddings_np[medoid_idx]
            seed = torch.from_numpy(seed_np).to(self.device)
            available_mask = predictions == -1

            for _ in range(self.num_steps):
                # full embedding block-by-block for mv
                sims_full = []
                for i in range(0, N, self.block_size):
                    i_end = min(i + self.block_size, N)
                    block_np = self.embeddings_np[i:i_end]
                    block = torch.from_numpy(block_np).to(self.device)
                    sims_part = torch.mv(block, seed)
                    sims_full.append(sims_part)
                    del block, sims_part
                    torch.cuda.empty_cache()
                sim_vec = torch.cat(sims_full)
                del sims_full

                candidate_mask = (sim_vec >= min_similarity) & available_mask
                candidates = torch.where(candidate_mask)[0]

                if len(candidates) == 0:
                    break

                emb_candicates_np = self.embeddings_np[candidates.cpu().numpy()]
                emb_candidates = torch.from_numpy(emb_candicates_np).to(self.device)
                seed = torch.mean(emb_candidates, dim=0)

            predictions[candidates] = cluster_id
            seeds.append(seed.cpu().numpy())
            seed_labels.append(cluster_id)

            # decrement density via 2D chunking on candidates
            for i in range(0, N, self.block_size):
                i_end = min(i + self.block_size, N)
                block_i_np = self.embeddings_np[i:i_end]
                block_i = torch.from_numpy(block_i_np).to(self.device)

                for c0 in range(0, len(candidates), self.block_size):
                    c1 = min(len(candidates), c0 + self.block_size)
                    cand_np = self.embeddings_np[candidates[c0:c1].cpu().numpy()]
                    block_c = torch.from_numpy(cand_np).to(self.device)

                    sim = torch.mm(block_i, block_c.T)
                    sim = torch.where(sim >= min_similarity, sim, torch.zeros_like(sim))
                    density_vector[i:i_end] -= sim.sum(dim=1)

                    del sim, block_c
                    torch.cuda.empty_cache()
                del block_i
                torch.cuda.empty_cache()

            density_vector[candidates] = -100
            pbar.update(1)

        pbar.close()
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
            len(predictions) == len(self.embeddings_np) == len(self.contig_names)
        ), f"Len of predictions {len(predictions)} does not match embeddings {len(self.embeddings_np)} and contig_names {len(self.contig_names)}"

        predictions, contig_names = self.remove_unassigned_sequences(predictions)
        self.save_clusters(knn_k, knn_p, predictions, contig_names)

        # clean-up
        del density_vector, seed
        torch.cuda.empty_cache()

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
