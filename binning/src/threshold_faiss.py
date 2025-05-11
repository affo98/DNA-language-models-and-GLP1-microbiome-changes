import os
import json
import numpy as np
import faiss
import torch
import matplotlib.pyplot as plt
from src.utils import get_available_device, Logger, get_gpu_mem


class ThresholdFAISS:
    """
    Computes k-NN similarity threshold using FAISS for fast GPU-accelerated search,
    then finalizes histogram and saves outputs exactly like the original implementation.
    """

    def check_params(
        self,
        embeddings: np.ndarray,
        n_bins: int,
    ):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            print("Embeddings changed to dtype float32")
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
        self.check_params(embeddings, n_bins)

        # Setup
        device, _ = get_available_device()
        self.device = device
        self.log = log
        self.n_bins = n_bins
        self.block_size = block_size
        self.save_path = save_path
        self.model_name = model_name
        self.embeddings_np = embeddings  # store for later use

        # Build FAISS GPU index for inner product search
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()

        # IVF16384 + PQ48
        nlist = 16384  # Number of Voronoi cells
        M = 48  # Subquantizers (768/48=16)
        nbits = 8  # Bits per subquantizer

        quantizer = faiss.IndexFlatIP(d)  # <- Inner product matches cosine similarity
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        # Train on a subset (convert to float32 temporarily)
        gpu_index.train(embeddings[:10_000].astype(np.float32))

        # Add data in batches
        batch_size = 1_000_000
        for i in range(0, len(embeddings), batch_size):
            gpu_index.add(embeddings[i : i + batch_size])

        self.index = gpu_index
        self.log.append(
            f"FAISS IVF16384_PQ48 (IP quantizer) GPU index built: {embeddings.shape[0]} normalized vectors, "
            f"dim {d}, GPU mem: {get_gpu_mem()} MiB"
        )

        # flat index
        # cpu_index = faiss.IndexFlatIP(d)
        # self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        # self.index.add(embeddings)
        # self.N = embeddings.shape[0]
        # self.log.append(
        #     f"FAISS GPU index built: {self.N} normalized vectors of dim {d}, using MiB: {get_gpu_mem()}"
        # )

    def get_knn_threshold(self, knn_k: int, knn_p: float) -> float:
        """
        Query top-(k+1) neighbors (including self) via FAISS in batches,
        then compute similarity of each neighbor to the centroid of its k-NN.
        Assumes embeddings are already normalized.
        """
        self.knn_k = knn_k
        self.knn_p = knn_p

        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

        for start in range(0, self.N, self.block_size):
            end = min(start + self.block_size, self.N)
            queries = self.index.reconstruct_n(start, end - start)

            # Get top-k+1 (self + k neighbors)
            distances, indices = self.index.search(queries, knn_k + 1)
            neighbor_ids = indices[:, 1:]  # (block_size, k)

            topk_embs = (
                torch.from_numpy(self.embeddings_np[neighbor_ids])
                .to(self.device)
                .half()
            )  # (block_size, k, D)

            centroids = topk_embs.mean(dim=1, keepdim=True).transpose(
                1, 2
            )  # (bs, D, 1)
            csims = (
                torch.bmm(topk_embs, centroids).squeeze(-1).float().flatten()
            )  # (bs * k,)

            bin_vector += torch.histc(csims, bins=self.n_bins, min=0.0, max=1.0)

        bin_vector /= bin_vector.sum()
        bin_vector = bin_vector.cpu().numpy()

        pairsim_vector = (
            torch.linspace(float(0), float(1), steps=self.n_bins).cpu().numpy()
        )

        cumulative_sum = np.cumsum(bin_vector)
        index = np.argmax(cumulative_sum >= (knn_p / 100))
        knn_threshold = pairsim_vector[index]

        self.knn_threshold = knn_threshold
        self.pairsim_vector = pairsim_vector
        self.bin_vector = bin_vector

        self.save_histogram(knn=True)
        self.save_to_json()

        torch.cuda.empty_cache()
        return self.knn_threshold

    def save_to_json(self) -> None:
        """Saves the knn_threshold, pairsim_vector, and bin_vector to a JSON file."""
        data = {
            "knn_threshold": float(self.knn_threshold),
            "pairsim_vector": self.pairsim_vector.tolist(),
            "bin_vector": self.bin_vector.tolist(),
        }
        file_path = os.path.join(
            self.save_path, f"k{self.knn_k}_p{self.knn_p}_similarity_histogram.json"
        )
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        self.log.append(f"Threshold data saved to: {file_path}")

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
