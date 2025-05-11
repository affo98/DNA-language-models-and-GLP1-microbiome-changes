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
        self.save_path = save_path
        self.model_name = model_name

        # Build FAISS GPU index for inner product search
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(d)
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.index.add(embeddings)
        self.N = embeddings.shape[0]
        self.log.append(
            f"FAISS GPU index built: {self.N} normalized vectors of dim {d}"
        )

    def get_knn_threshold(self, knn_k: int, knn_p: float) -> float:
        """
        Query top-(k+1) neighbors (including self) via FAISS,
        then finalize histogram and threshold exactly like the original code.
        """
        self.knn_k = knn_k
        self.knn_p = knn_p

        # Step 1: retrieve top-(k+1) for all points
        queries = self.index.reconstruct_n(0, self.N)
        distances, _ = self.index.search(queries, knn_k + 1)

        # Step 2: drop self-match
        sims_all = distances[:, 1:].reshape(-1)

        # compute histogram
        bin_vector = torch.tensor(
            np.histogram(
                sims_all,
                bins=self.n_bins,
                range=(float(sims_all.min()), float(sims_all.max())),
            )[0],
            dtype=torch.float32,
            device=self.device,
        )
        bin_vector /= bin_vector.sum()
        bin_vector = bin_vector.cpu().numpy()

        pairsim_vector = (
            torch.linspace(
                float(sims_all.min()), float(sims_all.max()), steps=self.n_bins
            )
            .cpu()
            .numpy()
        )

        cumulative_sum = np.cumsum(bin_vector)
        index = np.argmax(cumulative_sum >= (knn_p / 100))
        knn_threshold = pairsim_vector[index]

        self.knn_threshold = knn_threshold
        self.pairsim_vector = pairsim_vector
        self.bin_vector = bin_vector

        # use original save methods
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
