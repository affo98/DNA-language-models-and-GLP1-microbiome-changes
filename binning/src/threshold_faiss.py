import os
import json
from tqdm import tqdm
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
        if embeddings.dtype != np.float16:
            embeddings = embeddings.astype(np.float16)
            print("Embeddings changed to dtype float16")
        if n_bins < 1:
            raise ValueError("Number of bins must be at least 1")

    def build_faiss(self, IVF_index):
        d = self.embeddings_np.shape[1]
        self.log.append(f"N GPUs Faiss: {faiss.get_num_gpus()}")

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True

        if IVF_index == False:
            cpu_index = faiss.IndexFlatIP(d)
            index = faiss.index_cpu_to_all_gpus(cpu_index, co)
            index.add(self.embeddings_np)
            self.log.append(f"FLAT FAISS GPU index built using MiB: {get_gpu_mem()}")

        else:
            nlist = (
                4096  # Number of Voronoi cells/clusters (tune based on dataset size)
            )
            nprobe = 32  # Number of clusters to search (balance speed/accuracy)
            quantizer = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
            cpu_index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index = faiss.index_cpu_to_all_gpus(cpu_index, co)
            index.train(self.embeddings_np)
            index.add(self.embeddings_np)
            index.nprobe = nprobe

            self.log.append(
                f"IVF--FLAT FAISS GPU index built using MiB: {get_gpu_mem()}; nlist={nlist}, nprobe={nprobe}"
            )

        return index

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
        self.embeddings_np = embeddings
        self.N = embeddings.shape[0]

        self.index = self.build_faiss(IVF_index=True)

    def get_knn_threshold(self, knn_k: int, knn_p: float) -> float:
        """
        Query top-(k+1) neighbors (including self) via FAISS in batches,
        then compute similarity of each neighbor to the centroid of its k-NN.
        Assumes embeddings are already normalized.
        """
        self.knn_k = knn_k
        self.knn_p = knn_p

        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

        for i in tqdm(range(0, self.N, self.block_size), desc="Calculating Threshold"):
            i_end = min(i + self.block_size, self.N)
            batch = self.embeddings_np[i:i_end]

            # Search in FAISS index
            _, indices = self.index.search(batch, knn_k + 1)

            topk_embs = (
                torch.from_numpy(self.embeddings_np[indices]).to(self.device).half()
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
