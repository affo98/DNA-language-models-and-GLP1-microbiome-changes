import os
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from src.utils import get_available_device, Logger, get_gpu_mem


class Threshold:

    def check_params(
        self,
        embeddings: np.ndarray,
        n_bins: int,
        block_size: int,
    ):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            print("Embeddings changed to dtype float32")
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

        self.embeddings_np = embeddings
        self.n_bins = n_bins
        self.block_size = block_size
        self.save_path = save_path
        self.model_name = model_name
        self.log = log
        self.device = device

        self.log.append(f"Using {device} for Threshold calculations")

    def get_knn_threshold(self, knn_k, knn_p) -> float:
        self.knn_k = knn_k
        self.knn_p = knn_p

        n_samples, _ = self.embeddings_np.shape
        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)
        global_min = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        global_max = torch.tensor(-1.0, dtype=torch.float32, device=self.device)

        # ---------------- Global min/max pass ----------------
        for i in tqdm(
            range(0, n_samples, self.block_size), desc="Calculating global min/max"
        ):
            i_end = min(i + self.block_size, n_samples)
            emb_i = torch.from_numpy(self.embeddings_np[i:i_end]).to(self.device).half()
            # compute similarities row-wise in j-chunks
            for j in range(0, n_samples, self.block_size):
                j_end = min(j + self.block_size, n_samples)
                emb_j = (
                    torch.from_numpy(self.embeddings_np[j:j_end]).to(self.device).half()
                )
                sims_block = torch.mm(emb_i, emb_j.T)

                # collect full row sims in memory-chunked fashion
                # for each row in this block, get top-k indices across all j-chunks
                # accumulate partial topk per row
                if j == 0:
                    partial_vals, partial_idx = torch.topk(
                        sims_block, self.knn_k, dim=1
                    )
                else:
                    vals, idx = torch.topk(sims_block, self.knn_k, dim=1)
                    combined_vals = torch.cat([partial_vals, vals], dim=1)
                    combined_idx = torch.cat([partial_idx, idx + j], dim=1)
                    partial_vals, sel = torch.topk(combined_vals, self.knn_k, dim=1)
                    partial_idx = combined_idx.gather(1, sel)
                del sims_block, emb_j
                torch.cuda.empty_cache()

            # now partial_idx holds top-k global neighbor indices for each emb_i row
            topk_embs = torch.from_numpy(
                self.embeddings_np[partial_idx.cpu().numpy()]
            ).to(
                self.device
            )  # shape (bs_i, knn_k, D)
            centroids = topk_embs.mean(dim=1, keepdim=True).transpose(
                1, 2
            )  # (bs_i, 1, D)→(bs_i, D, 1)
            csims = torch.bmm(topk_embs, centroids).squeeze(-1).float().flatten()
            global_min = torch.min(global_min, csims.min()).float()
            global_max = torch.max(global_max, csims.max()).float()

            del topk_embs, centroids, csims, emb_i
            torch.cuda.empty_cache()

        # ---------------- Histogram pass ----------------
        for i in tqdm(range(0, n_samples, self.block_size), desc="Calculating knns"):
            i_end = min(i + self.block_size, n_samples)
            emb_i = torch.from_numpy(self.embeddings_np[i:i_end]).to(self.device).half()

            # compute top-k indices per row same as above
            for j in range(0, n_samples, self.block_size):
                j_end = min(j + self.block_size, n_samples)
                emb_j = (
                    torch.from_numpy(self.embeddings_np[j:j_end]).to(self.device).half()
                )
                sims_block = torch.mm(emb_i, emb_j.T)
                if j == 0:
                    partial_vals, partial_idx = torch.topk(
                        sims_block, self.knn_k, dim=1
                    )
                else:
                    vals, idx = torch.topk(sims_block, self.knn_k, dim=1)
                    combined_vals = torch.cat([partial_vals, vals], dim=1)
                    combined_idx = torch.cat([partial_idx, idx + j], dim=1)
                    partial_vals, sel = torch.topk(combined_vals, self.knn_k, dim=1)
                    partial_idx = combined_idx.gather(1, sel)
                del sims_block, emb_j
                torch.cuda.empty_cache()

            topk_embs = torch.from_numpy(
                self.embeddings_np[partial_idx.cpu().numpy()]
            ).to(
                self.device
            )  # shape (bs_i, knn_k, D)
            centroids = topk_embs.mean(dim=1, keepdim=True).transpose(
                1, 2
            )  # (bs_i, 1, D)→(bs_i, D, 1)

            csims = torch.bmm(topk_embs, centroids).squeeze(-1).float().flatten()

            bin_vector += torch.histc(
                csims,
                bins=self.n_bins,
                min=global_min.item(),
                max=global_max.item(),
            )
            del topk_embs, centroids, csims, emb_i
            torch.cuda.empty_cache()

        # finalize histogram
        bin_vector /= bin_vector.sum()
        bin_vector = bin_vector.cpu().numpy()
        pairsim_vector = (
            torch.linspace(
                global_min.item(),
                global_max.item(),
                steps=self.n_bins,
            )
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

        torch.cuda.empty_cache()
        return self.knn_threshold

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
