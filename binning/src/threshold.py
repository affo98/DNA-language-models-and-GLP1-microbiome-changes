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
        convert_million_emb_gpu_seconds: int,
    ):
        self.check_params(embeddings, n_bins, block_size)

        device, gpu_count = get_available_device()

        # log.append(
        #     f"[Before embeddings GPU16 allocation]; {get_gpu_mem(log)} MIB"
        #     f"Converting embeddings to GPU16. Estimated time: "
        #     f"{embeddings.shape[0] / 1_000_000 * convert_million_emb_gpu_seconds:.1f} seconds",
        # )

        # embeddings_torch = torch.from_numpy(embeddings).half().to(device)
        # log.append(
        #     f"[After embedding GPU16 allocation] GPU mem used: {get_gpu_mem(log)} MiB"
        # )

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
            emb_i = torch.from_numpy(self.embeddings_np[i:i_end]).to(self.device)
            # compute similarities row-wise in j-chunks
            for j in range(0, n_samples, self.block_size):
                j_end = min(j + self.block_size, n_samples)
                emb_j = torch.from_numpy(self.embeddings_np[j:j_end]).to(self.device)
                sims_block = torch.mm(emb_i, emb_j.T).float()

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
            global_min = torch.min(global_min, csims.min())
            global_max = torch.max(global_max, csims.max())

            del topk_embs, centroids, csims, emb_i
            torch.cuda.empty_cache()

        # RM!!
        # global_min = torch.tensor(-1.0, dtype=torch.float32, device=self.device)
        # global_max = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        # ---------------- Histogram pass ----------------
        for i in tqdm(range(0, n_samples, self.block_size), desc="Calculating knns"):
            i_end = min(i + self.block_size, n_samples)
            emb_i = torch.from_numpy(self.embeddings_np[i:i_end]).to(self.device)

            # compute top-k indices per row same as above
            for j in range(0, n_samples, self.block_size):
                j_end = min(j + self.block_size, n_samples)
                emb_j = torch.from_numpy(self.embeddings_np[j:j_end]).to(self.device)
                sims_block = torch.mm(emb_i, emb_j.T).float()
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

    # def get_knn_threshold(self, knn_k, knn_p) -> float:

    #     self.knn_k = knn_k
    #     self.knn_p = knn_p

    #     n_samples = self.embeddings_np.shape[0]
    #     bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

    #     # first, find min/max
    #     global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
    #     global_max = torch.tensor([-1], dtype=torch.float32, device=self.device)
    #     for i in tqdm(
    #         range(0, n_samples, self.block_size), desc="Calculating global min/max"
    #     ):
    #         block_start = i
    #         block_end = min(i + self.block_size, n_samples)
    #         block_embeddings = self.embeddings_np[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings_np.T)
    #         top_k_similarities, top_k_indices = torch.topk(
    #             block_sim_matrix, self.knn_k, dim=-1
    #         )

    #         top_k_embeddings = self.embeddings_np[
    #             top_k_indices
    #         ]  # shape: (block_size, knn_k, embedding_dim)
    #         centroids = top_k_embeddings.mean(
    #             dim=1, keepdim=True
    #         )  # shape: (block_size, 1, embedding_dim)

    #         centroids = centroids.transpose(
    #             1, 2
    #         )  # Shape: (block_size, embedding_dim, 1)

    #         centroid_similarities = torch.bmm(top_k_embeddings, centroids).squeeze(-1)
    #         centroid_similarities_flat = centroid_similarities.flatten().to(
    #             dtype=torch.float32
    #         )

    #         global_min = torch.min(global_min, centroid_similarities_flat.min())
    #         global_max = torch.max(global_max, centroid_similarities_flat.max())

    #     # loop through again to get histogram
    #     for i in tqdm(range(0, n_samples, self.block_size), desc="Calculating knns"):
    #         block_start = i
    #         block_end = min(i + self.block_size, n_samples)
    #         block_embeddings = self.embeddings_np[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings_np.T)
    #         top_k_similarities, top_k_indices = torch.topk(
    #             block_sim_matrix, self.knn_k, dim=-1
    #         )

    #         top_k_embeddings = self.embeddings_np[
    #             top_k_indices
    #         ]  # shape: (block_size, knn_k, embedding_dim)
    #         centroids = top_k_embeddings.mean(
    #             dim=1, keepdim=True
    #         )  # shape: (block_size, 1, embedding_dim)
    #         centroids = centroids.transpose(
    #             1, 2
    #         )  # Shape: (block_size, embedding_dim, 1)

    #         centroid_similarities = torch.bmm(top_k_embeddings, centroids).squeeze(-1)
    #         centroid_similarities_flat = centroid_similarities.flatten().to(
    #             dtype=torch.float32
    #         )

    #         bin_vector += torch.histc(
    #             centroid_similarities_flat,
    #             bins=self.n_bins,
    #             min=global_min.item(),
    #             max=global_max.item(),
    #         )

    #     bin_vector = bin_vector / bin_vector.sum()
    #     bin_vector = bin_vector.cpu().numpy()

    #     pairsim_vector = (
    #         torch.linspace(global_min.item(), global_max.item(), self.n_bins)
    #         .cpu()
    #         .numpy()
    #     )

    #     cumulative_sum = np.cumsum(bin_vector)
    #     index = np.argmax(cumulative_sum >= (self.knn_p / 100))
    #     knn_threshold = pairsim_vector[index]

    #     self.knn_threshold, self.pairsim_vector, self.bin_vector = (
    #         knn_threshold,
    #         pairsim_vector,
    #         bin_vector,
    #     )

    #     self.save_histogram(knn=True)
    #     self.save_to_json()

    #     # cleanup
    #     del self.embeddings_np
    #     del embeddings_torch
    #     torch.cuda.empty_cache()

    #     return knn_threshold

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

    #         def get_similarity_bin_vector(self) -> float:
    #     """
    #     Calculates pairwise similarities of embeddings and returns a histogram of similarities.
    #     """

    #     n_samples = self.embeddings_np.shape[0]
    #     bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

    #     # loop through to get global min/max pairwise similarity
    #     global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
    #     global_max = torch.tensor([-1], dtype=torch.float32, device=self.device)
    #     for i in tqdm(
    #         range(0, n_samples, self.block_size), desc="Calculating global min/max"
    #     ):
    #         block_start = i
    #         block_end = min(i + self.block_size, n_samples)
    #         block_embeddings = self.embeddings_np[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings_np.T)
    #         local_min = block_sim_matrix.flatten().min()
    #         local_max = block_sim_matrix.flatten().max()
    #         global_min = torch.min(global_min, local_min)
    #         global_max = torch.max(global_max, local_max)

    #     # loop through again to get histogram
    #     for i in tqdm(
    #         range(0, n_samples, self.block_size), desc="Calculating histogram"
    #     ):
    #         block_start = i
    #         block_end = min(i + self.block_size, n_samples)
    #         block_embeddings = self.embeddings_np[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings_np.T)

    #         block_sim_flatten = block_sim_matrix.flatten()

    #         bin_vector += torch.histc(
    #             block_sim_flatten,
    #             bins=self.n_bins,
    #             min=global_min.item(),
    #             max=global_max.item(),
    #         )

    #     bin_vector = bin_vector / bin_vector.sum()
    #     bin_vector = bin_vector.cpu().numpy()

    #     pairsim_vector = (
    #         torch.linspace(global_min.item(), global_max.item(), self.n_bins)
    #         .cpu()
    #         .numpy()
    #     )
    #     print(global_min.item(), global_max.item())

    #     return bin_vector, pairsim_vector

    # def get_threshold(self) -> tuple[float, float, float, float, float]:

    #     otsu = filters.threshold_otsu(self.bin_vector[:980])
    #     otsu_mul = filters.threshold_multiotsu(self.bin_vector, classes=3)
    #     isodata = filters.threshold_isodata(self.bin_vector)
    #     minimum = filters.threshold_minimum(self.bin_vector)
    #     yen = filters.threshold_yen(self.bin_vector)

    #     return (otsu, otsu_mul, isodata, minimum, yen)
    # plt.axvline(
    #     x=np.argmin(np.abs(self.pairsim_vector - self.otsu)),
    #     color="r",
    #     linestyle="--",
    #     label=f"Otsu (t={self.otsu:.5f})",
    # )
    # plt.axvline(
    #     x=np.argmin(np.abs(self.pairsim_vector - self.otsu_mul[0])),
    #     color="indianred",
    #     linestyle="--",
    #     label=f"MULTIPLE OTSU (t={self.otsu_mul[0]:.5f})",
    # )
    # plt.axvline(
    #     x=np.argmin(np.abs(self.pairsim_vector - self.isodata)),
    #     color="b",
    #     linestyle="--",
    #     label=f"ISODATA  (t={self.isodata:.5f})",
    # )
    # plt.axvline(
    #     x=np.argmin(np.abs(self.pairsim_vector - self.minimum)),
    #     color="y",
    #     linestyle="--",
    #     label=f"MINIMUM (t={self.minimum:.5f})",
    # )
    # plt.axvline(
    #     x=np.argmin(np.abs(self.pairsim_vector - self.yen)),
    #     color="slategrey",
    #     linestyle="--",
    #     label=f"YEN Threshold (t={self.yen:.5f})",
    # )
    # plt.axvline(
    #     x=350 + self.pairsim_vector[350:700].argmin(),
    #     color="k",
    #     linestyle="--",
    #     label=f"Manual: {self.pairsim_vector[self.pairsim_vector[350:700].argmin()]:.5f}",
    # )

    # NORMALPDF = 0.005 * torch.tensor(
    #     [
    #         2.43432053e-11,
    #         9.13472041e-10,
    #         2.66955661e-08,
    #         6.07588285e-07,
    #         1.07697600e-05,
    #         1.48671951e-04,
    #         1.59837411e-03,
    #         1.33830226e-02,
    #         8.72682695e-02,
    #         4.43184841e-01,
    #         1.75283005e00,
    #         5.39909665e00,
    #         1.29517596e01,
    #         2.41970725e01,
    #         3.52065327e01,
    #         3.98942280e01,
    #         3.52065327e01,
    #         2.41970725e01,
    #         1.29517596e01,
    #         5.39909665e00,
    #         1.75283005e00,
    #         4.43184841e-01,
    #         8.72682695e-02,
    #         1.33830226e-02,
    #         1.59837411e-03,
    #         1.48671951e-04,
    #         1.07697600e-05,
    #         6.07588285e-07,
    #         2.66955661e-08,
    #         9.13472041e-10,
    #         2.43432053e-11,
    #     ],
    #     device=self.device,
    # )
    # pdf_len = len(NORMALPDF)
    # densities = torch.zeros(len(bin_vector) + pdf_len - 1, device=self.device)
    # for i in range(len(densities) - pdf_len + 1):
    #     densities[i : i + pdf_len] += NORMALPDF * bin_vector[i]
    # densities = densities[15:-15]
    # densities = densities.to("cpu").numpy()

    # return densities, global_min.item(), global_max.item()
