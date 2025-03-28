import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import skimage.filters as filters

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

    def get_knn_threshold(self, knn_k, knn_p) -> float:

        self.knn_k = knn_k
        self.knn_p = knn_p

        n_samples = self.embeddings.shape[0]
        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

        global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
        global_max = torch.tensor([-1], dtype=torch.float32, device=self.device)
        for i in tqdm(
            range(0, n_samples, self.block_size), desc="Calculating global min/max"
        ):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
            top_k_similarities, _ = torch.topk(block_sim_matrix, self.knn_k, dim=-1)
            top_k_similarities_flat = top_k_similarities.flatten()

            global_min = torch.min(global_min, top_k_similarities_flat.min())
            global_max = torch.max(global_max, top_k_similarities_flat.max())

        # loop through again to get histogram
        for i in tqdm(range(0, n_samples, self.block_size), desc="Calculating knns"):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
            top_k_similarities, _ = torch.topk(block_sim_matrix, self.knn_k, dim=-1)
            top_k_similarities_flat = top_k_similarities.flatten()

            bin_vector += torch.histc(
                top_k_similarities_flat,
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
        print(global_min.item(), global_max.item())

        cumulative_sum = np.cumsum(bin_vector)
        index = np.argmax(cumulative_sum >= self.knn_p)
        knn_threshold = pairsim_vector[index]

        self.knn_threshold, self.pairsim_vector, self.bin_vector = (
            knn_threshold,
            pairsim_vector,
            bin_vector,
        )

        return knn_threshold, pairsim_vector, bin_vector

    def save_histogram(self, knn=True) -> None:
        """Plots and saves the histogram of similarities from the provided bin_vector."""

        plt.figure(figsize=(8, 6))

        if knn:
            plt.axvline(
                self.knn_threshold,
                color="g",
                linestyle="--",
                label=f"KNN Threshold: {self.knn_threshold} (k={self.knn_k}, p={int(self.knn_p*100)})",
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
            "histograms",
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

    #     n_samples = self.embeddings.shape[0]
    #     bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

    #     # loop through to get global min/max pairwise similarity
    #     global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
    #     global_max = torch.tensor([-1], dtype=torch.float32, device=self.device)
    #     for i in tqdm(
    #         range(0, n_samples, self.block_size), desc="Calculating global min/max"
    #     ):
    #         block_start = i
    #         block_end = min(i + self.block_size, n_samples)
    #         block_embeddings = self.embeddings[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
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
    #         block_embeddings = self.embeddings[block_start:block_end]

    #         block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)

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
