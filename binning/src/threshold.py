import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from src.utils import get_available_device


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
        save_path: str,
        n_bins: int,
        block_size: int,
    ):
        self.check_params(embeddings, n_bins, block_size)

        device, gpu_count = get_available_device()
        embeddings = torch.from_numpy(embeddings).to(device)
        print(f"Using {device} for Threshold calculations")

        self.embeddings = embeddings
        self.n_bins = n_bins
        self.device = device
        self.block_size = block_size
        self.save_path = save_path

        self.bin_vector = self.similarity_bin_vector(self)

    def similarity_bin_vector(self) -> float:
        """
        Calculates pairwise similarities of embeddings and returns a histogram of similarities.
        """

        n_samples = self.embeddings.shape[0]
        bin_vector = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

        # loop through to get global min/max pairwise similarity
        global_min = torch.tensor([1], dtype=torch.float32, device=self.device)
        global_max = torch.tensor([0], dtype=torch.float32, device=self.device)
        for i in range(0, n_samples, self.block_size):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)
            local_min = block_sim_matrix.flatten().min()
            local_max = block_sim_matrix.flatten().max()
            global_min = torch.min(global_min, local_min)
            global_max = torch.max(global_max, local_max)

        # loop through again to get histogram
        for i in range(0, n_samples, self.block_size):
            block_start = i
            block_end = min(i + self.block_size, n_samples)
            block_embeddings = self.embeddings[block_start:block_end]

            block_sim_matrix = torch.mm(block_embeddings, self.embeddings.T)

            block_sim_flatten = block_sim_matrix.flatten()

            bin_vector += torch.histc(
                block_sim_flatten,
                bins=self.n_bins,
                min=global_min.item(),
                max=global_max.item(),
            )

        bin_vector = bin_vector / bin_vector.sum()
        bin_vector = bin_vector.cpu().numpy()

        return bin_vector


    def get_threshold(self) -> float:
        


    def save_histogram(self) -> None:
        """
        Plots and saves the histogram of similarities from the provided bin_vector.

        Parameters:
        - bin_vector: The normalized histogram values from similarity calculations.
        - output_dir: Directory where the plot will be saved.
        """

        plt.figure(figsize=(8, 6))
        plt.plot(
            np.arange(len(self.bin_vector)),
            self.bin_vector,
            color="skyblue",
            linestyle="-",
            linewidth=2,
        )
        plt.xlabel("Similarity Bins")
        plt.ylabel("Frequency")
        plt.title("Similarity Histogram")
        plt.xticks(np.arange(len(self.bin_vector)))

        # Save the figure
        file_path = os.path.join(self.save_path, "similarity_histogram.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        print(f"Plot saved at: {self.save_path}")
