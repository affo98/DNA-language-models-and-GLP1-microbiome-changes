"""
This script is a test run for the binning process using KMediod and Threshold classes using randomly generated embeddings.
"""

#!/usr/bin/env python3
import numpy as np
import sys
import os
from tqdm import tqdm

from sklearn.preprocessing import normalize


sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from src.utils import Logger, get_gpu_mem
from src.clustering import KMediod
from src.threshold import Threshold


def main():
    save_path = "./binning_testrun/"
    os.makedirs(save_path, exist_ok=True)

    embeddings_file = "cami2_benchmark/model_results/metahit/dnaberts_output/test/embeddings/embeddings.npy"
    # embeddings_file = f"{save_path}embeddings.npy"

    N, D = 29_458_443, 768  # number of embeddings × dim #size of T2D-EW contigs
    chunk_size = 5_000  # rows per write/load chunk
    N_BINS = 1000
    BLOCK_SIZE = 100000
    model_name = "binning_testrun"
    log = Logger(os.path.join(save_path, "log.txt"))
    MIN_BIN_SIZE = 2  # changed from 10 to 2, because bins less than MINSIZE_BINS (250000) will be removed in postprocessing.
    NUM_STEPS = 3
    MAX_ITER = 2000  # increased from 1000
    KNN_K = 400
    KNN_P = 25
    CONVERT_MILLION_EMB_GPU_SECONDS = 6

    log.append(f"[Before any allocation] GPU memory used: {get_gpu_mem()} MiB")

    # generate normalized embeddings
    # if not os.path.exists(embeddings_file):
    #     log.append(f"Generating {embeddings_file} with shape ({N},{D}) ...")
    #     mm = np.lib.format.open_memmap(
    #         embeddings_file, mode="w+", dtype=np.float32, shape=(N, D)
    #     )

    #     for start in tqdm(range(0, N, chunk_size), desc="Writing chunks"):
    #         end = min(start + chunk_size, N)
    #         # Generate random embeddings for the current chunk
    #         embeddings_chunk = np.random.randn(end - start, D).astype(np.float32)
    #         embeddings_chunk = normalize(embeddings_chunk)

    #         mm[start:end] = embeddings_chunk

    #     # Flush & close
    #     del mm
    #     log.append("Done writing embeddings.npy")

    # embeddings_mm = np.memmap(
    #     embeddings_file,
    #     dtype="float32",
    #     mode="r",
    #     shape=(N, D),
    # )
    # assert embeddings_mm.shape == (N, D), "Shape mismatch loading memmap!"

    embeddings_mm = np.load(embeddings_file, allow_pickle=True)
    N = embeddings_mm.shape[0]
    # embeddings_mm = embeddings_mm_file["embeddings"]

    # Create contig names
    N = embeddings_mm.shape[0]
    contig_names_test = np.array(
        [f"contig_{i:06d}" for i in range(N)],
        dtype="<U12",  # enough width for your numbering
    )

    # ------- Test heavy memory operations -------
    embeddings_test = embeddings_mm
    log.append(f"[After allocation of memmap] GPU memory used: {get_gpu_mem()} MiB")
    log.append(f"Running Testrun with {embeddings_test.shape[0]} embeddings")

    # embeddings_test = embeddings_test[:5_000_000]
    # contig_names_test = contig_names_test[:5_000_000]

    # thresholder_test = Threshold(
    #     embeddings_test,
    #     N_BINS,
    #     BLOCK_SIZE,
    #     save_path,
    #     model_name,
    #     log,
    #     CONVERT_MILLION_EMB_GPU_SECONDS,
    # )
    # threshold = thresholder_test.get_knn_threshold(KNN_K, KNN_P)

    threshold = 0.749390721321106

    kmediod_test = KMediod(
        embeddings_test,
        contig_names_test,
        save_path,
        log,
        True,
        "test",
        CONVERT_MILLION_EMB_GPU_SECONDS,
        MIN_BIN_SIZE,
        NUM_STEPS,
        MAX_ITER,
        BLOCK_SIZE,
    )
    _, _ = kmediod_test.fit(threshold, KNN_K, KNN_P)


if __name__ == "__main__":
    main()
