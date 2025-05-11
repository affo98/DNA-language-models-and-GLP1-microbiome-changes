"""
This script is a test run for the binning process using KMediod and Threshold classes using randomly generated embeddings.
"""

#!/usr/bin/env python3
import numpy as np
import sys
import os
from tqdm import tqdm
import time
import torch


from sklearn.preprocessing import normalize


sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from src.utils import Logger, get_gpu_mem
from src.clustering import KMediod
from src.clustering_faiss import KMediodFAISS
from src.threshold import Threshold
from src.threshold_faiss import ThresholdFAISS


def main():
    save_path = "./binning_testrun/"
    os.makedirs(save_path, exist_ok=True)

    # embeddings_file = "cami2_benchmark/model_results/metahit/dnaberts_output/test/embeddings/embeddings.npy"
    embeddings_file = f"{save_path}embeddings.npy"

    N, D = 29_458_443, 768  # number of embeddings × dim #size of T2D-EW contigs
    chunk_size = 5_000  # rows per write/load chunk
    N_BINS = 1000
    BLOCK_SIZE = 10_000
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

    embeddings_mm = np.memmap(
        embeddings_file,
        dtype="float32",
        mode="r",
        shape=(N, D),
    )
    # embeddings_mm = np.load(embeddings_file)

    # n_test = 161581
    # embeddings_mm = np.memmap(
    #     embeddings_file,
    #     dtype="float32",
    #     mode="r",
    #     shape=(n_test, 768),
    # )  # embeddings_array = np.array(embeddings)
    N = embeddings_mm.shape[0]

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

    # free_mem, _ = torch.cuda.mem_get_info()  # bytes available, e.g. ~2.3e10
    # margin = 0.9
    # M = free_mem * margin

    # D = embeddings_mm.shape[1]  # embedding dim
    # # solve b^2 + D*b - M/4 = 0 for b:
    # max_bs = int((-D + (D**2 + M) ** 0.5) / 2)

    # block_size = max_bs
    # # block_size = BLOCK_SIZE
    # print(f"free_mem={free_mem}, max_bs={max_bs}, chosen block_size={block_size}")

    # thresholder_test = Threshold(
    #     embeddings_test,
    #     N_BINS,
    #     block_size,
    #     save_path,
    #     model_name,
    #     log,
    #     CONVERT_MILLION_EMB_GPU_SECONDS,
    # )
    # threshold = thresholder_test.get_knn_threshold(KNN_K, KNN_P)

    thresholder_test = ThresholdFAISS(
        embeddings_test,
        N_BINS,
        BLOCK_SIZE,
        save_path,
        model_name,
        log,
    )

    start = time.perf_counter()
    threshold = thresholder_test.get_knn_threshold(KNN_K, KNN_P)
    elapsed = time.perf_counter() - start
    print(f"\n>> Total runtime Threshold: {elapsed:.2f} seconds")
    print(threshold)

    # threshold = 0.749390721321106

    kmediod_test = KMediodFAISS(
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
    start = time.perf_counter()
    _, _ = kmediod_test.fit(threshold, KNN_K, KNN_P)
    elapsed = time.perf_counter() - start
    print(f"\n>> Total runtime K-MEDOID: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
