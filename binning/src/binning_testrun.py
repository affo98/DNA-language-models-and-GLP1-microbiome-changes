"""
This script is a test run for the binning process using KMediod and Threshold classes using randomly generated embeddings.
"""

#!/usr/bin/env python3
import numpy as np
import sys
import os
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from src.utils import Logger, get_gpu_mem
from src.clustering import KMediod
from src.threshold import Threshold


def main():
    save_path = "./binning_testrun/"
    os.makedirs(save_path, exist_ok=True)
    embeddings_file = f"{save_path}embeddings.npy"
    N, D = 29_458_443, 768  # number of embeddings × dim
    chunk_size = 5_000  # rows per write/load chunk
    N_BINS = 1000
    BLOCK_SIZE = 20
    model_name = "binning_testrun"
    log = Logger(os.path.join(save_path, "log.txt"))
    MIN_BIN_SIZE = 2  # changed from 10 to 2, because bins less than MINSIZE_BINS (250000) will be removed in postprocessing.
    NUM_STEPS = 3
    MAX_ITER = 2000  # increased from 1000
    KNN_K = 300
    KNN_P = 25

    log.append(f"[Before any allocation] GPU memory used: {get_gpu_mem(log)} MiB")

    # 1) Generate embeddings.npy if missing, in a memory-safe way#
    if not os.path.exists(embeddings_file):
        log.append(f"Generating {embeddings_file} with shape ({N},{D}) ...")
        mm = np.lib.format.open_memmap(
            embeddings_file, mode="w+", dtype=np.float32, shape=(N, D)
        )
        for start in tqdm(range(0, N, chunk_size), desc="Writing chunks"):
            end = min(start + chunk_size, N)
            mm[start:end] = np.random.randn(end - start, D).astype(np.float32)
        # flush & close
        del mm
        log.append("Done writing embeddings.npy")

    # Load embeddings
    embeddings_mm = np.load(embeddings_file, mmap_mode="r")
    assert embeddings_mm.shape == (N, D), "Shape mismatch loading memmap!"

    # Create contig names
    N = embeddings_mm.shape[0]
    contig_names_test = np.array(
        [f"contig_{i:06d}" for i in range(N)],
        dtype="<U12",  # enough width for your numbering
    )

    # ------- Test heavy memory operations -------
    embeddings_test = embeddings_mm
    log.append(f"[After allocation] GPU memory used: {get_gpu_mem(log)} MiB")

    thresholder_test = Threshold(
        embeddings_test,
        N_BINS,
        BLOCK_SIZE,
        save_path,
        model_name,
        log,
    )

    kmediod_test = KMediod(
        embeddings_test,
        contig_names_test,
        save_path,
        log,
        True,
        "test",
        MIN_BIN_SIZE,
        NUM_STEPS,
        MAX_ITER,
        BLOCK_SIZE,
    )

    threshold = thresholder_test.get_knn_threshold(KNN_K, KNN_P)
    _, _ = kmediod_test.fit(threshold, KNN_K, KNN_P)


if __name__ == "__main__":
    main()
