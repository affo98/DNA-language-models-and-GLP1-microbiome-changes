#!/usr/bin/env python3
import numpy as np
import torch
import subprocess
import sys
import os
from tqdm import tqdm


def get_gpu_mem():
    """Return current GPU memory usage in MiB (as int)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        # if multiple GPUs, take the first line
        used = int(out.strip().split("\n")[0])
        return used
    except Exception as e:
        print("Warning: could not query nvidia-smi:", e, file=sys.stderr)
        return None


def to_fp16_tensor(
    embeddings: np.ndarray,
    chunk_size: int = 5_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Convert a large NumPy array or memmap of shape (N, D) to a single
    torch.cuda.FloatTensor in float16, streaming in chunk_size rows at a time.

    Returns:
        emb_fp16 (torch.Tensor): the full (N, D) tensor on GPU in float16.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a NumPy ndarray or memmap")

    N, D = embeddings.shape
    print(f"Using device: {device}")
    print(f"[Before allocation] GPU mem used: {get_gpu_mem()} MiB")

    # Allocate the target FP16 tensor on GPU
    emb_fp16 = torch.empty((N, D), dtype=torch.float16, device=device)
    print(f"[After empty alloc] GPU mem used: {get_gpu_mem()} MiB")

    # Stream in chunk_size rows at a time
    for start in tqdm(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        slice_np = embeddings[start:end]
        if not slice_np.flags.writeable:
            slice_np = slice_np.copy()
        block_cpu = torch.from_numpy(slice_np)  # CPU float32
        emb_fp16[start:end].copy_(block_cpu.to(device).half())  # GPU float16
        del block_cpu, slice_np

    # Wait for all copies to finish
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"[After streaming] GPU mem used: {get_gpu_mem()} MiB")

    return emb_fp16


def main():
    # --- user settings: adjust as needed ---
    embeddings_file = "embeddings.npy"
    N, D = 29_458_443, 768  # number of embeddings × dim
    chunk_size = 5_000  # rows per write/load chunk
    block_size = 20  # rows in test matmul

    print(f"[Before any allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 1) Generate embeddings.npy if missing, in a memory-safe way#
    if not os.path.exists(embeddings_file):
        print(f"Generating {embeddings_file} with shape ({N},{D}) ...")
        mm = np.lib.format.open_memmap(
            embeddings_file, mode="w+", dtype=np.float32, shape=(N, D)
        )
        for start in tqdm(range(0, N, chunk_size), desc="Writing chunks"):
            end = min(start + chunk_size, N)
            mm[start:end] = np.random.randn(end - start, D).astype(np.float32)
        # flush & close
        del mm
        print("Done writing embeddings.npy")

    # 2) Memory-map the .npy as float32, read-only
    embeddings_mm = np.load(embeddings_file, mmap_mode="r")
    assert embeddings_mm.shape == (N, D), "Shape mismatch loading memmap!"

    embeddings = to_fp16_tensor(embeddings_mm, chunk_size=chunk_size)

    # 4) do a single block-wise matmul to test peak memory
    block = embeddings[:block_size]  # shape (block_size, D)
    torch.cuda.synchronize()
    before_mm = get_gpu_mem()

    sim = torch.mm(block, embeddings.T)  # big operation in FP16
    torch.cuda.synchronize()
    after_mm = get_gpu_mem()

    print(f"[Before matmul] GPU memory used: {before_mm} MiB")
    print(f"[After  matmul] GPU memory used: {after_mm} MiB")
    print("sim.shape:", sim.shape)
    print("Done. If after-mm ≲ total_capacity you’re all set to proceed.")


if __name__ == "__main__":
    main()
