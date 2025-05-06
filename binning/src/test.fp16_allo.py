#!/usr/bin/env python3
import numpy as np
import torch
import subprocess
import sys
import os


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


def main():
    # --- user settings: adjust as needed ---
    embeddings_file = "embeddings.npy"
    N, D = 29_458_443, 768  # number of embeddings × dim
    chunk_size = 5_000  # rows per write/load chunk
    block_size = 20  # rows in test matmul

    print(f"[Before any allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 1) Generate embeddings.npy if missing, in a memory-safe way
    if not os.path.exists(embeddings_file):
        print(f"Generating {embeddings_file} with shape ({N},{D}) ...")
        mm = np.lib.format.open_memmap(
            embeddings_file, mode="w+", dtype=np.float32, shape=(N, D)
        )
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            mm[start:end] = np.random.randn(end - start, D).astype(np.float32)
        # flush & close
        del mm
        print("Done writing embeddings.npy")

    # 2) Memory-map the .npy as float32, read-only
    embeddings_mm = np.load(embeddings_file, mmap_mode="r")
    assert embeddings_mm.shape == (N, D), "Shape mismatch loading memmap!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    emb_fp16 = torch.empty((N, D), dtype=torch.float16, device=device)
    print(f"[After empty FP16 allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 3) fill the GPU tensor in chunks (never holds both full float32+float16 in VRAM)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # slice from memmap (this never loads everything into RAM at once)
        block_cpu = torch.from_numpy(embeddings_mm[start:end])
        # move and downcast
        emb_fp16[start:end].copy_(block_cpu.to(device).half())
        del block_cpu
    torch.cuda.synchronize()
    print(f"[After chunked conversion] GPU memory used: {get_gpu_mem()} MiB")

    # 4) do a single block-wise matmul to test peak memory
    block = emb_fp16[:block_size]  # shape (block_size, D)
    torch.cuda.synchronize()
    before_mm = get_gpu_mem()

    sim = torch.mm(block, emb_fp16.T)  # big operation in FP16
    torch.cuda.synchronize()
    after_mm = get_gpu_mem()

    print(f"[Before matmul] GPU memory used: {before_mm} MiB")
    print(f"[After  matmul] GPU memory used: {after_mm} MiB")
    print("sim.shape:", sim.shape)
    print("Done. If after-mm ≲ total_capacity you’re all set to proceed.")


if __name__ == "__main__":
    main()
