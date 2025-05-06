#!/usr/bin/env python3
import numpy as np
import torch
import subprocess
import sys


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
    # --- adjust these for your real embeddings file ---
    embeddings_file = "embeddings.dat"  # raw float32 🌐
    N, D = 29_458_443, 768  # rows, cols
    chunk_size = 5_000  # how many rows to convert at once
    block_size = 20  # small block for test matmul

    print(f"[Before any allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 1) memory-map the on-disk embeddings as float32, read-only
    embeddings_mm = np.memmap(
        embeddings_file,
        dtype=np.float32,
        mode="r",
        shape=(N, D),
    )

    # 2) allocate an empty FP16 tensor of the right shape on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
