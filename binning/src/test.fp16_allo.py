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
    # --- adjust these for your real embeddings ---
    N, D = 29458443, 768  # e.g. 20k × 1k dims; ~80 MiB in float32
    chunk_size = 5000  # move 5k rows at a time
    block_size = 1000  # one test matmul block

    print(f"[Before any allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 1) create dummy embeddings in float32 on CPU
    embeddings_np = np.random.randn(N, D).astype(np.float32)

    # 2) allocate an empty FP16 tensor of the right shape on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_fp16 = torch.empty((N, D), dtype=torch.float16, device=device)

    print(f"[After empty FP16 allocation] GPU memory used: {get_gpu_mem()} MiB")

    # 3) fill it in chunks (never holds both full float32+float16 at once)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        block = torch.from_numpy(embeddings_np[start:end]).to(device)
        emb_fp16[start:end].copy_(block.half())
        # optionally delete block explicitly:
        del block
    torch.cuda.synchronize()
    print(f"[After chunked conversion] GPU memory used: {get_gpu_mem()} MiB")

    # 4) do a single block-wise matmul
    i0 = 0
    i1 = block_size
    block = emb_fp16[i0:i1]  # shape (block_size, D)
    torch.cuda.synchronize()
    before_mm = get_gpu_mem()

    sim = torch.mm(block, emb_fp16.T)  # this is the big operation
    torch.cuda.synchronize()
    after_mm = get_gpu_mem()

    print(f"[Before matmul] GPU memory used: {before_mm} MiB")
    print(f"[After  matmul] GPU memory used: {after_mm} MiB")

    # simple sanity-check on sim
    print("sim.shape:", sim.shape)
    print("Done. If after-mm ≲ total_capacity you’re all set to proceed.")


if __name__ == "__main__":
    main()
