import time, torch
from transformers import AutoModel, AutoTokenizer, BertConfig


import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-S",
    padding_side="right",
    trust_remote_code=True,
    use_fast=True,
    padding="max_length",
)

config = BertConfig.from_pretrained(
    "zhihan1996/DNABERT-S",
)

model = (
    AutoModel.from_pretrained(
        "zhihan1996/DNABERT-S",
        config=config,
        trust_remote_code=True,
    )
    .eval()
    .to("cuda")
)


# for seqlen in [2000, 5000, 1000, 10000, 60000]:
#     sequence = "A" * seqlen  # example length
#     inputs = tokenizer([sequence] * 1, return_tensors="pt", padding=True).to("cuda")

#     for bs in [8, 16, 32, 64, 128]:
#         inputs_expanded = {k: v.repeat(bs, 1) for k, v in inputs.items()}
#         # warmup
#         for _ in range(5):
#             _ = model(**inputs_expanded)
#         # timed run
#         start = time.time()
#         _ = model(**inputs_expanded)
#         torch.cuda.synchronize()
#         elapsed = time.time() - start
#         tokens = bs * inputs_expanded["input_ids"].shape[1]
#         print(f"Batch {bs}, seq-len {seqlen}: {tokens/elapsed:.1f} tokens/sec")


seq_lens = [2000, 5000, 1000, 10000, 60000]

# Batch sizes: 2, 4, 6, … up to 128
batch_sizes = list(range(2, 130, 2))

for seqlen in seq_lens:
    sequence = "A" * seqlen
    inputs = tokenizer([sequence], return_tensors="pt", padding=True).to("cuda")
    print(f"\n=== Sequence length: {seqlen} ===")

    for bs in batch_sizes:
        # Expand inputs to current batch size
        inputs_expanded = {k: v.repeat(bs, 1) for k, v in inputs.items()}

        # Warm up
        try:
            for _ in range(3):
                _ = model(**inputs_expanded)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch {bs}: OOM on warmup, skipping.")
                torch.cuda.empty_cache()
                break
            else:
                raise

        # Timed run
        try:
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs_expanded)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            tokens = bs * inputs_expanded["input_ids"].shape[1]
            print(f"  Batch {bs}: {tokens/elapsed:.1f} tokens/sec")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch {bs}: OOM, skipping.")
                torch.cuda.empty_cache()
                break
            else:
                raise
