export PATH_TO_DATA_DICT=/path/to/data/dict # (e.g., /root/data/DNABERT_h_data)

# Multi-level Hierarchical Contrastive Learning (DNABERT-H)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_DEBUG=INFO \
NCCL_BLOCKING_WAIT=1 \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
TORCH_NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  main.py \
  --datapath PATH_TO_DATA_DICT \
  --train_dataname train_2m.tsv \
  --val_dataname val_40k.tsv \
  --batch_size 18