# Pretraining

This directory contains the official implementation for DNABERT-H.

## Data Requirements

Our code expects DNA sequence pairs with taxonomic labels for both training and validation. A TSV file with a minimum of 10 columns is required, consisting of:
- 2 DNA sequences 
- 8 corresponding taxonomic labels for each row

Please refer to `./data/sample.tsv` for a reference.

## Setup Environment and Dependencies

Create and activate conda environment:
```bash
conda create -n DNABERT_H python=3.9
conda activate DNABERT_H
```

Install required packages and dependencies:
```bash
pip install -r requirements.txt
pip uninstall triton 
```

For multi-GPU training, install NCCL libraries:
```bash
sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev
```

## Pretraining Configuration

### Key Arguments

- `--resdir`: Output directory for saving model checkpoints
- `--datapath`: Root directory containing training and validation data
- `--train_dataname`: Filename for training dataset (TSV format)
- `--val_dataname`: Filename for validation dataset (TSV format)
- `--batch_size`: Set it as 9 * number of GPUs (e.g., 18 when using 2 GPUs)
- `--max_length`: Set it as 0.2 * sequence length (e.g., 2000 for 10000 bp sequences)
- `--lr`: Maximum learning rate for training
- `--warmup_epochs`: Number of epochs for learning rate warmup

### Training Modes

1. **Single GPU Training**: Automatically detected when only one GPU is available
2. **Distributed Data Parallel (DDP)**: Automatically enabled with multiple GPUs

## Usage

This training script is configured for 2 NVIDIA H100. When using a different number of GPUs or different GPU models, you may need to adjust `batch_size` and `max_length` accordingly.

### Basic Training

```bash
torchrun \
    main.py \
    --datapath PATH_TO_DATA_DIR \
    --train_dataname train_2m.tsv \
    --val_dataname val_40k.tsv \
    --batch_size 18 \
    --max_length 2000 \
    --lr 2e-06 \ 
    --warmup_epochs 0.3
```

### Resuming Training

```bash
torchrun \
    main.py \
    --datapath PATH_TO_DATA_DIR \
    --train_dataname train_2m.tsv \
    --val_dataname val_40k.tsv \
    --batch_size 18 \
    --max_length 2000 \
    --lr 2e-06 \ 
    --warmup_epochs 0.3 \
    --resume_from CHECKPOINT_DIR
```

## Logging and Monitoring

- TensorBoard logs are saved in `./tensorboard_gpu{gpu_id}` directory
- Loss, validation loss and learning rate are logged

## Model Checkpointing

- Model checkpoints are saved under `./RESDIR/` directory
- Best model is saved in `./RESDIR/best/` directory
- Training state can be resumed from checkpoints




