import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer

from training import Trainer
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer
from models.dnabert_s import DNABert_S
from utils.losses import HMLC
import builtins
import torch.backends.cudnn as cudnn
from dataloader.hierarchical_dataset import load_deep_genome_hierarchical
import tensorboard_logger as tb_logger
from datetime import timedelta
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

def run(args):
    args.resPath = setup_path(args)
    #set_global_random_seed(args.seed)

    device_count = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_count))

    if device_count <= 1:
        print("Running on a single GPU.")
        # Add a flag to indicate non-distributed mode
        args.distributed = False
        main_single_gpu(0, 1, args) # Use gpu=0, ngpus_per_node=1
    else:
        print("Running on multiple GPUs using DDP.")
        # Add a flag to indicate distributed mode
        args.distributed = True
        mp.spawn(main_worker, nprocs=device_count, args=(device_count, args))

    return None

# New function for single GPU execution
def main_single_gpu(gpu, ngpus_per_node, args):
    print("Setting up for single GPU execution on GPU: {}".format(gpu))
    args.gpu = gpu

    # Setup logger
    args.tb_folder = f'./tensorboard_gpu{args.gpu}'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # Set the device
    torch.cuda.set_device(args.gpu)

    # Setup model and criterion (set_model handles moving to args.gpu)
    # Pass ngpus_per_node=1 to avoid DDP wrapping
    model, criterion = set_model(ngpus_per_node, args)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # Load data (pass distributed flag)
    # Ensure load_deep_genome_hierarchical can handle args.distributed = False
    dataloaders_dict, sampler = load_deep_genome_hierarchical(args)

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, args) # Pass the potentially unwrapped model

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=int(args.warmup_epochs * len(dataloaders_dict['train']))
    )

    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int((args.epochs - args.warmup_epochs) * len(dataloaders_dict['train'])),
        eta_min=args.min_lr
    )

    # Combine them
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(args.warmup_epochs * len(dataloaders_dict['train']))]
    )

    trainer = Trainer(model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, logger, args, scheduler)

    # Train the model
    print("Starting training on single GPU...")
    trainer.train()

    # Run validation on the single GPU
    print("Running validation on single GPU...")
    # Pass the model directly as it's not wrapped in DDP
    trainer.run_validation(model)

    print("Single GPU training and validation finished.")


def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    args.gpu = gpu

    args.tb_folder = f'./tensorboard_gpu{args.gpu}' # Use unique folder per GPU
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args_pass, **kwargs_pass): # Fixed signature
            pass
        builtins.print = print_pass

    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=ngpus_per_node, rank=args.gpu, timeout=timedelta(hours=24))
    dist.barrier()

    # Setup model and criterion (set_model handles DDP wrapping)
    model, criterion = set_model(ngpus_per_node, args)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    dist.barrier()

    # Load data (pass distributed flag)
    # Ensure load_deep_genome_hierarchical uses args.distributed = True here
    dataloaders_dict, sampler = load_deep_genome_hierarchical(args)
    dist.barrier()

    optimizer = get_optimizer(model, args) # Pass the DDP model

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=int(args.warmup_epochs * len(dataloaders_dict['train']))
    )

    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int((args.epochs - args.warmup_epochs) * len(dataloaders_dict['train'])),
        eta_min=args.min_lr
    )

    # Combine them
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(args.warmup_epochs * len(dataloaders_dict['train']))]
    )

    # Pass the distributed samplers contained in the sampler dict
    trainer = Trainer(model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, logger, args, scheduler)
    trainer.train()

    # DDP Validation logic (only on rank 0 after training)
    dist.barrier() # Ensure all training is done
    if args.gpu == 0:
        # Unwrap the model from DDP before destroying the process group
        unwrapped_model = model.module

        # First destroy the process group to exit DDP mode
        print("Rank 0 destroying process group for validation...")
        dist.destroy_process_group()

        # Then run validation on a single GPU with the unwrapped model
        print("Running validation on rank 0 GPU...")
        # Need to re-setup trainer or directly call validation logic
        # Re-create dataloader/sampler for single GPU validation if needed
        # Or reuse existing trainer instance with unwrapped model if possible.
        # For simplicity, let's assume trainer.run_validation can handle this
        # Temporarily set device for validation
        torch.cuda.set_device(0) # Validate on GPU 0
        trainer.run_validation(unwrapped_model) # Pass unwrapped model
        print("DDP validation finished on rank 0.")

    else:
        # Other GPUs just destroy the process group and exit
        print(f"Rank {args.gpu} destroying process group...")
        dist.destroy_process_group()
        print(f"Rank {args.gpu} finished.")


def set_model(ngpus_per_node, args):
    model = DNABert_S(feat_dim=args.feat_dim)
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # Adjust batch size per GPU. If ngpus_per_node is 1, batch size remains the same.
    args.batch_size = int(args.batch_size / ngpus_per_node)
    print(f"GPUs {ngpus_per_node}, Batch size per GPU {args.batch_size}")
    #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node) # This might need adjustment based on how workers are used

    # Only wrap with DDP if using more than one GPU
    if ngpus_per_node > 1 and args.distributed:
         # Set find_unused_parameters=True if necessary, might impact performance
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False) # Consider setting find_unused_parameters based on model structure
        print(f"GPU {args.gpu}: Wrapped model with DDP.")
    else:
        print(f"GPU {args.gpu}: Running on single GPU, DDP not applied.")


    criterion = criterion.cuda(args.gpu)
    return model, criterion

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--resdir', type=str, default="./results")
    parser.add_argument('--logging_step', type=int, default=1000, help="How many iteration steps to save the model checkpoint and loss value once")
    parser.add_argument('--logging_num', type=int, default=12, help="How many times to log totally")
    # Dataset
    parser.add_argument('--datapath', type=str, default='./data/reference_genome_links/', help="The dict of data")
    parser.add_argument('--train_dataname', type=str, default='train_2m.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='val_48k.csv', help="Name of the data used for validating")
    # Training parameters
    parser.add_argument('--max_length', type=int, default=512, help="Max length of tokens")
    # Make args.batch_size the *total* batch size across all GPUs
    parser.add_argument('--batch_size', type=int, default=36, help="Total batch size across all GPUs. Will be divided among GPUs.")
    parser.add_argument('--lr', type=float, default=1e-05, help="Learning rate")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--warmup_epochs', type=float, default=0.3, help='Number of warmup epochs for learning rate')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    # Contrastive learning
    parser.add_argument('--feat_dim', type=int, default=128, help="Dimension of the projected features for instance discrimination loss")
    parser.add_argument('--temp', type=float, default=0.1, help="Temperature used in InfoNCE loss")
    parser.add_argument('--loss', type=str, default='hmlc', help="The type of loss function")
    parser.add_argument('--workers', type=int, default=4, help="Number of workers for dataloader") # Added workers arg
    # Add distributed flag (internal use, not meant to be set by user)
    parser.add_argument('--distributed', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    # Ensure results directory exists
    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)
    run(args)




    


