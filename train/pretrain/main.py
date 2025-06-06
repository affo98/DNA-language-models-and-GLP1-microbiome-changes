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
    print("Logger setup complete.")

    # Set the device
    torch.cuda.set_device(args.gpu)
    print(f"Set device to GPU {args.gpu}.")

    # Setup model and criterion
    print("Setting up model and criterion...")
    model, criterion = set_model(ngpus_per_node, args)
    print("Model and criterion setup complete.")

    # Setup tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    print("Tokenizer loaded.")

    # Load data
    print("Loading data...")
    # Ensure load_deep_genome_hierarchical can handle args.distributed = False
    dataloaders_dict, sampler = load_deep_genome_hierarchical(args)
    print(f"Data loaded. Dataloader length: {len(dataloaders_dict['train'])}")

    # Setup optimizer and scheduler
    print("Setting up optimizer...")
    optimizer = get_optimizer(model, args) # Pass the potentially unwrapped model
    print("Optimizer setup complete.")

    # Create warmup scheduler
    print("Setting up schedulers...")
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
    print("Schedulers setup complete.")

    # Setup Trainer
    print("Initializing Trainer...")
    trainer = Trainer(model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, logger, args, scheduler)
    print("Trainer initialized.")

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
                            world_size=ngpus_per_node, rank=args.gpu, timeout=timedelta(hours=72))
    dist.barrier()

    model, criterion = set_model(ngpus_per_node, args)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    dist.barrier()

    dataloaders_dict, sampler = load_deep_genome_hierarchical(args)
    dist.barrier()

    optimizer = get_optimizer(model, args)
    
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
    trainer.train()
    
    # Only run validation on the main GPU (0)
    if args.gpu == 0:
        # Unwrap the model from DDP before destroying the process group
        unwrapped_model = model.module
        
        # First destroy the process group to exit DDP mode
        dist.destroy_process_group()
        
        # Then run validation on a single GPU with the unwrapped model
        print("Running validation on a single GPU...")
        trainer.run_validation(unwrapped_model)
    else:
        # Other GPUs just exit DDP
        dist.destroy_process_group()

        
            
def set_model(ngpus_per_node, args):
    model = DNABert_S(feat_dim=args.feat_dim)
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
    print(f"{ngpus_per_node} GPUs, Batch size per GPU {args.train_batch_size}")

    # Only wrap with DDP if using more than one GPU
    if ngpus_per_node > 1 and args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
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
    parser.add_argument('--logging_step', type=int, default=10000, help="How many iteration steps to save the model checkpoint and loss value once")
    parser.add_argument('--logging_num', type=int, default=36, help="How many times to log totally")
    # Dataset
    parser.add_argument('--datapath', type=str, default='./data/reference_genome_links/', help="The dict of data")
    parser.add_argument('--train_dataname', type=str, default='train_2m.tsv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='val_40k.tsv', help="Name of the data used for validating")
    # Training parameters
    parser.add_argument('--max_length', type=int, default=2000, help="Max length of tokens")
    parser.add_argument('--train_batch_size', type=int, default=18, help="Batch size used for training dataset")
    parser.add_argument('--val_batch_size', type=int, default=18, help="Batch size used for validating dataset")
    parser.add_argument('--max_lr', type=float, default=2e-06, help="Maximum learning rate")
    parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_scale', type=int, default=100, help="Learning rate scale")
    parser.add_argument('--warmup_epochs', type=float, default=0.3, help='Number of warmup epochs for learning rate')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--distributed', action='store_true', help=argparse.SUPPRESS)
    # Contrastive learning
    parser.add_argument('--feat_dim', type=int, default=128, help="Dimension of the projected features for instance discrimination loss")
    parser.add_argument('--temp', type=float, default=0.07, help="Temperature required by contrastive loss")
    parser.add_argument('--loss', type=str, default='hmce', help='loss type', choices=['hmc', 'hce', 'hmce'])
    
    # Add resume training arguments
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint directory to resume training from')
    
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




    


