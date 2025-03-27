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

def run(args):
    args.resPath = setup_path(args)
    #set_global_random_seed(args.seed)

    device_id = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_id))
    mp.spawn(main_worker, nprocs=device_id, args=(device_id, args))
    
    return None

def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    args.gpu = gpu

    args.tb_folder = f'./tensorboard_{args.gpu}'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=ngpus_per_node, rank=args.gpu, timeout=timedelta(minutes=5))
    dist.barrier()
    model, criterion = set_model(ngpus_per_node, args)
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    optimizer = get_optimizer(model, args)
    #cudnn.benchmark = True
 
    dataloaders_dict, sampler = load_deep_genome_hierarchical(args)
    dist.barrier()

    trainer = Trainer(model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, logger, args)
    trainer.train()
    trainer.val()
    dist.destroy_process_group()

        
            
def set_model(ngpus_per_node, args):
    model = DNABert_S(feat_dim=args.feat_dim)
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    print("Updated batch size is {}".format(args.batch_size))
    #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

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
    parser.add_argument('--batch_size', type=int, default=48, help="Batch size used for training/validating dataset")
    parser.add_argument('--lr', type=float, default=3e-06, help="Learning rate")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    #parser.add_argument('--train_batch_size', type=int, default=48, help="Batch size used for training dataset")
    #parser.add_argument('--val_batch_size', type=int, default=360, help="Batch size used for validating dataset")
     
    # Contrastive learning
    parser.add_argument('--feat_dim', type=int, default=128, help="Dimension of the projected features for instance discrimination loss")
    parser.add_argument('--temp', type=float, default=0.07, help="Temperature required by contrastive loss")
    parser.add_argument('--loss', type=str, default='hmce', help='loss type', choices=['hmc', 'hce', 'hmce'])
    #parser.add_argument('--con_method', type=str, default='same_species', help="Which data augmentation method used, include dropout, double_strand, mutate, same_species")
    #parser.add_argument('--mix', action="store_true", help="Whether use i-Mix method")
    #parser.add_argument('--dnabert2_mix_dict', type=str, default="./DNABERT-2-117M-MIX", help="Dictionary of the modified code for DNABert-2 to perform i-Mix")
    #parser.add_argument('--mix_alpha', type=float, default=1.0, help="Value of alpha to generate i-Mix coefficient")
   # parser.add_argument('--mix_layer_num', type=int, default=-1, help="Which layer to perform i-Mix, if the value is -1, it means manifold i-Mix")
   # parser.add_argument('--curriculum', action="store_true", help="Whether use curriculum learning")
    
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)




    


