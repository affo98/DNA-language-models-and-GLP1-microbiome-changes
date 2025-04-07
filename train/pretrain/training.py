import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import time

class Trainer(nn.Module):
    def __init__(self, model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, logger, args, scheduler=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = dataloaders_dict['train']
        self.val_loader = dataloaders_dict['val']
        self.train_sampler = sampler['train']
        self.val_sampler = sampler['val']
        self.gstep = 0
        self.criterion = criterion
        self.logger = logger

    def get_batch_token(self, dna_seq):
        max_length = self.args.max_length
        token_feat = self.tokenizer.batch_encode_plus(
            dna_seq, 
            max_length=max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        
    def prepare_pairwise_input(self, sequences):
        rand_i = np.random.randint(200)
        rand_j = np.random.randint(200)
        sequences1, sequences2 = sequences[0], sequences[1]
        sequences1 = [s[0][rand_i:] for s in sequences1]
        sequences2 = [s[0][rand_j:] for s in sequences2]
        feat1 = self.get_batch_token(sequences1)
        feat2 = self.get_batch_token(sequences2)

        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids, attention_mask

    def save_model(self, step=None, save_best=False):
        if self.args.gpu == 0:
            if save_best:
                save_dir = os.path.join(self.args.resPath, 'best')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.dnabert2.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.model.contrast_head.state_dict(), save_dir+"/con_weights.ckpt")
            else:
                save_dir = os.path.join(self.args.resPath, str(step))
                self.last_saved_step = step
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.module.dnabert2.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.model.module.contrast_head.state_dict(), save_dir+"/con_weights.ckpt")

    def train_step(self, input_ids, attention_mask, labels):    
        if self.args.gpu is not None:
            input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
            attention_mask = attention_mask.cuda(self.args.gpu, non_blocking=True)
            labels = labels.squeeze().cuda(self.args.gpu, non_blocking=True)
            bsz = labels.shape[0]
        with torch.autocast(device_type="cuda"):
            feat1, feat2, _, _ = self.model(input_ids, attention_mask)
            features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            losses = self.criterion(features, labels)
            loss = losses["instdisc_loss"]
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
            if self.gstep % self.args.print_freq == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.log_value('learning_rate', current_lr, self.gstep)
        
        return losses, bsz
    
    def train(self):
        self.all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(self.all_iter, len(self.train_loader)))

        self.model.train()
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')

            end = time.time()

            progress = ProgressMeter(len(self.train_loader),
                    [batch_time, data_time, losses],
                    prefix = "Epoch: [{}]".format(epoch))
            
            for _, (sequences, labels) in enumerate(epoch_iterator):
                data_time.update(time.time() - end)
                labels = labels.squeeze()
                input_ids, attention_mask = self.prepare_pairwise_input(sequences)
                loss, bsz = self.train_step(input_ids, attention_mask, labels)
                losses.update(loss["instdisc_loss"].item(), bsz)

                batch_time.update(time.time() - end)
                end = time.time()
                sys.stdout.flush()
                if self.gstep % self.args.print_freq == 0:
                    progress.display(self.gstep%len(self.train_loader))
                if self.gstep%self.args.logging_step==0:
                    self.logger.log_value('loss', losses.avg, self.gstep)
                    self.save_model(step=self.gstep)
                #if self.gstep > self.args.logging_step*self.args.logging_num:
                #    break
                self.gstep += 1
                
            print("Finish Epoch: ", epoch)
            dist.barrier()
        
        return None
    
    def run_validation(self, unwrapped_model=None):
        """
        Validation method to run on a single GPU after exiting DDP mode
        
        Args:
            unwrapped_model: The model unwrapped from DDP, if provided
        """
        print("Start Validation... ")
        
        # Use the provided unwrapped model if available
        if unwrapped_model is not None:
            # Replace the DDP-wrapped model with the unwrapped one
            self.model = unwrapped_model
        elif hasattr(self.model, 'module'):
            # Unwrap the model if it's still wrapped in DDP
            self.model = self.model.module
        
        self.model.eval()
        best_checkpoint = 0
        best_val_loss = 10000
        
        # Determine the range of training steps to evaluate
        max_steps = self.args.logging_step * self.args.logging_num
        
        for step in range(self.args.logging_step, max_steps + 1, self.args.logging_step):
            load_dir = os.path.join(self.args.resPath, str(step))
            if not os.path.exists(load_dir):
                print(f"Checkpoint directory {load_dir} does not exist, skipping.")
                continue
            
            try:
                # Load model weights directly to the unwrapped model
                self.model.dnabert2.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
                self.model.contrast_head.load_state_dict(torch.load(load_dir+'/con_weights.ckpt'))
                print(f"Successfully loaded checkpoint from {load_dir}")
            except Exception as e:
                print(f"Error loading checkpoint from {load_dir}: {e}")
                continue
            
            # Set validation sampler epoch
            self.val_sampler.set_epoch(1)

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')

            end = time.time()

            progress = ProgressMeter(len(self.val_loader),
                    [batch_time, data_time, losses],
                    prefix="Step: [{}]".format(step))
            
            for idx, (sequences, labels) in enumerate(self.val_loader):
                data_time.update(time.time() - end)
                with torch.no_grad():
                    labels = labels.squeeze()
                    labels = labels.squeeze().cuda(self.args.gpu, non_blocking=True)
                    bsz = labels.shape[0]
                    input_ids, attention_mask = self.prepare_pairwise_input(sequences)
                    input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
                    attention_mask = attention_mask.cuda(self.args.gpu, non_blocking=True)
                    
                    with torch.autocast(device_type="cuda"):
                        # Forward pass through the unwrapped model
                        feat1, feat2, _, _ = self.model(input_ids, attention_mask)
                        features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
                        loss = self.criterion(features, labels)

                        losses.update(loss["instdisc_loss"].item(), bsz)
                        batch_time.update(time.time() - end)
                        end = time.time()

                        if idx % self.args.print_freq == 0:
                            progress.display(idx)

            self.logger.log_value('val_loss', losses.avg, step)            
            if losses.avg < best_val_loss:
                best_val_loss = losses.avg
                best_checkpoint = step
                self.save_model(save_best=True)
            
            print(f"Finish Step: {step}, Val Loss: {losses.avg:.6f}")

        print("Best Checkpoint at Step: ", best_checkpoint)
        
        return None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
