import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self, model, tokenizer, criterion, optimizer, dataloaders_dict, sampler, args):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = dataloaders_dict['train']
        self.val_loader = dataloaders_dict['val']
        self.train_sampler = sampler['train']
        self.val_sampler = sampler['val']
        self.gstep = 0
        self.criterion = criterion

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
        sequences1, sequences2 = sequences[0], sequences[1]
        sequences1 = [s[0] for s in sequences1]
        sequences2 = [s[0] for s in sequences2]
        feat1 = self.get_batch_token(sequences1)
        feat2 = self.get_batch_token(sequences2)

        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids, attention_mask

    def save_model(self, step=None, save_best=False):
        if dist.get_rank() == 0:
            if save_best:
                save_dir = os.path.join(self.args.resPath, 'best')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.module.dnabert2.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.model.module.contrast_head.state_dict(), save_dir+"/con_weights.ckpt")
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
        with torch.autocast(device_type="cuda"):
            feat1, feat2, _, _ = self.model(input_ids, attention_mask)
            features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            losses = self.criterion(features, labels)
            loss = losses["instdisc_loss"]
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return losses
    
    def train(self):
        self.all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(self.all_iter, len(self.train_loader)))

        self.model.train()
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            for _, (sequences, labels) in enumerate(epoch_iterator):
                labels = labels.squeeze()
                input_ids, attention_mask = self.prepare_pairwise_input(sequences)
                losses = self.train_step(input_ids, attention_mask, labels)
                if self.gstep%self.args.logging_step==0:
                    self.save_model(step=self.gstep)
                if self.gstep > self.args.logging_step*self.args.logging_num:
                    break
                self.gstep += 1
                
            print("Finish Epoch: ", epoch)
        return None
    
    def val(self):
        self.model.eval()
        best_checkpoint = 0
        best_val_loss = 10000
        for step in range(self.args.logging_step, np.min([self.all_iter, self.args.logging_step*self.args.logging_num+1]), self.args.logging_step):
            load_dir = os.path.join(self.args.resPath, str(step))
            self.model.module.dnabert2.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
            self.model.module.contrast_head.load_state_dict(torch.load(load_dir+'/con_weights.ckpt'))
            self.val_sampler.set_epoch(1)
            val_loss = 0.
            for idx, (sequences, labels) in enumerate(self.val_loader):
                with torch.no_grad():
                    labels = labels.squeeze()
                    labels = labels.squeeze().cuda(self.args.gpu, non_blocking=True)
                    input_ids, attention_mask = self.prepare_pairwise_input(sequences)
                    input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
                    attention_mask = attention_mask.cuda(self.args.gpu, non_blocking=True)
                    
                    with torch.autocast(device_type="cuda"):
                        feat1, feat2, _, _ = self.model(input_ids, attention_mask)
                        features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
                        losses = self.criterion(features, labels)
                        val_loss += losses["instdisc_loss"]
            val_loss = val_loss.item()/(idx+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = step
                self.save_model(save_best=True)
    