'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 *
 * Modified by Eisuke Okuda in 2025
'''
import os
import csv
import torch
import math
import torch.distributed as dist
from typing import Optional
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import random
from transformers import AutoTokenizer


class GenomeHierarchihcalDataset(Dataset):
    def __init__(self, args, load_train=True):
        if load_train:
            with open(os.path.join(args.datapath, args.train_dataname)) as tsvfile:
                data = list(csv.reader(tsvfile, delimiter="\t"))
        else:
            with open(os.path.join(args.datapath, args.val_dataname)) as tsvfile:
                data = list(csv.reader(tsvfile, delimiter="\t"))

        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.id = [i for i in range(len(data[1:]))]
        self.seq1 = [d[11] for d in range(len(data[1:]))]
        self.seq2 = [d[12] for d in range(len(data[1:]))]
        self.species = [int(d[3]) for d in data[1:]]
        self.genus = [int(d[4]) for d in data[1:]]
        self.family = [int(d[5]) for d in data[1:]]
        self.order = [int(d[6]) for d in data[1:]]
        self.class_ = [int(d[7]) for d in data[1:]]
        self.phylum = [int(d[8]) for d in data[1:]]
        self.kingdom = [int(d[9]) for d in data[1:]]
        self.superkingdom = [int(d[10]) for d in data[1:]]
        self.labels = {}

        for i in range(len(self.id)):
            # Create a tuple of labels in hierarchical order
            label_hierarchy = (
                self.superkingdom[i],
                self.kingdom[i],
                self.phylum[i],
                self.class_[i],
                self.order[i],
                self.family[i],
                self.genus[i],
                self.species[i],
                self.id[i]
            )
            
            # Navigate and create nested dictionaries
            current = self.labels
            for level in label_hierarchy[:-1]:  # Exclude the last item (id)
                current = current.setdefault(level, {})
            current[label_hierarchy[-1]] = i  # Set the final id mapping to index

    
    def __len__(self):
        return len(self.species)
    
    def get_label_by_index(self, index):
        superkingdom = self.superkingdom[index]
        kingdom = self.kingdom[index]
        phylum = self.phylum[index]
        class_ = self.class_[index]
        order = self.order[index]
        family = self.family[index]
        genus = self.genus[index]
        species = self.species[index]
        id = self.id[index]

        return superkingdom, kingdom, phylum, class_, order, family, genus, species, id
    
    def __getitem__(self, index):
        sequences1, sequences2, labels = [], [], []
        for i in index:
            label = [self.species[i], self.genus[i], self.family[i], self.order[i], self.class_[i], self.phylum[i], self.kingdom[i], self.superkingdom[i]]
            sequences1.append(self.seq1[i])
            sequences2.append(self.seq2[i])
            labels.append(label)
        
        return [sequences1, sequences2], torch.tensor(labels)

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):
                        random_label = random.sample(list(curr_dict.keys()), 1)[0]
            else:
                random_label = random.sample(list(curr_dict.keys()), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: GenomeHierarchihcalDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        # Set random generator with epoch seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.epoch)
        visited = set()
        # Shuffle dataset indices
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if not self.drop_last:
            # Add extra samples to make indices evenly divisible by total_size
            indices += indices[:(self.total_size - len(indices))]
        else:
            # Remove tail samples to make indices evenly divisible
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample indices for the current rank in distributed training
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Define a fixed number of batches per epoch
        num_batches = self.__len__()
        for _ in range(num_batches):
            batch = []
            # Calculate remaining indices that haven't been visited
            remaining = list(set(indices).difference(visited))
            # If there are no remaining indices, reset the visited set to allow re-sampling
            if not remaining:
                visited = set()
                remaining = indices.copy()
            # Sample an initial index from the remaining indices
            init_idx = remaining[torch.randint(len(remaining), (1,)).item()]
            batch.append(init_idx)
            visited.add(init_idx)
            # Get the hierarchical labels for the chosen index
            superkingdom, kingdom, phylum, class_, order, family, genus, species, id = \
                self.dataset.get_label_by_index(init_idx)
            
            # Create hierarchical path with corresponding label dictionaries
            hierarchy_path = [
                (superkingdom, self.dataset.labels),
            #    (kingdom, self.dataset.labels[superkingdom]),
            #    (phylum, self.dataset.labels[superkingdom][kingdom]),
            #    (class_, self.dataset.labels[superkingdom][kingdom][phylum]),
                (order, self.dataset.labels[superkingdom][kingdom][phylum][class_]),
            #    (family, self.dataset.labels[superkingdom][kingdom][phylum][class_][order]),
            #    (genus, self.dataset.labels[superkingdom][kingdom][phylum][class_][order][family]),
                (species, self.dataset.labels[superkingdom][kingdom][phylum][class_][order][family][genus]),
            #    (id, self.dataset.labels[superkingdom][kingdom][phylum][class_][order][family][genus][species])
            ]
            
            # Sample one index for each level in the hierarchical path
            for label, dict_ in hierarchy_path:
                sampled_idx = self.random_unvisited_sample(label, dict_, visited, indices, remaining)
                batch.append(sampled_idx)
                visited.add(sampled_idx)
            
            # If batch size exceeds the target, truncate the batch
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            # If the batch has fewer samples than required, fill in with random indices
            while len(batch) < self.batch_size:
                extra = indices[torch.randint(len(indices), (1,)).item()]
                batch.append(extra)
                visited.add(extra)
            
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
    
def load_deep_genome_hierarchical(args):
    train_dataset = GenomeHierarchihcalDataset(args, load_train=True)
    val_dataset = GenomeHierarchihcalDataset(args, load_train=False)
    
    sequence_datasets = {'train': train_dataset,
                      'val': val_dataset}
    
    train_sampler = HierarchicalBatchSampler(batch_size=args.batch_size,
                                       drop_last=True,
                                       dataset=train_dataset)
    val_sampler = HierarchicalBatchSampler(batch_size=args.batch_size,
                                           drop_last=True,
                                           dataset=val_dataset)
    sampler = {'train': train_sampler,
               'val': val_sampler}
    
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(sequence_datasets[x], sampler=sampler[x],
                                       num_workers=4, batch_size=1,
                                       pin_memory=True)
        for x in ['train', 'val']}
    
    return dataloaders_dict, sampler
