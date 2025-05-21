import os
import csv
import torch
import math
import numpy as np
import torch.distributed as dist
from typing import Optional
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import random
import itertools

class GenomeHierarchihcalDataset(Dataset):
    def __init__(self, args, load_train=True):
        if load_train:
            with open(os.path.join(args.datapath, args.train_dataname)) as tsvfile:
                data = list(csv.reader(tsvfile, delimiter="\t"))
                self.train = True
        else:
            with open(os.path.join(args.datapath, args.val_dataname)) as tsvfile:
                data = list(csv.reader(tsvfile, delimiter="\t"))
                self.train = False

        self.id = [i for i in range(len(data[1:]))]
        self.seq1 = [d[11] for d in data[1:]]
        self.seq2 = [d[12] for d in data[1:]]
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
        if self.train:
            for i in index:
                label = [self.superkingdom[i], self.kingdom[i], self.phylum[i], self.class_[i], self.order[i], self.family[i], self.genus[i], self.species[i]]
                rand_i = random.choice(list(range(200)))
                sequences1.append(self.seq1[i][rand_i:])
                sequences2.append(self.seq2[i][rand_i:])
                labels.append(label)
        else:
            for i in index:
                label = [self.superkingdom[i], self.kingdom[i], self.phylum[i], self.class_[i], self.order[i], self.family[i], self.genus[i], self.species[i]]
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
                (kingdom, self.dataset.labels[superkingdom]),
                (phylum, self.dataset.labels[superkingdom][kingdom]),
                (class_, self.dataset.labels[superkingdom][kingdom][phylum]),
                (order, self.dataset.labels[superkingdom][kingdom][phylum][class_]),
                (family, self.dataset.labels[superkingdom][kingdom][phylum][class_][order]),
                (genus, self.dataset.labels[superkingdom][kingdom][phylum][class_][order][family]),
                (species, self.dataset.labels[superkingdom][kingdom][phylum][class_][order][family][genus]),
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
    
class TrainDataset(Dataset):
    def __init__(self, args):
        # Set chunk size (multiple of batch size is efficient)
        self.chunk_size = args.batch_size * 3881

        self.datapath = args.datapath
        self.file_pattern = "train_2m_{}.tsv"
        
        # Manage current epoch and loaded data
        self.current_epoch = 0
        self.current_data = None
        self.current_chunk_start = 0
        self.current_chunk_end = 0
        
        # Check total number of samples (read initial file)
        self.total_samples = 2165598
        
        # Cache for line positions in files to avoid rescanning
        self.line_positions_cache = {}
        
    def get_current_file_path(self):
        return os.path.join(self.datapath, self.file_pattern.format(self.current_epoch))
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.current_data = None
        self.current_chunk_start = 0
        self.current_chunk_end = 0
    
    def ensure_data_loaded(self, idx):
        if (self.current_data is None or 
            idx < self.current_chunk_start or 
            idx >= self.current_chunk_end):
            self.load_chunk(idx)
    
    def index_file_lines(self, file_path):
        """Index all line positions in the file and cache them for future access"""
        if file_path in self.line_positions_cache:
            return self.line_positions_cache[file_path]
            
        line_positions = []
        with open(file_path, 'rb') as f:
            # Skip header
            f.readline()
            header_end = f.tell()
            
            # Record start position of each line
            line_positions.append(header_end)
            
            # Find all line start positions efficiently
            # Use larger buffer for faster reading
            buffer_size = 8 * 1024 * 1024  # 8MB buffer
            chunk = f.read(buffer_size)
            
            pos = header_end
            while chunk:
                for i in range(len(chunk)):
                    if chunk[i] == ord('\n'):
                        line_positions.append(pos + i + 1)
                
                pos += len(chunk)
                chunk = f.read(buffer_size)
                
                # Early stop if we've found enough lines
                if len(line_positions) > self.total_samples:
                    break
                
        # Cache the result
        self.line_positions_cache[file_path] = line_positions
        return line_positions
    
    def load_chunk(self, start_idx):
        # Adjust start position (align to chunk size multiple)
        chunk_start = (start_idx // self.chunk_size) * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.total_samples)
        
        # Get current file path
        current_file_path = self.get_current_file_path()
        
        # Get indexed line positions (cached if already computed)
        line_positions = self.index_file_lines(current_file_path)
        
        # Read chunk data efficiently using line positions
        chunk_data = []
        with open(current_file_path, 'r') as f:
            # Skip header if needed
            if chunk_start > 0 and chunk_start < len(line_positions):
                # Seek to the starting line position
                f.seek(line_positions[chunk_start])
            else:
                # Skip header line
                f.readline()
                
            # Read only the necessary lines (read in bulk for efficiency)
            lines_to_read = chunk_end - chunk_start
            chunk_data = []
            
            # Read in batches to improve performance with large files
            batch_size = 1000
            for i in range(0, lines_to_read, batch_size):
                batch_lines = list(itertools.islice(f, min(batch_size, lines_to_read - i)))
                for line in batch_lines:
                    # Parse the TSV line
                    row = line.strip().split('\t')
                    chunk_data.append(row)
        
        # Process data into required format using efficient numpy operations
        num_rows = len(chunk_data)
        
        # Preallocate arrays for better performance
        seq1 = []
        seq2 = []
        
        # Extract arrays directly from rows
        species = np.zeros(num_rows, dtype=np.int32)
        genus = np.zeros(num_rows, dtype=np.int32)
        family = np.zeros(num_rows, dtype=np.int32)
        order = np.zeros(num_rows, dtype=np.int32)
        class_ = np.zeros(num_rows, dtype=np.int32)
        phylum = np.zeros(num_rows, dtype=np.int32)
        kingdom = np.zeros(num_rows, dtype=np.int32)
        superkingdom = np.zeros(num_rows, dtype=np.int32)
        
        # Process rows in a single pass
        for i, row in enumerate(chunk_data):
                seq1.append(row[11])
                seq2.append(row[12])
                species[i] = int(row[3])
                genus[i] = int(row[4])
                family[i] = int(row[5])
                order[i] = int(row[6])
                class_[i] = int(row[7])
                phylum[i] = int(row[8])
                kingdom[i] = int(row[9])
                superkingdom[i] = int(row[10])
        
        # Store data in a more efficient format
        self.current_data = {
            'id': list(range(chunk_start, chunk_start + num_rows)),
            'seq1': seq1,
            'seq2': seq2,
            'species': species,
            'genus': genus,
            'family': family,
            'order': order,
            'class_': class_,
            'phylum': phylum,
            'kingdom': kingdom,
            'superkingdom': superkingdom,
        }
        
        # Update chunk range
        self.current_chunk_start = chunk_start
        self.current_chunk_end = chunk_start + num_rows
    
    def get_item_at_idx(self, idx):
        # Calculate relative position within chunk
        relative_idx = idx - self.current_chunk_start
        
        # Fast numpy array indexing for numerical values
        label = [
            int(self.current_data['superkingdom'][relative_idx]),
            int(self.current_data['kingdom'][relative_idx]),
            int(self.current_data['phylum'][relative_idx]),
            int(self.current_data['class_'][relative_idx]),
            int(self.current_data['order'][relative_idx]),
            int(self.current_data['family'][relative_idx]),
            int(self.current_data['genus'][relative_idx]),
            int(self.current_data['species'][relative_idx]),
        ]
        
        # For training data, cut from random position
        rand_i = random.choice(list(range(200)))
        seq1 = self.current_data['seq1'][relative_idx][rand_i:]
        seq2 = self.current_data['seq2'][relative_idx][rand_i:]
            
        return seq1, seq2, label
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        # For batch sampling, index is a list of indices
        if isinstance(index, list):
            sequences1, sequences2, labels = [], [], []
            
            # Load data using first index of the chunk
            self.ensure_data_loaded(index[0])
                
            # Process all indices in this chunk
            for idx in index:
                seq1, seq2, label = self.get_item_at_idx(idx)
                sequences1.append(seq1)
                sequences2.append(seq2)
                labels.append(label)
        
        return [sequences1, sequences2], torch.tensor(labels)
    
class TrainBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: TrainDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
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

    def __iter__(self):
        # Define a fixed number of batches per epoch
        num_batches = self.__len__()
        indices = list(range(self.dataset.__len__()))
        for i in range(num_batches):
            batch =indices[(self.num_replicas*i+self.rank) * self.batch_size:(self.num_replicas*i+self.rank + 1) * self.batch_size]
            yield batch
    
    def set_epoch(self, epoch):
        self.epoch = epoch


    def __len__(self) -> int:
        return self.num_samples // self.batch_size
    
class ValidationBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: GenomeHierarchihcalDataset,
        ) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of batches, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.batch_size != 0:  # type: ignore
            self.num_samples = (len(self.dataset) // self.batch_size) * self.batch_size
        else:
            self.num_samples = len(self.dataset)

    def __iter__(self):
        indices = list(range(self.num_samples))

        # Define a fixed number of batches per epoch
        num_batches = self.__len__()
        for i in range(num_batches):
            batch = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield batch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
    
def load_deep_genome_hierarchical(args):
    #train_dataset = TrainDataset(args)
    train_dataset = GenomeHierarchihcalDataset(args, load_train=True)
    val_dataset = GenomeHierarchihcalDataset(args, load_train=False)
    
    sequence_datasets = {'train': train_dataset,
                      'val': val_dataset}
    if args.distributed:
        train_sampler = TrainBatchSampler(batch_size=args.batch_size,
                                           drop_last=True,
                                           dataset=train_dataset)
    else:
        train_sampler = TrainBatchSampler(batch_size=args.batch_size,
                                           drop_last=True,
                                           dataset=train_dataset,
                                           num_replicas=1,
                                           rank=0)
    val_sampler = ValidationBatchSampler(batch_size=args.batch_size,
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
