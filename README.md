# Welcome to Shinkansen's Msc in Data Science Project
By Anders Hjulmand, Andreas Flensted, Eisuke Okuda.

*Find the weekly meetings outline in: [Weekly Meetings](weekly_meetings/README.md)*

*Find the overleaf report on: [Overleaf](https://www.overleaf.com/project/679796b5a02b660e4f96beff)* 


## Contents

- [Welcome to Shinkansen's Msc in Data Science Project](#welcome-to-shinkansens-msc-in-data-science-project)
  - [Contents](#contents)
  - [1. Introduction](#1-introduction)
  - [2. Model and Training Data](#2-model-and-training-data)
  - [3. Setup environment](#3-setup-environment)
  - [4. Quick start](#4-quick-start)
  - [5. Training](#5-training)
  - [6. Experiment 1: Metagenomics binning](#6-experiment-1-metagenomics-binning)
  - [7. Experiment 2: Phenotype Classification](#7-experiment-2-phenotype-classification)

## 1. Introduction
DNABERT-H is a DNA language model based on [DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S) and trained using Hierarchical Multi-label Contrastive Learning from [Use All The Labels](https://arxiv.org/abs/2204.13207).


Biological classification of genomes follows a hierarchical taxonomy, ranging from the broadest category - *superkingdom* (e.g. Eukarya) - down to the most specific level, *species* (e.g. Homo sapiens). DNABERT-H leverages all labels in the taxonomic hiearchy during training, and thereby creates embeddings that reflect the hiearchical categorization. The model is trained using bacteria, virus, and fungi genomes.

DNABERT-H is beneficial for application in metagenomics, including metagenomics binning and phenotype classification.  

[Insert tn-sne image.]

[insert some results.]




## 2. Model and Training Data

The pre-trained model is available at [insert link]. 

Place the model in the directory `train/dnaberth_weights/`.

The training data is available at [insert link].

The evaluation data is available at [insert link].


## 3. Setup environment

Clone the repository:
```bash
git clone https://github.com/affo98/DNA-language-models-and-GLP1-microbiome-changes.git
cd DNA-language-models-and-GLP1-microbiome-changes
```

If conda is not installed, run the following code to install it:
```bash
chmod +x get_conda.sh
bash get_conda.sh
source ~/.bashrc
```

## 4. Quick start

Setup conda environment.

```
conda env create -f binning/envs/binning.yml && conda activate binning 
pip uninstall -y triton
```

Load the model using the [transformers](https://github.com/huggingface/transformers) and backbone [DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S).

```
import torch
from transformers import AutoTokenizer, BertConfig, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-S")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", config, trust_remote_code=True)
model.load_state_dict(torch.load(train/dnaberth_weights/INESERTNAME/pytorch_model.bin))
```

Run inference with model to get embeddings of a DNA sequence.

```
dna = "ATGCGTACCTGAACTGGTACCGTATCGAGGCTTACCGGATAGCTTGAACCGTACGTTAGGCTAGCTTACGGAATGCCGT"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768
```

## 5. Training

The training procedure is described in the folder `train`. 



## 6. Experiment 1: Metagenomics binning

We assess the performance of our model on metagenomics binning. This task is essentially a clustering problem where the number of clusters is unknown. The goal is to identify clusters that correspond to individual genomes - known as metagenome-assembled genomes (MAGs).

The implementation details of experiment 1 is described in the folders `cami2_benchmark` and `binning`.


## 7. Experiment 2: Phenotype Classification

As a second experiment, we assess the performance of our model on phenotype classification using a Multiple Instance Learing (MIL) framework. 

The pre-processing of data for phenotype classification is described in the folder `ngs_pipeline`.

The implementation details of experiment 2 is described in the folders `phenotype_mil` and `binning`.




