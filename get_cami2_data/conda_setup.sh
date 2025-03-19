#!/bin/bash

# Clone the repository and switch to the specified branch
git clone https://github.com/affo98/DNA-language-models-and-GLP1-microbiome-changes.git
cd DNA-language-models-and-GLP1-microbiome-changes
git checkout andersbranch

# Download and install Miniconda
curl -s -L -o /tmp/miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda_installer.sh -b -f -p /work/miniconda3

# Initialize conda
eval "$(/work/miniconda3/bin/conda shell.bash hook)"
conda init

# Install mamba for faster package management
conda install -n base -c conda-forge mamba -y

# Create and activate the required conda environment
conda env create -f get_cami2_data/envs/cami2_processing.yml
conda activate cami2_processing