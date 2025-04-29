#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --output=t2dew%j.out
#SBATCH --error=slurm%j.err 
#SBATCH --partition=purrlab_students
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --exclude=cn[3-18]
#SBATCH --exclude=desktop[1-16]
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=cn19
#SBATCH --gres=gpu:l40s:4


# 1. Download & install Miniconda3 if missing
if [[ ! -d "$HOME/miniconda" ]]; then
    echo "Miniconda not found. Installing Miniconda..."

    # Download Miniconda for Linux
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    curl -sO "$URL"
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda"
    rm Miniconda3-latest-Linux-x86_64.sh

    echo "Miniconda installed successfully."
fi

# 2. Initialize in-script
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)" 2>/dev/null

# 3. Install Mamba in the base environment
conda install -n base -c conda-forge mamba --yes

conda config --set channel_priority flexible

# 3. Create & activate your env
ENV_DIR="$HOME/envs/snakeenv"
if [[ ! -d "$ENV_DIR" ]]; then
    conda env create -p "$ENV_DIR" -f phenotype_mil/envs/snakeenv.yml --yes
fi
conda activate "$ENV_DIR"

# Print node and GPU info
echo "Running on node: $(hostname)"; nvidia-smi


#Run the Snakemake pipeline
SNAKEFILE=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes/phenotype_mil/Snakefile
WORKDIR=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes
CONFIG="--config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True"

snakemake --snakefile "$SNAKEFILE" --directory "$WORKDIR" --unlock || true

snakemake --snakefile "$SNAKEFILE" --directory "$WORKDIR" $CONFIG --use-conda --cores all --rerun-incomplete

#snakemake --snakefile ~/l40_test/DNA-language-models-and-GLP1-microbiome-changes/phenotype_mil/Snakefile --config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True --use-conda --cores all \
#--rerun-incomplete 
#--unlock


echo "Job completed successfully."



