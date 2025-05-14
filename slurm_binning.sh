#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --partition=purrlab_students
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00


#-----------------1. Pick Node-----------------------

#1a For 4 L40s
# #SBATCH --output=t2dew_l40_%j.out
# #SBATCH --error=slurm_l40_%j.err 
# #SBATCH --exclude=cn[3-18]
# #SBATCH --exclude=desktop[1-16]
# #SBATCH --nodelist=cn19
# #SBATCH --gres=gpu:l40s:4


#1b For small testing
#SBATCH --output=t2dew_small_%j.out
#SBATCH --error=slurm_small_%j.err 
#SBATCH --partition=scavenge
#SBATCH --time=04:00:00


#1c Using 2x H100 
# #SBATCH --output=t2dew_h100_%j.out
# #SBATCH --error=slurm_h100_%j.err 
# #SBATCH --exclude=cn[1-10]
# #SBATCH --exclude=cn[12-19]
# #SBATCH --exclude=desktop[1-16]
# #SBATCH --nodelist=cn11                        
# #SBATCH --gres=gpu:h100:2


#1b Using 4x A100 (40gb) 
# #SBATCH --output=t2dew_a100_%j.out
# #SBATCH --error=slurm_a100_%j.err 
# #SBATCH --exclude=cn[3-12]
# #SBATCH --exclude=cn[14-19]
# #SBATCH --exclude=desktop[1-16]
# #SBATCH --nodelist=cn13                        
# #SBATCH --gres=gpu:a100_40gb:4


#---------------------Setup-------------------------
# Download & install Miniconda3 if missing
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

# rm -rf ~/miniconda/pkgs/ffmpeg-4.4.2-gpl_hdf48244_113
conda clean --index-cache --packages --tarballs --yes #uncomment if problems with conda envs


#--------------------------2. Pick job-------------------------

#2a Run sweeps on cami2_benchmark
#chmod +x cami2_benchmark/sweeps/run_models.sh
#cami2_benchmark/sweeps/run_models.sh airways_short gastro_short oral_short skin_short urogenital_short marine_short plant_short metahit -- dnaberth_2mv1_150k

#2b Run the Snakemake pipeline
SNAKEFILE=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes/phenotype_mil/Snakefile
WORKDIR=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes
CONFIG="--config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True"

# Ensure any stale Snakemake lock is removed
#echo "Unlocking any stale Snakemake locks..."
#snakemake --unlock --directory "$WORKDIR" --quiet

snakemake --snakefile "$SNAKEFILE" --directory "$WORKDIR" $CONFIG --use-conda --cores all --rerun-incomplete --rerun-triggers mtime #add --unlock here if necessary



echo "Job completed successfully."



