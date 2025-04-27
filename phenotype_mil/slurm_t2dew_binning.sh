#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --output=t2dew%j.out
#SBATCH --error=slurm%j.err 
#SBATCH --partition=purrlab_students
#SBATCH --exclusive

# #SBATCH --nodes=1
# #SBATCH --exclude=cn[3-18]
# #SBATCH --exclude=desktop[1-16]
# #SBATCH --time=5-00:00:00
# #SBATCH --nodelist=cn19
# #SBATCH --gres=gpu:l40s:4


# 1. Download & install Anaconda3 if missing
if [[ ! -d "$HOME/anaconda3" ]]; then
    curl -sO https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p "$HOME/anaconda3"
    rm Anaconda3-2024.10-1-Linux-x86_64.sh
fi

# 2. Initialize in-script
export PATH="$HOME/anaconda3/bin:$PATH"
eval "$(conda shell.bash hook)" 2>/dev/null

# 3. Create & activate your env
ENV_DIR="$HOME/envs/snakeenv"
if [[ ! -d "$ENV_DIR" ]]; then
    conda env create -p "$ENV_DIR" -f phenotype_mil/envs/snakeenv.yml --yes
fi
conda activate "$ENV_DIR"

# Print node and GPU info
echo "Running on node: $(hostname)"; nvidia-smi


#Run the Snakemake pipeline
snakemake --snakefile phenotype_mil/SnakeFile --config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True --use-conda --cores all -np

echo "Job completed successfully."



