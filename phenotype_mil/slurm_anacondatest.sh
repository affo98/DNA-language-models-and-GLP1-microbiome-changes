#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --output=t2dew%j.out
#SBATCH --error=slurm%j.err 
#SBATCH --partition=scavenge
#SBATCH --time=04:00:00


module load Anaconda3


ENV_DIR="$HOME/envs/snakeenv"
if [[ ! -d "$ENV_DIR" ]]; then
    conda env create -p "$ENV_DIR" -f phenotype_mil/envs/snakeenv.yml --yes
fi
conda activate "$ENV_DIR"

echo "Running on node: $(hostname)"; nvidia-smi


#Run the Snakemake pipeline
SNAKEFILE=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes/phenotype_mil/Snakefile
WORKDIR=~/l40_test/DNA-language-models-and-GLP1-microbiome-changes
CONFIG="--config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True"

snakemake --snakefile "$SNAKEFILE" --directory "$WORKDIR" $CONFIG --use-conda --cores all --rerun-incomplete




