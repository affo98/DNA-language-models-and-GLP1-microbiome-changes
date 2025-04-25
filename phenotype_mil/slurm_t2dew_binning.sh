#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --output=t2dew.out
#SBATCH --exclude=cn[3-18]
#SBATCH --exclude=desktop[1-16]
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=5-00:00:00
#SBATCH --nodelist=cn19
#SBATCH --partition=purrlab_students

nvidia-smi

chmod +x get_conda.sh
bash get_conda.sh
source ~/.bashrc

conda env create -f phenotype_mil/envs/snakeenv.yml --yes && conda activate snakeenv

snakemake --snakefile phenotype_mil/SnakeFile --config DATASET=T2D-EW MODEL=dnaberts --use-conda --cores all