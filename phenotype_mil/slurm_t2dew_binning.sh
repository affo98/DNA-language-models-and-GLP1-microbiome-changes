#!/bin/bash

#SBATCH --job-name=t2dew # Job name
#SBATCH --output=t2dew%j.out
#SBATCH --error=slurm%j.err 
#SBATCH --nodes=1
#SBATCH #--exclude=cn[3-18]
#SBATCH #--exclude=desktop[1-16]
#SBATCH #--time=5-00:00:00
#SBATCH #--nodelist=cn19
#SBATCH #--gres=gpu:l40s:4
#SBATCH --partition=purrlab_students
#SBATCH --exclusive

#
#chmod +x get_conda.sh
# bash get_conda.sh
# source ~/.bashrc


#Load conda
module load Anaconda3
module load CUDA/12.1.1
conda env create -f phenotype_mil/envs/snakeenv.yml --yes && source activate snakeenv
pip uninstall -y triton

# Print node and GPU info
echo "Running on node: $(hostname)"
echo "CUDA devices:"
nvidia-smi

#Run the Snakemake pipeline
snakemake --snakefile phenotype_mil/SnakeFile --config DATASET=T2D-EW MODEL=dnaberts --use-conda --cores all -np

echo "Job completed successfully."