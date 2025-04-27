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

#Conda setup---------------------------------------------------------------------------
MINICONDA_DIR="$HOME/miniconda"
CONDA_VERSION="24.7.1"

# Check if Conda is installed and at the required version
if command -v conda &>/dev/null; then
    INSTALLED_CONDA_VERSION=$(conda --version | awk '{print $2}')
    if [[ "$INSTALLED_CONDA_VERSION" == "$CONDA_VERSION" ]]; then
        echo "Conda version $CONDA_VERSION is already installed."
    else
        echo "Updating Conda to version $CONDA_VERSION..."
        conda install -c conda-forge conda=$CONDA_VERSION --yes
    fi
else
    echo "Conda not found. Installing Miniconda $CONDA_VERSION..."

    # Download Miniconda installer for Linux
    INSTALLER="Miniconda3-py39_$(uname -m).sh"
    if [[ ! -f "$INSTALLER" ]]; then
        wget "https://repo.anaconda.com/miniconda/$INSTALLER"
    fi

    # Install Miniconda silently
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"

    # Remove the installer
    rm "$INSTALLER"

    # Add Conda to PATH
    export PATH="$MINICONDA_DIR/bin:$PATH"

    # Install the required Conda version
    conda install -c conda-forge conda=$CONDA_VERSION --yes
fi

# Initialize Conda
source "$MINICONDA_DIR/bin/conda" init bash

# Ensure Conda environment is activated
source "$MINICONDA_DIR/bin/activate"
#--------------------------------------------------------------------------

conda env create -f phenotype_mil/envs/snakeenv.yml --yes && conda activate snakeenv
pip uninstall -y triton

# Print node and GPU info
echo "Running on node: $(hostname)"
echo "CUDA devices:"
nvidia-smi

#Run the Snakemake pipeline
snakemake --snakefile phenotype_mil/SnakeFile --config DATASET=T2D-EW MODEL=dnaberts --use-conda --cores all -np

echo "Job completed successfully."