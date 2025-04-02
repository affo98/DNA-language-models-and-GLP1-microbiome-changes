#!/bin/bash

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATASET> <MODEL1> [MODEL2] [MODEL3] ..."
    exit 1
fi

# Extract dataset from CLI arguments
dataset="$1"
shift  # Remove first argument (dataset) from the list

# Remaining arguments are the models
models=("$@")

# Loop over each model and run Snakemake
for model in "${models[@]}"; do
    echo "Running Snakemake for dataset: $dataset and model: $model"

    snakemake --snakefile cami2_benchmark/Snakefile \
              --config DATASET=$dataset MODEL=$model CHECKM2=True \
              --use-conda

    # Check if Snakemake was successful
    if [ $? -ne 0 ]; then
        echo "Snakemake failed for model: $model in dataset: $dataset"
        exit 1
    fi
done

echo "All models processed successfully for dataset: $dataset!"
