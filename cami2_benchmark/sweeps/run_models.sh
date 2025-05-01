#!/bin/bash

# Check if at least one dataset and one model are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <DATASET1> [DATASET2 ...] -- <MODEL1> [MODEL2 ...]"
    exit 1
fi

# Separate datasets and models using '--' as a delimiter
datasets=()
models=()
delimiter_found=0

for arg in "$@"; do
    if [ "$arg" == "--" ]; then
        delimiter_found=1
        continue
    fi

    if [ "$delimiter_found" -eq 0 ]; then
        datasets+=("$arg")
    else
        models+=("$arg")
    fi
done


# Loop through each dataset and model
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Snakemake for dataset: $dataset and model: $model"

        snakemake --snakefile cami2_benchmark/Snakefile \
                  --config DATASET=$dataset MODEL=$model CHECKM2=True \
                  --use-conda --cores all

        if [ $? -ne 0 ]; then
            echo "Snakemake failed for model: $model in dataset: $dataset"
            exit 1
        fi
    done
done
