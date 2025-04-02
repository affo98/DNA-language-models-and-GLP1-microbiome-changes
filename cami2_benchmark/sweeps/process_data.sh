#!/bin/bash

# Check if at least one dataset is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <DATASET1> [DATASET2] ... [DATASETN]"
    exit 1
fi

# Loop over each dataset
for dataset in "$@"; do
    echo "Processing dataset: $dataset"
    
    snakemake --snakefile cami2_benchmark/Snakefile \
              --config DATASET="$dataset" DOWNLOAD=True CONCATENATE=True ALIGNMENT=True \
              --use-conda

    # Check if Snakemake was successful
    if [ $? -ne 0 ]; then
        echo "Snakemake failed for dataset: $dataset"
        exit 1
    fi
done

