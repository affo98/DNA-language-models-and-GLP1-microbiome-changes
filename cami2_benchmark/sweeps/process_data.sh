#!/bin/bash

# Check if at least one dataset is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <DATASET1> [DATASET2] ... [DATASETN]"
    exit 1
fi

# Convert datasets into a comma-separated list for Snakemake
DATASETS=$(IFS=,; echo "$*")

# Run Snakemake for all datasets in one command
snakemake --snakefile cami2_benchmark/Snakefile \
          --config DATASET="$DATASETS" DOWNLOAD=True CONCATENATE=True ALIGNMENT=True \
          --use-conda

echo "All datasets processed successfully!"
