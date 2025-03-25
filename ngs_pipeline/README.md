# SETUP BIOBAKERY METAGENOMICS PIPELINE


1) Make get_conda.sh executable with: `chmod +x get_conda.sh`

2) Run the download script with: `./get_conda.sh`. Once downloaded, restart the shell, by the instructions written to stdout.

3) Create the conda env from global snakemake pipeline configuration "environment.yaml" file : `conda env create -f environment.yaml`

4) Download the human genome bowtie2 index, unpack it, and place into the folder created at `ngs_pipeline/src/databases/bowtie2-index`.

5) Activate the created env: `conda activate snakemake-global`

6) Run the pipeline with: `snakemake --cores all -s pipeline.smk`

