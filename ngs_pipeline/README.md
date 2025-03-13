# SETUP BIOBAKERY METAGENOMICS PIPELINE


1) Make get_conda.sh executable with: `chmod +x get_conda.sh`

2) Run the download script with: `./get_conda.sh`. Once downloaded, restart the shell, by the instructions written to stdout.

2.5) Debugging conda steps to integrate with shell for macos: `conda init zsh`, `source ~/.zshrc`, for bash: `conda init bash` `source ~/.bashrc`.

2.6) Install mamba with conda from conda-forge channel: `conda install -n base -c conda-forge mamba`

3) Create the conda env from global snakemake pipeline configuration "environment.yaml" file : `conda env create -f environment.yaml`

4) Activate the created env: `conda activate snakemake-global`

5) Make `merge_abundances` executable with cmd: `chmod +x merge_abundances.sh`


6) Run the pipeline with: `snakemake --cores all -s pipeline.smk`

