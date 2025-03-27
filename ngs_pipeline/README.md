# SETUP BIOBAKERY METAGENOMICS PIPELINE


1) Make get_conda.sh executable with: `chmod +x get_conda.sh`

2) Run the download script with: `./get_conda.sh`. Once downloaded, restart the shell, by the instructions written to stdout.

3) Create the conda env from global snakemake pipeline configuration "environment.yaml" file : `conda env create -f pipeline.yaml` and activate it with `conda activate snakemake-global`

4) Download the human genome bowtie2 index, unpack it, and place into the folder created at `ngs_pipeline/src/databases/bowtie2-index`.


```
wget https://huttenhower.sph.harvard.edu/kneadData_databases/Homo_sapiens_hg39_T2T_Bowtie2_v0.1.tar.gz
tar -xzf Homo_sapiens_hg39_T2T_Bowtie2_v0.1.tar.gz
rm Homo_sapiens_hg39_T2T_Bowtie2_v0.1.tar.gz
mv bowtie2-index src/databases/bowtie2-index/

```

5) Activate the created env: `conda activate snakemake-global`

6) Run the pipeline with: `python run_pipeline.py --snakemake-args --jobs 2 --use-conda`


# Get Phenotype Reads

This folder contains all script and data necessary to obtain raw reads from studies and the according sample labels (healthy/disease in each study). 

The list of included studies (id, name) are specified in the file `config/phenotype_studies.yml`. They include:

| Disease   | ENA         | Paper Link                                      | Year |
|-----------|-------------|-------------------------------------------------|------|
| Obesity   | PRIEB4336   | [Link](https://pubmed.ncbi.nlm.nih.gov/23985870/) | 2013 |
| Obesity   | PRIEB12123  | [Link](https://pubmed.ncbi.nlm.nih.gov/28628112/) | 2017 |
| T2D       | PRJNA422434        | [Link](https://pubmed.ncbi.nlm.nih.gov/23023125/) | 2012 |
| T2D       | PRJNA448494 |                                                 |      |
| T2D       | PRIEB1786   | [Link](https://pubmed.ncbi.nlm.nih.gov/23719380/) | 2013 |
| Cardio    | PRIEB21528  | [Link](https://pubmed.ncbi.nlm.nih.gov/29018189/) | 2017 |

The output path of reads and sample labels are `data/phenotype`.


Run the following command to obtain reads and sample labels for all studies:
```bash
python ./get_phenotype_reads/download_fastq_from_studies.py
```

Parameters:

- `-s`: Select to run on specific study_ids. Optional. Default to all studies.


#### Sample Labels

Sample labels are unfortuantly not available as metadata on ENA. Below is a description of how sample labels were obtained for each study. 

For study_ids `PRJNA448494`, `PRJEB12123`, `PRJEB21528`, and `PRJNA422434`, sample labels are obtained from the GMREPO, which is a curated database of reads and metadata from gut microbiome studies. We downloaded the metadata for each study respectively. They are saved in `.data/sample_labels_raw/gmrepo/`
Reference: https://gmrepo.humangut.info/home


For study_id `PRJEB1786`, and `PRJEB4336`, sample labels are obtained from the `abundance.txt` file from the Github Repo of MetAML study. Many other phenotype prediction studies also uses this as a data-source for sample labels. The file is saved in `./data/metaml/abundance.txt` 
File source.txt: https://github.com/SegataLab/metaml/tree/master/data
Reference: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004977

