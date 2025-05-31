## Phenotype MIL

The code in this folder is used to run the phenotype classification using Multiple Instance Learning (MIL).

The prerequisites for running the code is a a contig catalogue produced in *ngs_pipeline* for a phenotype dataset.

The following processing steps are performed in the Snakefile.

1. Get embeddings and obtain binning results.
2. Process and map contig abundances to cluster abundances.
3. Run MIL and get performance metrics. 


### Datasets 


The full list of possible datasets are specified in the file `ngs_pipeline/config/phenotype_studies.yaml. Due to time-constraints we only used the following datasets:

* [**T2D-EW**](https://pubmed.ncbi.nlm.nih.gov/23719380/): consisting of 96 samples.
* **WEGOVY**: Novel study on WEGOVY from copenhagen consisting 24 patients with 2 samples per patient, resulting in 48 total samples.  

### Models

The list of models are specified in `binning/models.yml`. We only use the following models:

* **VAMB** [github](https://github.com/RasmussenLab/vamb).
* **DNABERT-S**: [github](https://github.com/MAGICS-LAB/DNABERT_S)
* **DNABERT-H** Our DNABERT-H model.



### Usage

#### Setup

Create the environment for Snakemake and activate it.
```bash
conda env create -f phenotype_mil/envs/snakeenv.yml --yes && conda activate snakeenv
```

#### Output



#### Arguments

The Snakemake file controls the workflow. It takes the following args:

* `DATASET` Choose a dataset from the `ngs_pipeline/config/phenotype_studies.yaml`. Only one dataset can be used at a time.
* `MODEL` Choose a model from the `binning/models.yml`. Only one model can be used at a time.
* `CHECKM2` Optional. Set to `True` to run Checkm2 on the model output.
* `MIL_METHODS`. Choose one or more MIL methods: ``knn`` ``classifier``
* `PHENOTYPE_MIL`. Whether to run the MIL. Default to False.



#### Example Runs

Run the following command to get embeddings, binning results, and running knn MIL on T2D-EW dataset using DNABERT-S.


```bash
snakemake --snakefile phenotype_mil/SnakeFile \
 --config DATASET=T2D-EW MODEL=dnaberts CHECKM2=True \
  HAUSDORFF=True PROCESS_ABUNDANCES=True MIL_METHOD knn \
--use-conda --cores all 
```










