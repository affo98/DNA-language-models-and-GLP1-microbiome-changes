## Cami2 Benchmark 

The code in this folder is used to benchmark models on metagenomics binning on Cami2 datasets. The datasets in CAMI2 are designed to mimic realistic microbiome environments and include a vast array of both new and known genomes. CAMI2 datasets are often used to benchmark metagenomic binners: https://cami-challenge.org/.

We also include the MetaHIT "error free" dataset from [metabat](https://figshare.com/articles/dataset/MetaHIT_error-free_contigs_from_MetaBAT/27933807?file=50894283).

The following steps are performed: 

1. Download the Cami2 Datasets: sample-reads and sample-contigs. Each sample is assembled individually. These files are found on the Cami2 website.
2. All sample-contigs are concatenated into a contig catalogue, using the script `concatenate.py` from VAMB.
3. Sample reads are aligned to the contig catalogue, producing a bam-file for each sample, following the "multi-split" workflow. 
4. Models *Vamb*, *Taxvamb*, and *Comebin* take as input the contig catalogue and the bam-files, and outputs bins. Models *tnf*, *tnfkernel*, *dna2vec* *dnaberts*, *dnabert2*, and *dnabert2random* take only contig catalogue as input.
5. CheckM2 is used to evaluate the performance of the model.


### Datasets

The list of datasets are specified in the file `config.yml`. All datasets are **short-read**. They include:

* [**marine**](https://cami-challenge.org/datasets/Marine/)  (Simulated Illumina HiSeq metagenome data).
* [**plant**](https://cami-challenge.org/datasets/Plant-associated/)  (Illumina HiSeq metagenome data). 
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **airways**.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **gastro**.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **oral**.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **skin**.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **urogenital**.
* [**MetaHIT "Error Free**](https://figshare.com/articles/dataset/MetaHIT_error-free_contigs_from_MetaBAT/27933807?file=50894283). 


### Models

The list of models are specified in the file `config.yml`. They include:

* **VAMB** [github](https://github.com/RasmussenLab/vamb).
* **TaxVAMB** [github](https://github.com/RasmussenLab/vamb).
* **Comebin** [github](https://github.com/ziyewang/COMEBin).
* **TNF** Simple tetranucleotide frequencies with dim=256
* **TNFKERNEL** TNFs downprojected by a kernel to dim=103 [github](https://github.com/RasmussenLab/vamb/blob/master/src/create_kernel.py)
* **DNA2VEC** [github](https://github.com/pnpnpn/dna2vec).
* **DNABERT-S**: [github](https://github.com/MAGICS-LAB/DNABERT_S)
* **DNABERT-2**: [github](https://github.com/MAGICS-LAB/DNABERT_2)
* **DNABERT-2 Random** DNABERT-2 with randomly initialized weights. 
* **DNABERT-H** Our DNABERT-H model.


### Usage

#### Setup

Create the environment for Snakemake and activate it.
```bash
conda env create -f cami2_benchmark/envs/cami2_processing.yml --yes && conda activate cami2_processing
```

#### Output

1. Contig catalogues and BAM-files are saved in `processed_data/<DATASET_NAME>`
2. The output (results) of the workflow is Checkm2 quality results for the bins. These are found in folder `model_results/<DATASET_NAME>/checkm2`. Run the command below to parse checkm2 results for all datasets and models, as well as knn-histograms, and contig-length histograms. The parsed results are saved in `model_results/parsed_results/parsed_checkm2_results.json`

```bash
python cami2_benchmark/src/parse_all_results.py
```



#### Arguments

The Snakemake file controls the workflow. It takes the following args:

* `DATASET` Choose a dataset from the `config.yml`. Only one dataset can be used at a time.
* `MODEL` Optional. Choose a model from the `config.yml`. Only one model can be used at a time.
* `DOWNLOAD` Optional. Set to `True` to download the dataset.
* `CONCATENATE` Optional. Set to `True` to run `concatenate.py` from vamb on sample-contigs.
* `ALIGNMENT` Optional. Set to `True` to run `strobealign`.
* `CHECKM2` Optional. Set to `True` to run Checkm2 on the model output.



#### Example Runs

Run the following command to download, concatenate and align the airways short dataset:

```bash
snakemake --snakefile cami2_benchmark/Snakefile \
  --config DATASET=airways_short DOWNLOAD=True CONCATENATE=True ALIGNMENT=True \
  --use-conda --cores all 
```

Run the following command to run vamb and benchmark results with checkm2:
```bash
snakemake --snakefile cami2_benchmark/Snakefile \
   --config DATASET=airways_short MODEL=vamb CHECKM2=True \
   --use-conda --cores all 
```


#### Sweeps

To run multiple datasets/models, see folder `sweeps` that contains 
* `process_data.sh`: Runs `DOWNLOAD`, `CONCATENATE`, and `ALIGNMENT` on one or more datasets.
*  `run_models.sh`: Runs binning and `CHECKM2` for one or more models on a dataset.

**Example run to download and process ``airways_short`` and ``gastro_short`` datasets**
```bash
bash cami2_benchmark/sweeps/process_data.sh airways_short gastro_short
```

**Example run with LLM models on ``metahit`` dataset**
```bash
bash cami2_benchmark/sweeps/run_models.sh metahit dnaberts dnabert2 dnabert2random
```










