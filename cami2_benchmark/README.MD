### Cami2 Benchmark

The code in this folder is used to benchmark models on Cami2 datasets. The datasets in CAMI2 are designed to mimic realistic microbiome environments and include a vast array of both new and known genomes. CAMI2 datasets are often used to benchmark metagenomic binners: https://cami-challenge.org/.


The following steps are performed: 

1. Download the Cami2 Datasets: sample-reads and sample-contigs. Each sample is assembled individually. These files are found on the Cami2 website.
2. All sample-contigs are concatenated into a contig catalogue, using the script `concatenate.py` from VAMB.
3. Sample reads are aligned to the contig catalogue, producing a bam-file for each sample, following the "multi-split" workflow. 
4. A model take as input the contig catalogue and the bam-files, and outputs bins. 
5. CheckM2 is used to evaluate the performance of the model.


#### Datasets

The list of datasets are specified in the file `config.yml`. They include:

* [**marine**](https://cami-challenge.org/datasets/Marine/) `short` (Gold standard assembly from Illumina HiSeq simulation).
* [**plant**](https://cami-challenge.org/datasets/Plant-associated/) `short` (From Illumina HiSeq data simulation).
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **airways** `short`.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **gastro** `short`.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **oral** `short`.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **skin** `short`.
* [**human**](https://cami-challenge.org/datasets/Toy%20Human%20Microbiome%20Project/) **urogenital** `short`.


#### Models

The list of models are specified in the file `config.yml`. They include:

* **VAMB** [github](https://github.com/RasmussenLab/vamb).
* **TaxVAMB** [github](https://github.com/RasmussenLab/vamb).
* **Comebin** [github](https://github.com/ziyewang/COMEBin).

### Output

The output of the workflow is Checkm2 results for the bins. These are found in folder `model_results/DATASET/checkm2`.


#### How to run

The Snakemake file specifies the workflow. It takes the following args:

* `DATASET` Choose a dataset from the `config.yml`. Only one dataset can be used at a time.
* `MODEL` Optional. Choose a model frmo the `contif.yml`. Only one model can be used at a time.
* `DOWNLOAD` Optional. Set to `True` to download the dataset.
* `CONCATENATE` Optional. Set to `True` to run `concatenate.py` from vamb on sample-contigs.
* `ALIGNMENT` Optional. Set to `True` to run `strobealign`.
* `CHECKM2` Optional. Set to `True` to run Checkm2 on the model output.


#### Examples

Run the following command to download, concatenate and align the airways short dataset:

```bash
snakemake --snakefile cami2_benchmark/Snakefile --config DATASET=airways_short DOWNLOAD=True CONCATENATE=True ALIGNMENT=True --use-conda
```

Run the following command to run vamb and benchmark results with checkm2:
```bash
snakemake --snakefile cami2_benchmark/Snakefile --config DATASET=airways_short MODEL=vamb CHECKM2=True --use-conda
```





