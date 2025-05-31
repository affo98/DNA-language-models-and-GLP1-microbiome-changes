## Binning 

This folder contains code to perform metagenomics binning of contigs using **KMediod clustering** and a **pretrained embedding model** or simple model such as TNF. 

The following steps are performed: 

1. Reads in contigs from a fasta file and splits them into val/test.
2. Computes embeddings for the contigs using a model.
3. Computes a threshold for the Kmediod algorithm using KNN.
4. Runs the Kmediod algorithm using the threshold.
 

### Threshold

The KMediod algorithm requires a threshold parameter, that is computed by selecting KNNs for contig, and calculating distances wihin these neighbours. A good threshold is crucial for the Kmediod algorithm, and requires setting a good `k` and  `p`.

The binning script supports hyperparameter search for optimal `k` and `p` values in when using `-m val` and can be used for binning in test mode with `-m test`.

The recommended usage is to first run the binning script with a hyperparameter search for optimal `k` and `p` and evaluate cluster results e.g. with CheckM2. And afterwards run the binning with the selected `k` and `p`. See more the Snakemake file in the directory `cami2_benchmark`.


### Models

The list of models are specified in the file `models.yml`. They include:

* **TNF** Simple tetranucleotide frequencies with dim=256
* **TNFKERNEL** TNFs downprojected by a kernel to dim=103 [github](https://github.com/RasmussenLab/vamb/blob/master/src/create_kernel.py)
* **DNA2VEC** [github](https://github.com/pnpnpn/dna2vec).
* **DNABERT-S**: [github](https://github.com/MAGICS-LAB/DNABERT_S)
* **DNABERT-2**: [github](https://github.com/MAGICS-LAB/DNABERT_2)
* **DNABERT-2 Random** DNABERT-2 with randomly initialized weights. 
* **DNABERT-H**: Our DNABERT-H model.


### Output

* Cluster Results: Stored in ``<save_path>/cluster_results/``.
* Embeddings: Saved in ``<save_path>/embeddings/``.
* Threshold Histograms: Saved in`` <save_path>/threshold_histograms/``.
* Log File: Saved in ``<save_path>`` Tracks execution details and errors.


### Usage

#### Setup

Create conda environment and activate it.
```bash
conda env create -f binning/envs/binning.yml
conda activate binning
```


#### Arguments


```bash
python binning.py \
  --contigs <contig_file> \
  --model_name <model_name> \
  --model_path <model_path> \
  --batch_sizes <batch_size1> <batch_size2> ... \
  --knnk <k_value1> <k_value2> ... \
  --knnp <p_value1> <p_value2> ... \
  --save_path <output_directory> \
  --log <log_file> \
  --mode <val|test>
```

`model_name`, `model_path` and `batch_sizes` are specified in ``models.yml``.

#### Example usage

**Hyperparameter Search (Validation Mode)**
Input multiple `knnk` and `knnp`.
```bash
python binning.py \
  --contigs <contig_file> \
  --model_name dnaberts \
  --model_path zhihan1996/DNABERT-S \
  --batch_sizes 40 8 1 \
  --knnk 100 20 \
  --knnp 50 70 \
  --save_path results/ \
  --log binning.log \
  --mode val
```

**Binning in Test Mode**
Only using one `knnk` and `knnp`.
```bash
python binning.py \
  --contigs <contig_file> \
  --model_name dnaberts \
  --model_path zhihan1996/DNABERT-S \
  --batch_sizes 40 8 1 \
  --knnk 100 \
  --knnp 50 \
  --save_path results/ \
  --log binning.log \
  --mode val
```



Note: the binning script is used in the Snakemake files in the `cami2_processing` and `phenotype_mil` directories.