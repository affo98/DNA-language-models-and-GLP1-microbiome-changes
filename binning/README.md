## Binning 

This folder contains code to run metagenomics binning using embeddings from LLM models or simpler models such as TNF.

The following steps are performed: 

1. 







### Models

The list of models are specified in the file `models.yml`. They include:

* **TNF** Simple tetranucleotide frequencies with dim=256
* **TNFKERNEL** TNFs downprojected by a kernel to dim=103 [github](https://github.com/RasmussenLab/vamb/blob/master/src/create_kernel.py)
* **DNA2VEC** [github](https://github.com/pnpnpn/dna2vec).
* **DNABERT-S**: [github](https://github.com/MAGICS-LAB/DNABERT_S)
* **DNABERT-2**: [github](https://github.com/MAGICS-LAB/DNABERT_2)
* **DNABERT-2 Random** DNABERT-2 with randomly initialized weights. 
