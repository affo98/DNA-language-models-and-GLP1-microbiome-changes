### Get Cami2 Data

This folder contains a script to obtain contigs from specific datasets in the CAMI2.

The datasets in CAMI2 are designed to mimic realistic microbiome environments and include a vast array of both new and known genomes. CAMI2 datasets are often used to benchmark metagenomic binners.
https://cami-challenge.org/

We include the following datasets:

* [**marine**](https://cami-challenge.org/datasets/Marine/) **short** (Gold standard assembly from Illumina HiSeq simulation)
* [**marine**](https://cami-challenge.org/datasets/Marine/) **long** (Gold standard assembly from Pacific Bioscience simulation)
* [**plant**](https://cami-challenge.org/datasets/Plant-associated/) **short** (From Illumina HiSeq data simulation)
* [**plant**](https://cami-challenge.org/datasets/Plant-associated/) **long** (From Pacific Bioscience data simulation)

The output path of processed contigs are `data/cami2`.

Run the following command to obtain processed contigs for the datasets mentioned above. 
```bash
python ./get_cami2_data/get_cami2_data.py
```

Parameters:

- `-d`: Choose Cami2 datasets. E.g. marine_long, plant_short. Optional. Default to all datasets above.





