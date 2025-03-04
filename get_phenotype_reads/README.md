### Get Phenotype Reads

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

For study_ids `PRJDB3601`, `PRJNA448494`, `PRJEB12123`, `PRJEB21528`, and `PRJNA422434`, sample labels are obtained from the GMREPO, which is a curated database of reads and metadata from gut microbiome studies. We downloaded the metadata for each study respectively. They are saved in `.data/sample_labels_raw/gmrepo/`
Reference: https://gmrepo.humangut.info/home


For study_id `PRJEB1786`, and `PRJEB4336`, sample labels are obtained from the `abundance.txt` file from the Github Repo of MetAML study. Many other phenotype prediction studies also uses this as a data-source for sample labels. The file is saved in `./data/metaml/abundance.txt` 
File source.txt: https://github.com/SegataLab/metaml/tree/master/data
Reference: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004977

