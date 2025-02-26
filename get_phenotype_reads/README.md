### Get Phenotype Reads

This folder contains all script and data necessary to obtain raw reads from studies and the according sample labels (healthy/disease in each study). 

See the list of included studies (id, name) in the file `config/phenotype_studies.yml`

The output path of reads and sample labels are `data/phenotype`.


Run the following command to obtain reads and sample labels for all studies:
```bash
python ./get_phenotype_reads/download_fastq_from_studies.py
```

Parameters:

- `-s`: Select to run on specific study_ids. Optional. Default to all studies.


#### Sample Labels

Sample labels are unfortuantly not available as metadata on ENA. Below is a description of how sample labels were obtained for each study. 

For study_ids `PRJDB3601`, `PRJNA448494`, `PRJEB12123`, `PRJEB21528`, and `PRJNA422434`, sample labels are obtained from the GMREPO, which is a curated database of reads and metadata from gut microbiome studies. We downloaded the metadata for each study respectively.
Reference: https://gmrepo.humangut.info/home


For study_id `PRJEB1786`, and `PRJEB4336`, sample labels are obtained from the `abundance.txt` file from the Github Repo of MetAML study. Many other phenotype prediction studies also uses this as a data-source for sample labels. 
abundance.txt: https://github.com/SegataLab/metaml/tree/master/data
Reference: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004977

