"""HDBSCAN snakemake pipeline"""

import os
import glob
configfile: "config/config.yaml"



rule cluster:
    input:
        embeddings = "path to embedddings",
        contig_catalouge = "path to contig catalogue"
    output:
        "clusters.tsv"
    threads:
        -48
    shell:
    """
    
    python -m cuml.accel cluster.py {input.embeddings} {input.contig_catalogue}
   
    """


rule write_fasta:
    input:
        "clusters.tsv"
    output:
        "tmp/{sample}.fasta"
    shell:
    """
    mkdir -p tmp

    python create_fasta.py "catalogue.fna.gz" clusters.tsv 250000 tmp
    """