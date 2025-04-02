"""HDBSCAN snakemake pipeline"""


rule cluster:
    input:
        embeddings = "dnaberts.npy",
        contig_catalogue = "catalogue.fna.gz"
    output:
        "clusters.tsv"
    threads:
        48
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


rule checkm2:
    input:
        "tmp/{sample}.fasta",
    output:
        directory("checkm2_results"),
    params:
        db = "db/checkm2_database"
    threads:
        48
    shell:
        """
        if [ ! -d "{params.db}" ]; then
                checkm2 database --download --path {params.db}
        fi
            checkm2 predict\ 
                --threads {threads}\
                --input {input}\
                --output-directory {output}\
                --database_path {params.db}/CheckM2_database/uniref100.KO.1.dmnd
        """

rule parse_checkm2_results:
    input:
        "checkm2_results",
    output:
        "checkm2_validation_results",
    shell:
        """
        mkdir -p {output}
        python parse_checkm2_val.py -i {input} -o {output}
        """