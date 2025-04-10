"""HDBSCAN snakemake pipeline"""


rule all:
    input:
        "hdbscan_results.txt"

rule cluster:
    input:
        embeddings = "dnaberts.npy",
        contig_catalogue = "catalogue.fna.gz"
    output:
        "clusters.tsv"
    threads:
        36
    resources:
        mem_gb=75
    shell:
        """
        python cluster.py {input.embeddings} {input.contig_catalogue} {threads}
        """


rule write_fasta:
    input:
        "clusters.tsv"
    output:
       directory("tmp")
    params:
        contig_catalogue = "catalogue.fna.gz"
    shell:
        """
        mkdir -p tmp

        python create_fasta.py {params.contig_catalogue} clusters.tsv 250000 tmp --log fasta_logs
        """


rule checkm2:
    input:
        "tmp"
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
        "hdbscan_results.txt",
    shell:
        """
        python parse_checkm2_val.py -i {input}/quality_report.tsv
        """