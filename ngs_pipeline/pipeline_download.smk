"""Local Snakemake pipeline for testing reads, mapping, sorting, indexing, alignment, and quality checks"""
import os
import glob
configfile: "config/config.yaml"
configfile: "config/phenotype_studies.yaml"

PY_SCRIPTS = config['PY_SCRIPTS']
CONDA_ENVS = config['CONDA_ENVS']
DB = config['DB']
FASTQC_PATH = config['FASTQC_PATH']

STUDY_ID = config["studies"][3].get("id")
STUDY_NAME = config["studies"][3].get("name")


OUTDIR = f"{STUDY_NAME}_{STUDY_ID}"
DATAPATH = os.path.join("raw_data", "reads", OUTDIR)
print(DATAPATH)
BENCHMARKS = os.path.join(OUTDIR, "benchmarks")
LOGS = os.path.join(OUTDIR,"logs")

def get_samples(wildcards):
    checkpoint_output = checkpoints.download.get().output[0]
    samples = glob.glob(f"{checkpoint_output}/*/1.fastq.gz")
    sample_names = []
    for sample in samples:
        sample_name = sample.split("/")[-2].split("_")[0][5:]
        sample_names.append(sample_name)
    return sample_names



rule all:
    input:
        os.path.join(OUTDIR, "abdn_coverm/abundances.tsv"),

checkpoint download:
    output:
        directory(DATAPATH)
    conda:
        os.path.join(CONDA_ENVS, "get_phenotype_reads.yaml"),
    params:
        dataset_id = STUDY_ID,
        dataset_name = STUDY_NAME,
        get_reads_py = os.path.join(PY_SCRIPTS, "get_phenotype_reads.py"),
    shell:
        """
        python {params.get_reads_py} -i {params.dataset_id} -n {params.dataset_name}
        """

rule fastqc:
    input: 
        r1 = os.path.join(DATAPATH, "SAMEA{sample}/1.fastq.gz"),
        r2 = os.path.join(DATAPATH, "SAMEA{sample}/2.fastq.gz")
    output:
        html1 = os.path.join(OUTDIR, "fastqc/{sample}/1_fastqc.html"),
        zip1 = os.path.join(OUTDIR, "fastqc/{sample}/1_fastqc.zip"),
        html2 = os.path.join(OUTDIR, "fastqc/{sample}/2_fastqc.html"),
        zip2 = os.path.join(OUTDIR, "fastqc/{sample}/2_fastqc.zip")
    params:
        output_dir = os.path.join(OUTDIR, "fastqc"),
        tmp_dir = os.path.join(OUTDIR, "tmp/{sample}", "fastqc")
    log:
        os.path.join(LOGS, "fastqc/{sample}.log")
    threads:
        16
    conda:
        os.path.join(CONDA_ENVS, "fastqc.yaml")
    shell:
        """
        mkdir -p {params.tmp_dir}

        (fastqc -t {threads} {input.r1} {input.r2} -o {params.tmp_dir}) 2> {log}

        cp {params.tmp_dir}/1_fastqc.html {output.html1}
        cp {params.tmp_dir}/1_fastqc.zip {output.zip1}
        cp {params.tmp_dir}/2_fastqc.html {output.html2}
        cp {params.tmp_dir}/2_fastqc.zip {output.zip2}

        rm -rf tmp
        """

rule detect_adapter:
    input: 
        os.path.join(OUTDIR, "fastqc/{sample}/1_fastqc.zip")
    output: 
        os.path.join(OUTDIR, "fastqc/{sample}/adapters.txt")
    log: 
        os.path.join(LOGS, "detect_adapter/{sample}.log")
    params:
        detect_adapters = os.path.join(PY_SCRIPTS, "detect_adapters.py")    
    shell:
        """
        python {params.detect_adapters} {input} > {output}
        """


rule kneaddata:
    input:
        r1 = os.path.join(DATAPATH, "SAMEA{sample}/1.fastq.gz"),
        r2 = os.path.join(DATAPATH, "SAMEA{sample}/2.fastq.gz"),
        adapter=os.path.join(OUTDIR, "fastqc/{sample}/adapters.txt")
    output:
        p1 = os.path.join(OUTDIR, "knead/{sample}/paired_1.fastq"),
        p2 = os.path.join(OUTDIR, "knead/{sample}/paired_2.fastq"),
    log:
        os.path.join(LOGS, "knead/{sample}.log")
    threads: 64
    conda:
        os.path.join(CONDA_ENVS, "kneaddata.yaml"),
    benchmark:
        os.path.join(BENCHMARKS, "kneaddata", "{sample}.txt"),
    params:
        tmp_dir = os.path.join(OUTDIR, "knead/{sample}/tmp"),
        phred="phred33",
        trim="SLIDINGWINDOW:5:20",
        fastqc_path=FASTQC_PATH,
        db_path=DB # path to human genome bowtie index
    shell:
         """
         mkdir -p {params.tmp_dir}

         kneaddata\
         --input1 {input.r1}\
         --input2 {input.r2}\
         -db {params.db_path}\
         --sequencer-source $(cat {input.adapter})\
         --fastqc {params.fastqc_path}\
         --run-fastqc-end\
         --run-trim-repetitive\
         --quality-scores {params.phred}\
         --trimmomatic-options {params.trim}\
         --threads {threads}\
         -o {params.tmp_dir}\

         cp {params.tmp_dir}/1_kneaddata_paired_1.fastq {output.p1}
         cp {params.tmp_dir}/1_kneaddata_paired_2.fastq {output.p2}
        
         rm -rf {params.tmp_dir}
        """

rule metaspades: 
    input:
        r1=os.path.join(OUTDIR, "knead/{sample}/paired_1.fastq"),
        r2=os.path.join(OUTDIR, "knead/{sample}/paired_2.fastq")
    output:
        contigs=os.path.join(OUTDIR, "spades/asm_{sample}/contigs.fasta"),
    benchmark:
        os.path.join(BENCHMARKS, "assembly", "{sample}.txt"),
    params:
        k="auto", #k-mer size
        m="100", #memory 100gb vamb
        tmp_dir = os.path.join(OUTDIR, "spades/asm_{sample}/tmp"),
    log:
        os.path.join(LOGS, "metaspades/{sample}.log")
    threads: 64 # 24 vamb
    conda:
        os.path.join(CONDA_ENVS, "metaspades.yaml")
    shell:
        """
        spades.py --meta -1 {input.r1} -2 {input.r2} -m {params.m} -o {params.tmp_dir} -k {params.k} -t {threads} --only-assembler
        
        cp {params.tmp_dir}/contigs.fasta {output.contigs}

        rm -rf {params.tmp_dir}
        """ 

    
# remember to change naming scheme in concatenate.py depending on dataset.
rule concatenate:
    input:
        lambda wildcards: expand(
            os.path.join(OUTDIR, "spades/asm_{sample}/contigs.fasta"),
            sample=get_samples(wildcards)
        )
    output:
        os.path.join(OUTDIR, "global_contig_catalogue.fna.gz")
    conda:
        os.path.join(CONDA_ENVS, "concatenate.yaml")
    params:
        concatenate = os.path.join(PY_SCRIPTS, "concatenate.py")
    threads:
        64
    shell:
        """
        python {params.concatenate} {output} {input} -m 2000
        """

rule alignment:
    input:
        r1=os.path.join(OUTDIR, "knead/{sample}/paired_1.fastq"),
        r2=os.path.join(OUTDIR, "knead/{sample}/paired_2.fastq"),
        contig_catalouge=os.path.join(OUTDIR,"global_contig_catalogue.fna.gz")
    output:
        os.path.join(OUTDIR, "algn/{sample}_sorted.bam"),
    benchmark:
        os.path.join(BENCHMARKS, "alignment", "{sample}.txt")
    threads:
        48
    resources:
        mem_gb=60
    conda:
        os.path.join(CONDA_ENVS, "strobealign.yaml")
    shell:
        """
        strobealign -t {threads} {input.contig_catalouge} {input.r1} {input.r2} | samtools sort -o {output}
        """
    
rule get_coverage:
    input:
        os.path.join(OUTDIR, "algn/{sample}_sorted.bam"),
    output:
        os.path.join(OUTDIR, "abdn_coverm/{sample}.tsv"),
    threads:
        64
    conda:
        os.path.join(CONDA_ENVS, "coverm.yaml")
    params:
        coverage_py = os.path.join(PY_SCRIPTS, "calc_coverage.py")
    shell:
        """
        python {params.coverage_py} {input} {output} {threads}
        """

rule merge_abundances:
    input:
        lambda wildcards: expand(
            os.path.join(OUTDIR, "abdn_coverm/{sample}.tsv"),
            sample=get_samples(wildcards)
        )
    output:
        os.path.join(OUTDIR, "abdn_coverm/abundances.tsv")
    params:
        # merge abundances from vamb takes in a directory of tsv files.
        abdn_path = os.path.join(OUTDIR, "abdn_coverm"),
        merge_aemb = os.path.join(PY_SCRIPTS, "merge_aemb.py")
    conda:
        os.path.join(CONDA_ENVS, "concatenate.yaml"),
    shell:
        """
        python {params.merge_aemb} {params.abdn_path} {output}
        """

