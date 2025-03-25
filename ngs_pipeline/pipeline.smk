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
BENCHMARKS = os.path.join(OUTDIR, "benchmarks")
LOGS = os.path.join(OUTDIR,"logs")

rule all:
    input:
        os.path.join(OUTDIR, "abdn_coverm/abundances.tsv"),
        directory(DATAPATH),

rule download:
    output:
        directory(DATAPATH),
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

# Retrieve Sample Paths
samples = glob.glob(f"{DATAPATH}/*/*/*_1.fastq.gz")
# Construct Wildcards
READS = ["1","2"]
SAMPLES_LU = {}
SAMPLES = []
for sample in samples:
    sample_name = sample.split("/")[-1].split("_")[0]
    SAMPLES_LU[sample_name] = sample
    SAMPLES.append(sample_name)

print("\n"*2)
print("\t"*1,"#"*100)
print("\n")
print("\t"*5,f"# Dataset: {STUDY_NAME} ({STUDY_ID})")
print("\t"*5,f"# Total Samples Found: {len(SAMPLES)}")
print("\n")
print("\t"*1,"#"*100)
print("\n"*2)

rule fastqc:
    input: 
        r1 = lambda wildcards: SAMPLES_LU[wildcards.sample],
        r2 = lambda wildcards: SAMPLES_LU[wildcards.sample].replace("_1.fastq.gz", "_2.fastq.gz")
    output:
        html = os.path.join(OUTDIR, "fastqc/{sample}_{reads}_fastqc.html"),
        zip = os.path.join(OUTDIR, "fastqc/{sample}_{reads}_fastqc.zip"),
        
    params:
        output_dir = os.path.join(OUTDIR, "fastqc"),
        tmp_dir = os.path.join(OUTDIR, "tmp/{sample}", "fastqc")
    log:
        os.path.join(LOGS, "fastqc/{sample}_{reads}.log")
    threads:
        64
    conda:
        os.path.join(CONDA_ENVS, "fastqc.yaml")
    shell:
        """
        mkdir -p {params.tmp_dir}

        (fastqc -t {threads} {input.r1} {input.r2} -o {params.tmp_dir}) 2> {log}
        
        cp {params.tmp_dir}/{wildcards.sample}_{wildcards.reads}_fastqc.html {output.html}
        cp {params.tmp_dir}/{wildcards.sample}_{wildcards.reads}_fastqc.zip {output.zip}

        rm -rf {params.tmp_dir}
        """


rule detect_adapter:
    input: 
        os.path.join(OUTDIR, "fastqc/{sample}_1_fastqc.zip")
    output: 
        os.path.join(OUTDIR, "adapters/{sample}_fastqc.txt")
    log: 
        os.path.join(LOGS, "detect_adapter/{sample}.log")
    params:
        detect_adapters = os.path.join(PY_SCRIPTS, "detect_adapters.py")    
    shell:
        """
        (python {params.detect_adapters} {input} > {output}) 2> {log}
        """


rule kneaddata:
    input:
        r1 = lambda wildcards: SAMPLES_LU[wildcards.sample],
        r2 = lambda wildcards: SAMPLES_LU[wildcards.sample].replace("_1.fastq.gz", "_2.fastq.gz"),
        adapter=os.path.join(OUTDIR, "adapters/{sample}_fastqc.txt")
    output:
        # dir = temp(directory(os.path.join(OUTDIR, "knead/{sample}/tmp"))),
        p1 = os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_1.fastq"),
        p2 = os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_2.fastq"),
    log:
        os.path.join(LOGS, "knead/{sample}.log")
    threads: 96
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

         cp {params.tmp_dir}/{wildcards.sample}_1_kneaddata_paired_1.fastq {output.p1}
         cp {params.tmp_dir}/{wildcards.sample}_1_kneaddata_paired_2.fastq {output.p2}
        
         rm -rf {params.tmp_dir}
         """

rule metaspades: 
    input:
        r1=os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_1.fastq"),
        r2=os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_2.fastq")
    output:
        contigs=os.path.join(OUTDIR, "spades/asm_{sample}/contigs.fasta"),
    benchmark:
        os.path.join(BENCHMARKS, "assembly", "{sample}.txt"),
    params:
        k="auto", #k-mer size
        m="94", #memory 100gb vamb
        tmp_dir = os.path.join(OUTDIR, "spades/asm_{sample}/tmp"),
    log:
        os.path.join(LOGS, "metaspades/{sample}.log")
    threads: 96 # 24 vamb
    conda:
        os.path.join(CONDA_ENVS, "metaspades.yaml")
    shell:
        """
        spades.py --meta -1 {input.r1} -2 {input.r2} -m {params.m} -o {params.tmp_dir} -k {params.k} -t {threads} --only-assembler
        
        cp {params.tmp_dir}/contigs.fasta {output.contigs}

        rm -rf {params.tmp_dir}
        """ 


rule concatenate:
    input:
        expand(os.path.join(OUTDIR, "spades/asm_{sample}/contigs.fasta"),sample=SAMPLES)
    output:
        os.path.join(OUTDIR, "global_contig_catalogue.fna.gz")
    conda:
        os.path.join(CONDA_ENVS, "concatenate.yaml")
    params:
        concatenate = os.path.join(PY_SCRIPTS, "concatenate.py")
    shell:
        """
        python {params.concatenate} {output} {input} -m 94
        """

rule alignment:
    input:
        r1=os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_1.fastq"),
        r2=os.path.join(OUTDIR, "knead/{sample}/{sample}_1_kneaddata_paired_2.fastq"),
        contig_catalouge=os.path.join(OUTDIR,"global_contig_catalogue.fna.gz")
    output:
        os.path.join(OUTDIR, "algn/{sample}_sorted.bam"),
    benchmark:
        os.path.join(BENCHMARKS, "alignment", "{sample}.txt")
    threads:
        64
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
        32
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
        expand(os.path.join(OUTDIR, "abdn_coverm/{sample}.tsv"),sample=SAMPLES)
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

