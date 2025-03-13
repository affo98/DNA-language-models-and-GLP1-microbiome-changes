import os
from argparse import ArgumentParser
import logging
import gzip
import shutil
import tarfile
from Bio import SeqIO

import subprocess
import requests


OUTDIR_TMP = os.path.join(os.getcwd(), "get_cami2_data", "data")
OUTDIR = os.path.join(os.getcwd(), "data", "cami2")
CONFIG = os.path.join(os.getcwd(), "config")
LOG_PATH = os.path.join(os.getcwd(), "logs")


HUMAN_DATASETS = ["airways", "gastro", "oral", "skin", "urogenital"]


def setup_logfile(path_to_logfile: str):

    log_file = os.path.join(path_to_logfile, "cami2_processing.log")

    logging.basicConfig(
        filename=log_file,  # Log file name
        level=logging.INFO,  # Log level
        format="%(message)s",  # Only log the message (no timestamp, level, etc. in the format)
        filemode="w",  # 'w' to overwrite log file each time
    )
    logging.info(
        f"Run started at: {logging.Formatter('%(asctime)s').formatTime(logging.LogRecord('', '', '', '', '', '', ''))}"
    )
    return


def find_file_in_subdirectories(root_dir, filename):
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def filter_write_and_log_contigs(input_file, output_file, min_length=2000):
    original_count = 0
    removed_count = 0

    with gzip.open(input_file, "rt") as f_in, open(output_file, "w") as f_out:
        for record in SeqIO.parse(f_in, "fasta"):
            original_count += 1
            if len(record) >= min_length:
                SeqIO.write(record, f_out, "fasta")
            else:
                removed_count += 1

    logging.info(f"Number of contigs originally: {original_count}")
    logging.info(f"Removed {removed_count} contigs below 2500 bps")
    logging.info(f"Number of contigs: {original_count-removed_count}")

    return


def download_plant_marine(name, reads, samples, OUTDIR_TMP_DATASET):

    if name == "marine":
        base_url = f"https://frl.publisso.de/data/frl:6425521/{name}/{reads}_read/"
    elif name == "plant":
        base_url = (
            f"https://frl.publisso.de/data/frl:6425521/{name}_associated/{reads}_read/"
        )

    name_to_prefix = {"marine": "marmg", "plant": "rhimg"}

    url = f"{base_url}{name_to_prefix[name]}CAMI2_"

    for sample in samples:
        sample_read_url = f"{url}sample_{sample}_reads.tar.gz"
        sample_contig_url = f"{url}sample_{sample}_contigs.tar.gz"

        # reads
        try:
            reads_tar = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_reads.tar.gz")
            subprocess.run(
                [
                    "wget",
                    sample_read_url,
                    "-O",
                    reads_tar,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
        with tarfile.open(reads_tar, "r:gz") as tar:
            tar.extractall(path=OUTDIR_TMP_DATASET)
        read_file = find_file_in_subdirectories(
            OUTDIR_TMP_DATASET, f"anonymous_reads.fq.gz"
        )
        read_file_output = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_reads.fq.gz")
        shutil.copy(read_file, read_file_output)
        shutil.rmtree(os.path.join(OUTDIR_TMP_DATASET, "simulation_short_read"))
        os.remove(reads_tar)

        # contigs
        response_contig = requests.get(sample_contig_url)
        if response_contig.status_code == 200:
            contig_tar = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_contigs.tar.gz")
            with open(contig_tar, "wb") as handle:
                handle.write(response_contig.content)
            with tarfile.open(contig_tar, "r:gz") as tar:
                tar.extractall(path=OUTDIR_TMP_DATASET)
            contig_file_input = find_file_in_subdirectories(
                OUTDIR_TMP_DATASET, "anonymous_gsa.fasta.gz"
            )
            contig_file_output = os.path.join(
                OUTDIR_TMP_DATASET, f"{sample}_contigs.fasta"
            )
            filter_write_and_log_contigs(
                contig_file_input, contig_file_output, min_length=2000
            )
            shutil.rmtree(os.path.join(OUTDIR_TMP_DATASET, "simulation_short_read"))
            os.remove(contig_tar)


def download_human(name, samples, OUTDIR_TMP_DATASET):

    if name in ["airways", "skin", "urogenital"]:
        base_url = f"https://frl.publisso.de/data/frl:6425518/airskinurogenital/"

    elif name in ["oral", "gastro"]:
        base_url = f"https://frl.publisso.de/data/frl:6425518/gastrooral/"

    for sample in samples:

        try:
            sample_tar_url = f"{base_url}sample_{sample}.tar.gz"
            sample_tar = os.path.join(OUTDIR_TMP_DATASET, f"{sample}.tar.gz")
            subprocess.run(
                [
                    "wget",
                    sample_tar_url,
                    "-O",
                    sample_tar,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
        with tarfile.open(sample_tar, "r:gz") as tar:
            tar.extractall(path=OUTDIR_TMP_DATASET)

        # reads
        read_file = find_file_in_subdirectories(
            OUTDIR_TMP_DATASET, f"anonymous_reads.fq.gz"
        )
        read_file_output = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_reads.fq.gz")
        shutil.copy(read_file, read_file_output)

        # contigs
        contig_file_input = find_file_in_subdirectories(
            OUTDIR_TMP_DATASET, "anonymous_gsa.fasta.gz"
        )
        contig_file_output = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_contigs.fasta")
        filter_write_and_log_contigs(
            contig_file_input, contig_file_output, min_length=2000
        )

        tar_dir = [d.name for d in OUTDIR_TMP_DATASET.iterdir() if d.is_dir()][0]
        shutil.rmtree(os.path.join(OUTDIR_TMP_DATASET, tar_dir))
        os.remove(sample_tar)


def main(dataset, samples):

    setup_logfile(LOG_PATH)
    logging.info(f"---------- {dataset} ----------")

    OUTDIR_TMP_DATASET = os.path.join(OUTDIR_TMP, dataset)
    if os.path.exists(OUTDIR_TMP_DATASET):
        shutil.rmtree(OUTDIR_TMP_DATASET)
    os.makedirs(OUTDIR_TMP_DATASET)
    name, reads = dataset.split("_")

    if name in HUMAN_DATASETS:
        download_human(name, samples, OUTDIR_TMP_DATASET)

    else:
        download_plant_marine(name, reads, samples, OUTDIR_TMP_DATASET)


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("samples", nargs="+", help="Samples")
    args = parser.parse_args()
    dataset = args.dataset
    samples = args.samples

    return dataset, samples


if __name__ == "__main__":

    dataset, samples = add_arguments()

    print(dataset)
    print(samples)

    main(dataset, samples)
