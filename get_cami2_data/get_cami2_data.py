import os
from argparse import ArgumentParser
import logging
import gzip
import shutil
import tarfile

import requests


OUTDIR_TMP = os.path.join(os.getcwd(), "get_cami2_data", "data")
OUTDIR = os.path.join(os.getcwd(), "data", "cami2")
CONFIG = os.path.join(os.getcwd(), "config")
LOG_PATH = (os.path.join(os.getcwd(), "logs"),)


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


def download_plant_marine(name, reads, samples, OUTDIR_DATASET, OUTDIR_TMP_DATASET):

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

        print("get reponse")
        # response_read = requests.get(sample_read_url)
        response_contig = requests.get(sample_contig_url)

        # if response_read.status_code == 200:
        #     print("si")
        #     read_file = os.path.join(OUTDIR_TMP_DATASET, f"{sample}_reads.tar.gz")
        #     with open(read_file, "wb") as handle:
        #         handle.write(response_read.content)
        #     with tarfile.open(read_file, "r:gz") as tar:
        #         tar.extractall(path=OUTDIR_TMP_DATASET)

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
                OUTDIR_TMP_DATASET, f"{sample}_anonymous_gsa.fasta"
            )
            with gzip.open(contig_file_input, "rb") as f_in, open(
                contig_file_output, "wb"
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
            shutil.rmtree(os.path.join(OUTDIR_TMP_DATASET, "simulation_short_read"))
            os.remove(contig_tar)


def download_human(name, reads, samples, OUTDIR_DATASET, OUTDIR_TMP_DATASET): ...


def main(output_contigs, output_reads, dataset, samples):
    logging.info(f"---------- {dataset} ----------")

    OUTDIR_DATASET = os.path.join(OUTDIR, dataset)
    OUTDIR_TMP_DATASET = os.path.join(OUTDIR_TMP, dataset)
    os.makedirs(OUTDIR_TMP_DATASET)
    name, reads = dataset.split("_")

    if name in HUMAN_DATASETS:
        download_human(name, reads, samples, OUTDIR_DATASET, OUTDIR_TMP_DATASET)

    else:
        download_plant_marine(name, reads, samples, OUTDIR_DATASET, OUTDIR_TMP_DATASET)


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("output_contigs", help="Path to save the contigs file")
    parser.add_argument("output_reads", help="Path to save the reads file")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("samples", nargs="+", help="Samples")
    args = parser.parse_args()
    output_contigs = args.output_contigs
    output_reads = args.output_reads
    dataset = args.dataset
    samples = args.samples

    return output_contigs, output_reads, dataset, samples


if __name__ == "__main__":

    output_contigs, output_reads, dataset, samples = add_arguments()

    print(dataset)
    print(samples)

    main(output_contigs, output_reads, dataset, samples)
