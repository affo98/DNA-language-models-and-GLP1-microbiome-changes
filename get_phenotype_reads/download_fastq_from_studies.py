import os
import argparse
import subprocess
import logging
import yaml

import pandas as pd
import numpy as np

import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm


def setup_data_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "LOG_PATH": os.path.join(os.getcwd(), "logs"),
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "CONFIG_PATH": os.path.join(os.getcwd(), "config"),
        "STUDIES_FASTQ_PATH": os.path.join(
            os.getcwd(), "get_phenotype_reads", "data", "studies_fastq_list"
        ),
        "SAMPLE_LABELS_RAW_PATH": os.path.join(
            os.getcwd(), "get_phenotype_reads", "data", "sample_labels_raw"
        ),
        "SAMPLE_LABELS_RAW_METAML_PATH": os.path.join(
            os.getcwd(), "get_phenotype_reads", "data", "sample_labels_raw", "metaml"
        ),
        "SAMPLE_LABELS_OUTPUT_PATH": os.path.join(
            os.getcwd(), "data", "phenotype", "sample_labels"
        ),
        "READS_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "phenotype", "reads"),
    }

    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        os.environ[var_name] = path

    return


def setup_logfile(path_to_logfile: str):

    log_file = os.path.join(path_to_logfile, "phenotype_studies.log")

    logging.basicConfig(
        filename=log_file,  # Log file name
        level=logging.INFO,  # Log level
        format="%(message)s",  # Only log the message (no timestamp, level, etc. in the format)
        filemode="w",  # 'w' to overwrite log file each time
    )
    logging.info(
        f"Run started at: {logging.Formatter('%(asctime)s').formatTime(logging.LogRecord('', '', '', '', '', '', ''))}"
    )


def preprocess_abundance_metaml(path_to_abundance: str) -> None:
    """Extracts sample aliases and labels from abundance.txt and saves them as PRJEB4336_sample_labels.txt and PRJEB1786_sample_labels.txt.
    Args:
        path_to_abundance (str): Path to the abundance.txt file.
    """
    a = pd.read_csv(f"{path_to_abundance}/abundance.txt", sep="\t", low_memory=False)

    a = a.T[[0, 1, 3]]
    a = a.reset_index()
    a.columns = a.iloc[0]
    a = a.drop(0)

    obe = a[a["dataset_name"].str.contains("obesity")]
    obe = obe[obe["disease"] != "n"]
    obe["disease"] = np.where(obe["disease"] == "obesity", "1", "0")
    obe = obe.rename(columns={"disease": "label", "sampleID": "sample alias"})
    obe = obe[["sample alias", "label"]]
    obe = obe.reset_index(drop=True)
    obe.to_csv(
        f"{path_to_abundance}/PRJEB4336_sample_labels.txt", sep="\t", index=False
    )

    t2dew = a[a["dataset_name"].str.contains("WT2D")]
    t2dew = t2dew[t2dew["disease"] != "impaired_glucose_tolerance"]
    t2dew["disease"] = np.where(t2dew["disease"] == "t2d", "1", "0")
    t2dew = t2dew.rename(columns={"disease": "label", "sampleID": "sample alias"})
    t2dew = t2dew[["sample alias", "label"]]
    t2dew = t2dew.reset_index(drop=True)
    t2dew["sample alias"] = t2dew["sample alias"].str.replace("S", "")
    t2dew.to_csv(
        f"{path_to_abundance}/PRJEB1786_sample_labels.txt", sep="\t", index=False
    )
    return


def read_studies(file_path: str) -> tuple[list[str], dict[str, str]]:
    """Reads a YAML file and returns a list of study_ids and a dict mapping study id to name."""

    study_id_to_name = {}
    file_path = os.path.join(os.environ["CONFIG_PATH"], "phenotype_studies.yml")
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        for study in data["studies"]:
            study_id_to_name[study["id"]] = study["name"]

    return list(study_id_to_name.keys()), study_id_to_name


def fetch_metadata_from_study(study_id) -> None:
    """Download list of fastq files from a study id from ENA"""

    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={study_id}&result=read_run&fields=study_accession,sample_accession,experiment_accession,run_accession,fastq_ftp"
    destination = os.path.join(
        os.environ["STUDIES_FASTQ_PATH"],
        f"{study_id_to_names[study_id]}_{study_id}_fastq_list.txt",
    )

    try:
        subprocess.run(["wget", url, "-O", destination], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the file: {e}")


def create_studies_dictionary(file_path: str, study_ids: list) -> dict:
    """
    Read list of fastq files from multiple studies and organize them in a nested dictionary.
    Structure: {study_accession: {run_accession: [fastq_files]}}
    """
    studies_data = {}

    for filename in os.listdir(file_path):
        if filename.endswith("_fastq_list.txt"):
            study_id = filename.split("_")[1]
            if study_id not in study_ids:
                continue
            study_file_path = os.path.join(file_path, filename)

            with open(study_file_path, "r") as file:
                next(file)

                for line in file:
                    run, sample, _, _, fastq_ftp = line.strip().split("\t")
                    fastq_files = fastq_ftp.split(";")

                    if study_id not in studies_data:
                        studies_data[study_id] = {}

                    if sample not in studies_data[study_id]:
                        studies_data[study_id][sample] = {}

                    if run not in studies_data[study_id][sample]:
                        studies_data[study_id][sample][run] = []

                    studies_data[study_id][sample][run] += fastq_files

    return studies_data


def map_sampleid_to_alias(all_studies_fastq: dict) -> dict:
    """Creates a mapping of sample_id to sample_alias and updates the sample_labels file with the new sample_id.
    Also removes samples that are not in the sample_labels file.

    Args:
        all_studies_fastq (dict): dictionary from create_studies_dictionary

    Returns:
        dict: processed dictionary all_studies_fastq, where some samples have been removed.
    """
    for study_id, samples in tqdm(all_studies_fastq.items(), desc="Mapping Sample IDs"):
        study_name = study_id_to_names[study_id]
        logging.info(
            f"#Samples {study_id} {study_name} \n    Before: {len(all_studies_fastq[study_id])}"
        )

        all_sample_labels_raw = [
            os.path.relpath(
                os.path.join(root, file), os.environ["SAMPLE_LABELS_RAW_PATH"]
            )
            for root, _, files in os.walk(os.environ["SAMPLE_LABELS_RAW_PATH"])
            for file in files
        ]
        sample_labels_raw_file = os.path.join(
            os.environ["SAMPLE_LABELS_RAW_PATH"],
            [file for file in all_sample_labels_raw if study_id in file][0],
        )
        with open(sample_labels_raw_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        sample_labels_output_file = os.path.join(
            os.environ["SAMPLE_LABELS_OUTPUT_PATH"],
            f"{study_name}_{study_id}_sample_labels.txt",
        )

        if study_id in ["PRJEB4336", "PRJEB1786"]:
            sample_alias_to_id = {}
            for sample_id, _ in samples.items():
                url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{sample_id}?includeLinks=false"
                response = requests.get(url)
                root = ET.fromstring(response.content)

                if study_id == "PRJEB4336":
                    sample_alias = root.find(".//SAMPLE").attrib.get("alias")
                    sample_alias = sample_alias.split("-")[1]
                    sample_alias_to_id[sample_alias] = sample_id

                elif study_id == "PRJEB1786":
                    sample_alias = root.find(".//TITLE").text
                    sample_alias_to_id[sample_alias] = sample_id

            sample_aliases_to_include = [line.split("\t")[0] for line in lines[1:]]
            sample_alias_to_id = {
                k: v
                for k, v in sample_alias_to_id.items()
                if k in sample_aliases_to_include
            }

            all_studies_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in all_studies_fastq[study_id].items()
                if sample_id in [ele[1] for ele in sample_alias_to_id.items()]
            }

        elif study_id in ["PRJEB12123", "PRJEB21528", "PRJNA422434"]:
            sample_ids, sample_labels = (
                [line.split("\t")[2].split("-")[1] for line in lines[1:]],
                [line.split("\t")[4] for line in lines[1:]],
            )
            if study_id == "PRJEB12123":
                sample_labels = [
                    1 if label == "Obesity" else 0 for label in sample_labels
                ]
            elif study_id == "PRJEB21528":
                sample_labels = [
                    1 if label == "Cardiovascular Diseases" else 0
                    for label in sample_labels
                ]
            elif study_id == "PRJNA422434":
                sample_labels = [
                    1 if label == "Diabetes Mellitus, Type 2" else 0
                    for label in sample_labels
                ]

            all_studies_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in all_studies_fastq[study_id].items()
                if sample_id in sample_ids
            }

        elif study_id in ["PRJDB3601", "PRJNA448494"]:
            run_id_to_sample_id = {}
            run_ids, sample_labels = (
                [line.split("\t")[1] for line in lines[1:]],
                [line.split("\t")[11] for line in lines[1:]],
            )

            if study_id == "PRJDB3601":  # obese and overweight for japanese!
                run_ids, sample_labels = zip(
                    *[
                        (run_id, 1 if label in ["D009765", "D050177"] else 0)
                        for run_id, label in zip(run_ids, sample_labels)
                        if label in ["D006262", "D009765", "D050177"]
                    ]
                )
                run_ids = list(run_ids)
                sample_labels = list(sample_labels)

            elif study_id == "PRJNA448494":
                sample_labels = [
                    1 if label == "D003924" else 0 for label in sample_labels
                ]

            study_metadata = os.path.join(
                os.environ["STUDIES_FASTQ_PATH"],
                [
                    f
                    for f in os.listdir(os.environ["STUDIES_FASTQ_PATH"])
                    if study_id in f
                ][0],
            )

            with open(study_metadata, "r", encoding="utf-8") as file:
                metadata_lines = file.readlines()

            for metadata_line in metadata_lines[1:]:
                run_id, sample_id = (
                    metadata_line.split("\t")[0],
                    metadata_line.split("\t")[1],
                )

                if run_id not in run_id_to_sample_id:
                    run_id_to_sample_id[run_id] = []
                run_id_to_sample_id[run_id].append(sample_id)

            sample_ids = [run_id_to_sample_id[run_id][0] for run_id in run_ids]

            all_studies_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in all_studies_fastq[study_id].items()
                if sample_id in sample_ids
            }

        with open(sample_labels_output_file, "w") as file:
            header = lines[0].strip().split("\t")
            updated_header = "\t".join(["sample_id", header[1]])
            file.write(updated_header + "\n")

            if study_id in ["PRJEB4336", "PRJEB1786"]:
                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    sample_alias = parts[0]
                    sample_label = parts[1]
                    sample_id = sample_alias_to_id.get(sample_alias, "Error")

                    updated_line = "\t".join([sample_id, sample_label])
                    file.write(updated_line + "\n")

            elif study_id in [
                "PRJEB12123",
                "PRJEB21528",
                "PRJNA422434",
            ] or study_id in ["PRJDB3601", "PRJNA448494"]:
                for sample_id, sample_label in zip(sample_ids, sample_labels):
                    updated_line = "\t".join([sample_id, str(sample_label)])
                    file.write(updated_line + "\n")

        logging.info(f"    After:  {len(all_studies_fastq[study_id])} \n")
    return all_studies_fastq


def download_fastq(url, destination):
    """Download a fastq file using wget."""
    try:
        subprocess.run(["wget", url, "-O", destination], check=True)
        print(f"Downloaded Fastq: {destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")


def download_all_fastq_files(all_studies_fastq: dict) -> None:
    """
    Download all FASTQ files from all studies.

    Expects a nested dictionary structure:
      all_studies_fastq[study_id][sample_id][run_accession] = fastq_file

    Requires that `study_id_to_names` is defined (mapping study_id to study_name).
    """
    for study_id in all_studies_fastq.keys():
        study_name = study_id_to_names[study_id]
        study_dir = os.path.join(
            os.environ["READS_OUTPUT_PATH"], f"{study_name}_{study_id}"
        )
        os.makedirs(study_dir, exist_ok=True)

        for sample_id in all_studies_fastq[study_id]:
            sample_dir = os.path.join(study_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            for run_accession in all_studies_fastq[study_id][sample_id]:
                run_dir = os.path.join(sample_dir, run_accession)
                os.makedirs(run_dir, exist_ok=True)

                for fastq_file in all_studies_fastq[study_id][sample_id][run_accession]:
                    fastq_filename = os.path.basename(fastq_file)
                    destination_path = os.path.join(run_dir, fastq_filename)
                    download_fastq(fastq_file, destination_path)
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List files in multiple study directories."
    )
    parser.add_argument(
        "-s", "--studies", nargs="*", help="Study IDs to filter (optional)"
    )
    args = parser.parse_args()

    setup_data_paths()
    setup_logfile(os.environ["LOG_PATH"])

    preprocess_abundance_metaml(os.environ["SAMPLE_LABELS_RAW_METAML_PATH"])

    all_study_ids, study_id_to_names = read_studies(os.environ["CONFIG_PATH"])
    study_ids = args.studies if args.studies else all_study_ids

    [fetch_metadata_from_study(study_id) for study_id in study_ids]

    all_studies_fastq = create_studies_dictionary(
        os.environ["STUDIES_FASTQ_PATH"], study_ids
    )
    all_studies_fastq = map_sampleid_to_alias(all_studies_fastq)

    download_all_fastq_files(all_studies_fastq)
