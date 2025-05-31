import os
import argparse
import subprocess

import pandas as pd
import numpy as np

import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm

BASEDIR = os.path.join(os.getcwd())


def setup_data_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "STUDIES_FASTQ_PATH": os.path.join(BASEDIR, "metadata", "studies_fastq_list"),
        "SAMPLE_LABELS_RAW_PATH": os.path.join(
            BASEDIR, "src", "metadata", "sample_labels_raw"
        ),
        "SAMPLE_LABELS_RAW_METAML_PATH": os.path.join(
            BASEDIR,
            "src",
            "metadata",
            "sample_labels_raw",
            "metaml",
        ),
        "SAMPLE_LABELS_OUTPUT_PATH": os.path.join(BASEDIR, "raw_data", "sample_labels"),
        "READS_OUTPUT_PATH": os.path.join(BASEDIR, "raw_data", "reads"),
    }

    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        os.environ[var_name] = path

    return


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


def fetch_metadata_from_study(study_id: str, study_name: str) -> None:
    """Download list of fastq files from a study id from ENA"""

    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={study_id}&result=read_run&fields=study_accession,sample_accession,experiment_accession,run_accession,fastq_ftp"
    destination = os.path.join(
        os.environ["STUDIES_FASTQ_PATH"],
        f"{study_name}_{study_id}_fastq_list.txt",
    )

    try:
        subprocess.run(["wget", url, "-O", destination], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the file: {e}")


def create_studies_dictionary(study_id: str) -> dict:
    """
    Read list of fastq files from multiple studies and organize them in a nested dictionary.
    Structure: {study_accession: {run_accession: [fastq_files]}}
    """
    studies_data = {}

    for filename in os.listdir(os.environ["STUDIES_FASTQ_PATH"]):
        if study_id in filename:
            study_file_path = os.path.join(os.environ["STUDIES_FASTQ_PATH"], filename)

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


def map_sampleid_to_alias(samples_fastq: dict, study_name: str) -> dict:
    """Creates a mapping of sample_id to sample_alias and updates the sample_labels file with the new sample_id.
    Also removes samples that are not in the sample_labels file.

    Args:
        all_studies_fastq (dict): dictionary from create_studies_dictionary

    Returns:
        dict: processed dictionary all_studies_fastq, where some samples have been removed.
    """
    for study_id, samples in tqdm(samples_fastq.items(), desc="Mapping Sample IDs"):
        samples_before = len(samples_fastq[study_id])

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

            samples_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in samples_fastq[study_id].items()
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

            samples_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in samples_fastq[study_id].items()
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

            samples_fastq[study_id] = {
                sample_id: runs
                for sample_id, runs in samples_fastq[study_id].items()
                if sample_id in sample_ids
            }
        if not os.path.exists(sample_labels_output_file):
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

                samples_after = len(samples_fastq[study_id])
                # with open(args.log, "a") as log:
                #     log.write(
                #         f"#Samples {study_id} {study_name} \n    Before: {samples_before}"
                #     )
                #     log.write(f"    After:  {samples_after} \n")
        else:
            print(
                f"{sample_labels_output_file} already exists!\nContinueing workflow to downloads"
            )
            continue
    return samples_fastq


def download_fastq(url: str, destination: str) -> None:
    """Download a fastq file using wget."""
    try:
        subprocess.run(["wget", url, "-O", destination], check=True)
        print(f"Downloaded Fastq: {destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")


def download_all_fastq_files(samples_fastq: dict, study_name: str) -> None:
    """
    Download all FASTQ files from all studies.

    Expects a nested dictionary structure:
      samples_fastq[study_id][sample_id][run_accession] = fastq_file

    Requires that `study_id_to_names` is defined (mapping study_id to study_name).
    """
    for study_id in tqdm(samples_fastq.keys(), desc="Downloading Fastq Files"):
        study_dir = os.path.join(
            os.environ["READS_OUTPUT_PATH"], f"{study_name}_{study_id}"
        )

        os.makedirs(study_dir, exist_ok=True)

        for sample_id in samples_fastq[study_id]:
            sample_dir = os.path.join(study_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            for run_accession in samples_fastq[study_id][sample_id]:
                run_dir = os.path.join(sample_dir, run_accession)

                for fastq_file in samples_fastq[study_id][sample_id][run_accession]:
                    fastq_filename = os.path.basename(fastq_file)

                    destination_path = os.path.join(
                        sample_dir, fastq_filename.split("_")[-1]
                    )
                    print("#################\n")
                    print("DESTINATION", destination_path, flush=True)
                    print("CHECK", os.path.exists(destination_path), flush=True)
                    print("#################\n")
                    if not os.path.exists(destination_path):
                        download_fastq(fastq_file, destination_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List files in multiple study directories."
    )
    parser.add_argument("-i", "--studyid", help="Study ID to filter")
    parser.add_argument("-n", "--studyname", help="Study ID to filter")
    # parser.add_argument("--log", help="Path to log file", required=True)
    args = parser.parse_args()

    study_id = args.studyid
    study_name = args.studyname

    setup_data_paths()

    preprocess_abundance_metaml(os.environ["SAMPLE_LABELS_RAW_METAML_PATH"])

    fetch_metadata_from_study(study_id, study_name)

    samples_fastq = create_studies_dictionary(study_id)
    samples_fastq = map_sampleid_to_alias(samples_fastq, study_name)

    download_all_fastq_files(samples_fastq, study_name)
