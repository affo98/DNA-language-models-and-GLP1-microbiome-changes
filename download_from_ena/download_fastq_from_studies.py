import os
import gzip
import subprocess


def setup_data_paths():
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    # Define the paths
    paths = {
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "STUDIES_FASTQ_PATH": os.path.join(os.getcwd(), "data", "studies_fastq_list"),
    }

    # Check and create directories if they don't exist
    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # Set the environment variable
        os.environ[var_name] = path
        print(f"Environment variable {var_name} set to: {os.environ[var_name]}")


def read_txt_to_list(file_path):
    """Reads a text file with newline-separated values into a list"""
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def fetch_fastq_from_study(study_accession):
    """Download list of fastq files from a study accession from ENA"""

    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={study_accession}&result=read_run&fields=fastq_ftp"
    destination = os.path.join(
        os.environ["DATA_PATH"],
        os.environ["STUDIES_FASTQ_PATH"],
        f"{study_accession}_fastq_list.txt",
    )

    try:
        subprocess.run(["wget", url, "-O", destination], check=True)
        print(f"Downloaded Study: {destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the file: {e}")


def read_fastq_from_all_studies(file_path):
    """
    Read list of fastq files from multiple studies and organize them in a nested dictionary.
    Structure: {study_accession: {run_accession: [fastq_files]}}
    """
    studies_data = {}

    for filename in os.listdir(file_path):
        if filename.endswith("_fastq_list.txt"):
            study_accession = filename.split("_")[0]
            study_file_path = os.path.join(file_path, filename)

            with open(study_file_path, "r") as file:
                next(file)

                for line in file:
                    run_accession, fastq_ftp = line.strip().split("\t")
                    fastq_files = fastq_ftp.split(";")

                    if study_accession not in studies_data:
                        studies_data[study_accession] = {}

                    studies_data[study_accession][run_accession] = fastq_files

    return studies_data


def download_fastq(url, destination):
    """Download a fastq file using wget."""
    try:
        subprocess.run(["wget", "-q", url, "-O", destination], check=True)
        print(f"Downloaded Fastq: {destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")


def download_all_fastq_files(all_studies_fastq):
    """Download all FASTQ files"""

    for study_accession, runs in all_studies_fastq.items():
        study_dir = os.path.join(os.environ["DATA_PATH"], study_accession)
        os.makedirs(study_dir, exist_ok=True)

        for run_accession, fastq_files in runs.items():
            run_dir = os.path.join(study_dir, run_accession)
            os.makedirs(run_dir, exist_ok=True)

            for fastq_file in fastq_files:
                fastq_filename = os.path.basename(fastq_file)
                destination_path = os.path.join(run_dir, fastq_filename)

                # Download the FASTQ file
                download_fastq(fastq_file, destination_path)


if __name__ == "__main__":
    setup_data_paths()

    studies_path = f"studies_to_download.txt"
    studies = read_txt_to_list(file_path=studies_path)
    print(f"Studies included: {studies}")

    [fetch_fastq_from_study(study) for study in studies]

    all_studies_fastq = read_fastq_from_all_studies(os.environ["STUDIES_FASTQ_PATH"])
    print(all_studies_fastq.keys())

    download_all_fastq_files(all_studies_fastq)
