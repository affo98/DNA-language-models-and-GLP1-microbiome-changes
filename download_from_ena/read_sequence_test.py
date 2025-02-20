import os
import gzip
import subprocess
from Bio import SeqIO


def setup_data_paths():
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    # Define the paths
    paths = {"DATA_PATH": os.path.join(os.getcwd(), "data")}

    # Check and create directories if they don't exist
    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # Set the environment variable
        os.environ[var_name] = path
        print(f"Environment variable {var_name} set to: {os.environ[var_name]}")


def find_fastq_files(directory):
    fastq_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fastq.gz"):
                fastq_files.append(os.path.join(root, file))

    return fastq_files


def get_sequences_count(file_path):
    total_records = 0

    # First pass to count total records
    with gzip.open(file_path, "rt") as handle:
        for _ in SeqIO.parse(handle, "fastq"):
            total_records += 1

    return total_records


if __name__ == "__main__":
    setup_data_paths()

    fastq_files = find_fastq_files(os.environ["DATA_PATH"])
    print(fastq_files)

    for fastq_file in fastq_files:
        total_records = get_sequences_count(fastq_file)
        print(total_records)
