import os
import logging
import argparse
import yaml
import gzip
import tarfile

import requests

import pandas as pd

from Bio import SeqIO

from tqdm import tqdm


CONTIG_FILE = "anonymous_gsa_pooled.fasta.gz"
MAPPING_FILE = "gsa_pooled_mapping.tsv.gz"
TAXONOMIC_FILE = "taxonomic_profile_0.txt"


def setup_data_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "LOG_PATH": os.path.join(os.getcwd(), "logs"),
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "CONFIG_PATH": os.path.join(os.getcwd(), "config"),
        "CAMI2_DATA_PATH": os.path.join(os.getcwd(), "get_cami2_data", "data"),
        "CAMI2_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2"),
    }

    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        os.environ[var_name] = path

    return


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


def read_dataset_names(file_path: str) -> list[list[str]]:
    """Reads a YAML file and returns a list of dataset names and short/long reads."""
    file_path = os.path.join(os.environ["CONFIG_PATH"], "cami2_processing.yml")
    datasets = []
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        for dataset in data["datasets"]:
            datasets.append(dataset["name"])

    return datasets


def download_cami_contigs(dataset_name: str) -> None:

    logging.info(f"---------- {dataset_name} ----------")
    base_url = "https://frl.publisso.de/data/frl:6425521"
    dataset_to_tarfile = {
        "marine": "marmgCAMI2_setup.tar.gz",
        "plant": "rhimgCAMI2_setup.tar.gz",
    }
    dataset, reads = dataset_name.split("_")

    if dataset == "marine":
        url = f"{base_url}/{dataset}/{reads}_read/{dataset_to_tarfile[dataset]}"
    elif dataset == "plant":
        if reads == "short":
            url = f"{base_url}/{dataset}_associated/{reads}_read/{dataset_to_tarfile[dataset]}"
        elif reads == "long":
            url = f"{base_url}/{dataset}_associated/{reads}_read_pacbio/{dataset_to_tarfile[dataset]}"

    response = requests.get(url)

    os.environ["raw_data_path"] = os.path.join(
        os.environ["CAMI2_DATA_PATH"], f"{dataset}_{reads}"
    )

    if response.status_code == 200:
        with open(dataset_to_tarfile[dataset], "wb") as f:
            f.write(response.content)

        with tarfile.open(dataset_to_tarfile[dataset], "r:gz") as tar:
            tar.extractall(path=os.environ["raw_data_path"])
        print(f"{dataset}_{reads} Extracted successfully!")
        os.remove(dataset_to_tarfile[dataset])

    else:
        print(
            f"Failed to download {dataset}_{reads}. Status code:", response.status_code
        )


def load_raw_cami_files(
    contig_file: str, mapping_file: str, taxanomic_file: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    raw_data_path_unpacked = os.path.join(
        os.environ["raw_data_path"], os.listdir(os.environ["raw_data_path"])[0]
    )

    contig_file = os.path.join(raw_data_path_unpacked, contig_file)
    mapping_file = os.path.join(raw_data_path_unpacked, mapping_file)
    taxanomic_file = os.path.join(raw_data_path_unpacked, taxanomic_file)

    with gzip.open(contig_file, "rt") as handle:
        records = [
            {"contig_id": record.id, "seq": str(record.seq)}
            for record in SeqIO.parse(handle, "fasta")
        ]
    con = pd.DataFrame(records)
    n_contigs_original = con.shape[0]
    logging.info(f"Number of contigs originally: {n_contigs_original}")
    con = con[con["seq"].str.len() >= 2500]
    logging.info(f"Number of contigs above 2500 bps: {con.shape[0]}")
    logging.info(f"Removed {n_contigs_original - con.shape[0]} contigs below 2500 bps")

    map = pd.read_csv(mapping_file, compression="gzip", sep="\t")

    tax = pd.read_csv(taxanomic_file, sep="\t", skiprows=4)

    return con, map, tax


def preprocess_cami_files(
    con: pd.DataFrame, map: pd.DataFrame, tax: pd.DataFrame
) -> pd.DataFrame:

    out = pd.merge(
        con,
        map[["#anonymous_contig_id", "genome_id"]],
        how="left",
        left_on="contig_id",
        right_on="#anonymous_contig_id",
    )
    out = pd.merge(
        out,
        tax[["_CAMI_genomeID", "@@TAXID", "TAXPATH", "TAXPATHSN", "RANK"]],
        how="left",
        left_on="genome_id",
        right_on="_CAMI_genomeID",
    )
    out = out.drop(columns=["#anonymous_contig_id", "_CAMI_genomeID"])
    out = out.rename(
        columns={"RANK": "rank", "@@TAXID": "tax_id", "genome_id": "cami_genome_id"}
    )

    out["tax_id"] = out["tax_id"].astype(str)
    out["tax_id"] = out["tax_id"].str.split(".").str[0]

    taxpath_columns = out["TAXPATHSN"].str.split("|", expand=True)
    taxpath_columns.columns = [
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "strain",
    ]
    out = pd.concat([out, taxpath_columns], axis=1)
    out = out.drop(columns=["TAXPATHSN"])

    out = out[
        [
            "contig_id",
            "cami_genome_id",
            "tax_id",
            "rank",
            "superkingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
            "strain",
            "seq",
        ]
    ]

    n_genomes_original = out["cami_genome_id"].nunique()
    logging.info(f"Number of genomes originally: {n_genomes_original}")

    genomes_fewer_than_10 = (
        out["cami_genome_id"]
        .value_counts()[out["cami_genome_id"].value_counts() < 10]
        .index
    )
    out = out[~(out["cami_genome_id"].isin(genomes_fewer_than_10))]
    logging.info(
        f"Number of genomes w. more than 10 contigs {out['cami_genome_id'].nunique()}"
    )
    logging.info(
        f"Removed {genomes_fewer_than_10.shape[0]} genomes with fewer than 10 contigs"
    )

    return out


def get_summary_stats(output: pd.DataFrame) -> None:

    for col in [
        "cami_genome_id",
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "strain",
    ]:
        logging.info(f"{col}: {output[col].nunique()}")

    min_len = output["seq"].str.len().min()
    max_len = output["seq"].str.len().max()
    avg_len = output["seq"].str.len().mean()
    logging.info(
        f"Sequence length of all genomes -> Min: {min_len}, Max: {max_len}, Avg: {avg_len:.2f}"
    )

    genome_counts = output["cami_genome_id"].value_counts()
    min_genome_id = genome_counts.idxmin()
    min_genome_count = genome_counts.min()
    max_genome_id = genome_counts.idxmax()
    max_genome_count = genome_counts.max()
    avg_genome_count = genome_counts.mean()
    logging.info(f"Genome with Min Rows: {min_genome_id} ({min_genome_count} rows)")
    logging.info(f"Genome with Max Rows: {max_genome_id} ({max_genome_count} rows)")
    logging.info(f"Average Rows per Genome: {avg_genome_count:.2f}\n\n")

    return


def save_output(output: pd.DataFrame, dataset: str) -> None:
    output.to_csv(
        os.path.join(os.environ["CAMI2_OUTPUT_PATH"], f"{dataset}_contigs.csv"),
        index=False,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Cami2 datasets to include.")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        help="Choose Cami2 datasets (optional). E.g. marine_long, plant_short.",
    )
    args = parser.parse_args()

    setup_data_paths()
    setup_logfile(os.environ["LOG_PATH"])

    all_datasets = read_dataset_names(os.environ["CONFIG_PATH"])
    all_datasets = args.datasets if args.datasets else all_datasets

    for dataset in tqdm(all_datasets, desc="Processing Cami2-datasets"):
        download_cami_contigs(dataset)

        con, map, tax = load_raw_cami_files(CONTIG_FILE, MAPPING_FILE, TAXONOMIC_FILE)

        output = preprocess_cami_files(con, map, tax)

        get_summary_stats(output)
        save_output(output, dataset)
