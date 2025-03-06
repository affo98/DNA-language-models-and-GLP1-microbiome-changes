import os
import logging
from argparse import ArgumentParser
import yaml
import gzip
import tarfile
import shutil

import requests

import pandas as pd

from Bio import SeqIO

from tqdm import tqdm


def setup_data_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "LOG_PATH": os.path.join(os.getcwd(), "logs"),
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "CONFIG_PATH": os.path.join(os.getcwd(), "config"),
        "CAMI2_DATA_PATH": os.path.join(os.getcwd(), "get_cami2_data", "data"),
        "CAMI2_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2"),
        "CAMI2_GFM_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2", "gfm"),
        "CAMI2_VAMB_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2", "vamb"),
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


def download_cami_contigs(dataset: str, reads: str) -> None:

    logging.info(f"---------- {dataset}_{reads} ----------")

    human_datasets = ["airways", "gastro", "oral", "skin", "urogenital"]

    if dataset in ["marine", "plant"]:
        base_url = "https://frl.publisso.de/data/frl:6425521"
        dataset_to_tarfile = {
            "marine": "marmgCAMI2_setup.tar.gz",
            "plant": "rhimgCAMI2_setup.tar.gz",
        }

        if dataset == "marine":
            url = f"{base_url}/{dataset}/{reads}_read/{dataset_to_tarfile[dataset]}"
        elif dataset == "plant":
            if reads == "short":
                url = f"{base_url}/{dataset}_associated/{reads}_read/{dataset_to_tarfile[dataset]}"
            elif reads == "long":
                url = f"{base_url}/{dataset}_associated/{reads}_read_pacbio/{dataset_to_tarfile[dataset]}"

        response = requests.get(url)
        if response.status_code == 200:
            with open(dataset_to_tarfile[dataset], "wb") as f:
                f.write(response.content)

            with tarfile.open(dataset_to_tarfile[dataset], "r:gz") as tar:
                tar.extractall(path=os.environ["raw_data_path"])
            os.remove(dataset_to_tarfile[dataset])

        else:
            print(
                f"Failed to download {dataset}_{reads}. Status code:",
                response.status_code,
            )

    elif dataset in human_datasets:
        os.makedirs(os.environ["raw_data_path"])
        base_url = "https://openstack.cebitec.uni-bielefeld.de:8080/swift/v1/CAMI_"
        dataset_to_urlsuffix_taxid = {
            "airways": ["Airways", "10"],
            "gastro": ["Gastrointestinal_tract", "0"],
            "oral": ["Oral", "6"],
            "skin": ["Skin", "1"],
            "urogenital": ["Urogenital_tract", "0"],
        }

        url = f"{base_url}{dataset_to_urlsuffix_taxid[dataset][0]}/short_read/"
        human_files = [
            "gsa.fasta.gz",
            "gsa_pooled_mapping.tsv.gz",
            "taxonomic_profile_",
        ]
        for file in human_files:
            if file == "taxonomic_profile_":
                file = f"{file}{dataset_to_urlsuffix_taxid[dataset][1]}.txt"
            url_file = f"{url}{file}"

            print(url_file)

            response = requests.get(url_file)

            if response.status_code == 200:
                with open(os.path.join(os.environ["raw_data_path"], file), "wb") as f:
                    f.write(response.content)

            else:
                print(
                    f"Failed to download {dataset}_{reads}. Status code:",
                    response.status_code,
                )

        # contig abundances per sample
        url_file_list = f"{base_url}{dataset_to_urlsuffix_taxid[dataset][0]}/"
        response_file_list = requests.get(url_file_list)
        file_list = response_file_list.text.strip().split("\n")
        sample_gsa_mapping = [
            file for file in file_list if "short_read" in file and "gsa_mapping" in file
        ]
        os.makedirs(os.path.join(os.environ["raw_data_path"], "abundance"))
        for sample in sample_gsa_mapping:
            sample_name = sample.split("sample_")[1].split("/")[0]
            sample_response = requests.get(
                f"{base_url}{dataset_to_urlsuffix_taxid[dataset][0]}/{sample}"
            )
            if sample_response.status_code == 200:
                with open(
                    os.path.join(
                        os.environ["raw_data_path"],
                        "abundance",
                        f"{sample_name}_{sample.split('/')[-1]}",
                    ),
                    "wb",
                ) as f:
                    f.write(sample_response.content)
            else:
                print(
                    f"Failed to download {base_url}{dataset_to_urlsuffix_taxid[dataset][0]}/{sample} Status code:",
                    sample_response.status_code,
                )


def load_raw_cami_files(
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if dataset in ["marine", "plant"]:
        raw_data_path = os.path.join(
            os.environ["raw_data_path"], os.listdir(os.environ["raw_data_path"])[0]
        )
        contig_file = "anonymous_gsa_pooled.fasta.gz"
        mapping_file = "gsa_pooled_mapping.tsv.gz"
        taxonomic_file = "taxonomic_profile_0.txt"

    elif dataset in ["airways", "gastro", "oral", "skin", "urogenital"]:
        raw_data_path = os.environ["raw_data_path"]

        dataset_to_urlsuffix_taxid = {
            "airways": "10",
            "gastro": "0",
            "oral": "6",
            "skin": "1",
            "urogenital": "0",
        }
        contig_file = "gsa.fasta.gz"
        mapping_file = "gsa_pooled_mapping.tsv.gz"
        taxonomic_file = f"taxonomic_profile_{dataset_to_urlsuffix_taxid[dataset]}.txt"

    contig_file_path = os.path.join(raw_data_path, contig_file)
    mapping_file_path = os.path.join(raw_data_path, mapping_file)
    taxanomic_file_path = os.path.join(raw_data_path, taxonomic_file)

    with gzip.open(contig_file_path, "rt") as handle:
        records = [
            {"contig_id": record.id, "seq": str(record.seq)}
            for record in SeqIO.parse(handle, "fasta")
        ]
    con = pd.DataFrame(records)

    map = pd.read_csv(mapping_file_path, compression="gzip", sep="\t")

    if dataset in ["marine", "plant"]:
        tax = pd.read_csv(taxanomic_file_path, sep="\t", skiprows=4)

    elif dataset in ["airways", "gastro", "oral", "skin", "urogenital"]:
        with open(taxanomic_file_path, "r") as f:
            lines = f.readlines()
            del lines[4]
        with open(taxanomic_file_path, "w") as f:
            f.writelines(lines)

        tax = pd.read_csv(taxanomic_file_path, sep="\t", skiprows=3)

    return con, map, tax


def preprocess_cami_files(
    dataset: str, con: pd.DataFrame, map: pd.DataFrame, tax: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logging.info(f"\n** Preprocessing/Cleaning **")
    n_contigs_original = con.shape[0]
    logging.info(f"Number of contigs originally: {n_contigs_original}")
    con = con[con["seq"].str.len() >= 2500]
    logging.info(f"Number of contigs above 2500 bps: {con.shape[0]}")
    logging.info(f"Removed {n_contigs_original - con.shape[0]} contigs below 2500 bps")

    # invalied_contigs_byseq = con.loc[
    #     ~con["seq"].apply(lambda seq: any(c in {"A", "G", "T", "C"} for c in seq)),
    #     "contig_id",
    # ]
    # invalid_contigs_byid = map.groupby("#anonymous_contig_id").filter(
    #     lambda x: x["genome_id"].nunique() > 1
    # )
    # if not invalied_contigs_byseq.empty:
    #     logging.info(
    #         f"Removed {invalied_contigs_byseq.shape[0]} that did not have any A,G,T,C Nucleotides"
    #     )
    #     con = con[~con["contig_id"].isin(invalied_contigs_byseq)]
    # if not invalid_contigs_byid.empty:
    #     logging.info(
    #         f"Removed {invalid_contigs_byid.shape[0]} contigs where #anonymous_contig_id were mapped to multiple contig_ids in the mapping file"
    #     )
    #     con = con[~con["contig_id"].isin(invalid_contigs_byid["#anonymous_contig_id"])]

    map = map.drop_duplicates("#anonymous_contig_id", keep="first")

    if dataset in ["airways", "gastro", "oral", "skin", "urogenital"]:
        tax = tax.rename(columns={"_CAMI_GENOMEID": "_CAMI_genomeID"})

    return con, map, tax


def merge_cami_files(
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
    logging.info(f"Number of contigs after cleaning: {out.shape[0]}")

    del con, tax
    return out


def create_vamb_files(
    dataset: str, reads: str, map: pd.DataFrame, output: pd.DataFrame
) -> None:

    valid_contigs = set(output["contig_id"].to_list())
    fasta_output_path = os.path.join(
        os.environ["CAMI2_VAMB_OUTPUT_PATH"], f"{dataset}_{reads}_.gsa.fasta.gz"
    )
    with gzip.open(
        os.path.join(os.environ["raw_data_path"], "gsa.fasta.gz"), "rt"
    ) as handle, gzip.open(fasta_output_path, "wt") as out_handle:
        filtered_records = (
            record
            for record in SeqIO.parse(handle, "fasta")
            if record.id in valid_contigs
        )
        SeqIO.write(filtered_records, out_handle, "fasta")

    abundance_dir = os.path.join(os.environ["raw_data_path"], "abundance")
    abundance_files = [
        os.path.join(abundance_dir, f)
        for f in os.listdir(abundance_dir)
        if f.endswith(".tsv.gz")
    ]
    map = map[map["#anonymous_contig_id"].isin(output["contig_id"])]
    abundances_output = pd.DataFrame(
        {"contigname": map["#anonymous_contig_id"].unique()}
    )

    for file in abundance_files:
        sample_name = "S" + file.split("\\")[-1].split("_")[0]
        print(sample_name)
        sample_abundance = pd.read_csv(file, compression="gzip", sep="\t")
        sample_abundance["#anonymous_contig_id"] = (
            "P" + sample_abundance["#anonymous_contig_id"].str[2:]
        )
        sample_abundance = sample_abundance.rename(
            columns={"number_reads": sample_name}
        )

        abundances_output = pd.merge(
            abundances_output,
            sample_abundance[["#anonymous_contig_id", sample_name]],
            how="left",
            left_on="contigname",
            right_on="#anonymous_contig_id",
        )
        abundances_output = abundances_output.drop(
            columns=["#anonymous_contig_id"]
        ).fillna(0)
        abundances_output.to_csv(
            os.path.join(
                os.environ["CAMI2_VAMB_OUTPUT_PATH"],
                f"{dataset}_{reads}_abundance.tsv",
            ),
            index=False,
            sep="\t",
        )
    del map
    return


def get_summary_stats(output: pd.DataFrame) -> None:
    logging.info(f"\n** Summary Stats **")

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


def save_output(output: pd.DataFrame, dataset: str, reads: str) -> None:
    output.to_csv(
        os.path.join(
            os.environ["CAMI2_GFM_OUTPUT_PATH"], f"{dataset}_{reads}_contigs.csv"
        ),
        index=False,
    )
    print(f"{dataset} {reads} saved successfully.")
    return


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        help="Choose Cami2 datasets (optional). E.g. marine_long, plant_short.",
    )

    args = parser.parse_args()

    return args


def main(args):
    setup_data_paths()
    setup_logfile(os.environ["LOG_PATH"])

    all_datasets = read_dataset_names(os.environ["CONFIG_PATH"])
    all_datasets = args.datasets if args.datasets else all_datasets

    for dataset_id in tqdm(all_datasets, desc="Processing Cami2-datasets"):
        dataset, reads = dataset_id.split("_")
        os.environ["raw_data_path"] = os.path.join(
            os.environ["CAMI2_DATA_PATH"], f"{dataset}_{reads}"
        )
        # download_cami_contigs(dataset, reads)

        con, map, tax = load_raw_cami_files(dataset)

        print("done load")
        con, map, tax = preprocess_cami_files(dataset, con, map, tax)
        print("done preprocess")
        output = merge_cami_files(con, map, tax)
        print("done merge")
        create_vamb_files(dataset, reads, map, output)

        # get_summary_stats(output)
        # save_output(output, dataset, reads)

        # del output
        # shutil.rmtree(os.environ["raw_data_path"])


if __name__ == "__main__":

    args = add_arguments()

    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
