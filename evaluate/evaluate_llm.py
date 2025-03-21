import os
import csv
from argparse import ArgumentParser
import yaml

import numpy as np

from evaluate.src.clustering import KMediod
from evaluate.src.get_embeddings import get_embeddings

csv.field_size_limit(2**30)


def setup_paths() -> None:
    """Check if the required folders exist, create them if they don't, and set environment variables."""
    paths = {
        "DATA_PATH": os.path.join(os.getcwd(), "data"),
        "LOG_PATH": os.path.join(os.getcwd(), "logs"),
        "CONFIG_PATH": os.path.join(os.getcwd(), "config"),
        "CAMI2_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2"),
        "EMBEDDINGS_PATH": os.path.join(os.getcwd(), "embeddings"),
    }

    for var_name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        os.environ[var_name] = path

    return


def read_configs() -> tuple[dict, list]:
    model_config_path = os.path.join(os.environ["CONFIG_PATH"], "models.yml")
    models_config = {}
    with open(model_config_path, "r") as handle:
        models_config = yaml.safe_load(handle)

    cami2_config_path = os.path.join(os.environ["CONFIG_PATH"], "cami2_processing.yml")
    read_cami2_datasets = []
    with open(cami2_config_path, "r") as file:
        data = yaml.safe_load(file)
        for dataset in data["datasets"]:
            read_cami2_datasets.append(dataset["name"])

    return models_config, read_cami2_datasets


def read_cami2_dataset(dataset) -> tuple[list[str], list[list[str]]]:
    contigs_file = os.path.join("data/cami2/", f"{dataset}_contigs.csv")
    with open(contigs_file) as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
        contigs = [line[12] for line in data[1:]]
        metadata = [line[1:2] + line[4:11] for line in data[1:]]

    return contigs, metadata


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--cami2_datasets",
        "-d",
        help="dataset cami2 to include",
    )

    parser.add_argument(
        "--min_similarity",
        "-s",
        help="dataset cami2 to include",
    )

    args = parser.parse_args()

    return args


def main(args):

    setup_paths()
    models_config, cami2_datasets = read_configs()
    cami2_datasets = args.cami2_datasets if args.cami2_datasets else cami2_datasets

    for dataset in cami2_datasets:
        # contigs, metadata = read_cami2_dataset(dataset)

        for model_name in list(models_config.keys()):
            print("\n========================================= \n")
            print(f"Using {model_name} to calculate embeddings\n")
            print("========================================= \n\n")
            model_path = models_config[model_name]["model_path"]
            save_path = os.path.join(
                os.environ["EMBEDDINGS_PATH"],
                f"{dataset}",
                models_config[model_name]["save_name"],
            )
            print(save_path)

        #     embeddings = get_embeddings(
        #         contigs,
        #         model_name,
        #         model_path,
        #         save_path,
        #     )

        # kmediod = KMediod(
        #     embeddings,
        #     min_similarity=float(args.min_similarity),  # 0.0075
        #     min_bin_size=10,
        #     num_steps=3,
        #     max_iter=1000,
        #     normalized=False,
        # )
        # predictions = kmediod.fit()
        # print(predictions)


if __name__ == "__main__":

    args = add_arguments()

    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
