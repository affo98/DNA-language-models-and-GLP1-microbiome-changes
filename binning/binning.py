import csv
import os
from argparse import ArgumentParser

import traceback

import numpy as np


import torch

from src.clustering import KMediod
from src.get_embeddings import get_embeddings
from src.threshold import Threshold
from src.utils import read_contigs


def main(args):

    contigs = read_contigs(args.contigs)

    try:
        embeddings = get_embeddings(
            contigs,
            args.batch_sizes,
            args.model_name,
            args.model_path,
            os.path.join(args.save_path, "embeddings"),
            normalize=True,
        )
    except Exception:
        print(
            f"|===========| Error in getting embeddings for {args.model_name}|===========|\n{traceback.format_exc()}"
        )
    torch.cuda.empty_cache()

    thres = Threshold(
        embeddings,
        n_bins=1000,
        block_size=1000,
        save_path=os.path.join(args.save_path, "threshold"),
    )
    thres.save_histogram()

    # kmediod = KMediod(
    #     embeddings,
    #     min_similarity=0.008,  # 0.0075
    #     min_bin_size=10,
    #     num_steps=3,
    #     max_iter=1000,
    #     normalized=True,
    # )
    # predictions = kmediod.fit()
    # print(predictions)


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--contigs",
        "-c",
        help="contig catalogue from multi-split approach",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        help="Name of the model to use for embedding generation",
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        help="Path to the pretrained model file or directory",
    )
    parser.add_argument(
        "--batch_sizes",
        "-b",
        help="batch sizes for embeddings",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to save the computed embeddings or to load existing ones",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = add_arguments()

    print("BINNING:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)


# def setup_paths() -> None:
#     """Check if the required folders exist, create them if they don't, and set environment variables."""
#     paths = {
#         "DATA_PATH": os.path.join(os.getcwd(), "data"),
#         "LOG_PATH": os.path.join(os.getcwd(), "logs"),
#         "CONFIG_PATH": os.path.join(os.getcwd(), "config"),
#         "CAMI2_OUTPUT_PATH": os.path.join(os.getcwd(), "data", "cami2"),
#         "EMBEDDINGS_PATH": os.path.join(os.getcwd(), "embeddings"),
#     }

#     for var_name, path in paths.items():
#         if not os.path.exists(path):
#             os.makedirs(path)
#             print(f"Created directory: {path}")
#         os.environ[var_name] = path

#     return


# def read_configs() -> tuple[dict, list]:
#     model_config_path = os.path.join(os.environ["CONFIG_PATH"], "models.yml")
#     models_config = {}
#     with open(model_config_path, "r") as handle:
#         models_config = yaml.safe_load(handle)

#     cami2_config_path = os.path.join(os.environ["CONFIG_PATH"], "cami2_processing.yml")
#     read_cami2_datasets = []
#     with open(cami2_config_path, "r") as file:
#         data = yaml.safe_load(file)
#         for dataset in data["datasets"]:
#             read_cami2_datasets.append(dataset["name"])

#     return models_config, read_cami2_datasets
