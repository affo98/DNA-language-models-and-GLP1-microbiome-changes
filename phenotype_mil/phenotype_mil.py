import os
from argparse import ArgumentParser
from time import time

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.utils import Logger, read_sample_labels, read_cluster_abundances
from src.cluster_catalogue import get_cluster_catalogue

from src.knn_model import KNNModel

from src.eval import append_eval_metrics

DISTANCE_METRIC_BAG = "cosine"

# params knn
KNN_K = 5


def main(args, log):

    sample_ids, labels = read_sample_labels(
        args.sample_labels_path, log, split_train_test=False
    )
    print(sample_ids, labels)

    cluster_catalogue_centroid = get_cluster_catalogue(args.input_path, log)

    cluster_abundances = read_cluster_abundances(args.input_path, log)

    assert set(cluster_abundances.columns[1:].to_list()) == set(
        cluster_catalogue_centroid.keys()
    ), log.append("CLuster catalogue and abundances do not match!")

    assert set(cluster_abundances["sample"].values.tolist()) == set(
        sample_ids
    ), log.append("Sample ids do not match!")

    # Evaluate model
    eval_metrics = {"metrics": []}
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(cluster_abundances, labels)
    ):
        fold = f"fold_{fold_idx + 1}"
        log.append(fold)
        abundances_train, abundances_test = (
            cluster_abundances[train_idx],
            cluster_abundances[test_idx],
        )
        labels_train, labels_test = labels[train_idx], labels[test_idx]
        sample_ids_train, sample_ids_test = sample_ids[train_idx], sample_ids[test_idx]
        log.append(
            f"Fold {fold} - Train samples: {len(sample_ids_train)}, Test samples: {len(sample_ids_test)}"
        )

        # for mil_method in args.mil_method:
        #     log.append(f"Using MIL method: {mil_method}")

        #     if mil_method == "knn":
        #         knnmodel = KNNModel(
        #             labels_train,
        #             labels_test,
        #             abundances_train,
        #             abundances_test,
        #             log=log,
        #         )
        #         predictions = knnmodel.predict(
        #             k=KNN_K, distance_metric=DISTANCE_METRIC_BAG
        #         )

        #     elif mil_method == "classifier":
        #         pass
        #     elif mil_method == "graph":
        #         pass

        #     eval_metrics = append_eval_metrics(
        #         eval_metrics, labels_test, predictions, mil_method, fold
        #     )


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-mn",
        help="Name of the model to use for embedding generation",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--input_path",
        "-i",
        help="Path to the input directory contining cluster results",
    )
    parser.add_argument(
        "--sample_labels_path",
        "-s",
        help="Path to the sample labels file",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        help="Save path for MIL",
    )
    parser.add_argument(
        "--log",
        "-l",
        help="Path to save logfile",
    )
    parser.add_argument(
        "--mil_methods",
        "-m",
        nargs="+",
        choices=["knn", "classifier", "graph"],
        help="MIL method to use",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time()
    args = add_arguments()

    log = Logger(args.log)

    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    os.makedirs(args.output_path, exist_ok=True)
    main(args, log)

    end_time = time()
    elapsed_time = end_time - start_time
    log.append(f"MIL Phenotype of {args.model_name} ran in {elapsed_time:.2f} Seconds")
