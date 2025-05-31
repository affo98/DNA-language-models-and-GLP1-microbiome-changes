import os
from argparse import ArgumentParser
from time import time
import json


import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from src.utils import (
    Logger,
    read_sample_labels,
    read_cluster_abundances,
    read_hausdorff,
)
from src.cluster_catalogue import get_cluster_catalogue


from src.knn_model import fit_predict_knn
from src.classifiers import (
    fit_predict_logistic,
    fit_predict_sparsegrouplasso,
    fit_predict_rf,
    coefs_dict_to_df,
    choose_hyperparams,
)
from src.agglmorative_clustering import get_groups_agglomorative

from src.eval import append_eval_metrics, compute_summary_eval, append_permutation_test

# DISTANCE_METRIC_BAG = "cosine"

MIL_METHODS = ["knn", "logistic", "logistic_groupsparselasso", "rf"]


# agglomorative
DISTANCE_METRIC_AGG = "euclidean"  # wards can not use cosine
LINKAGE_AGG = "ward"
TSNE_PERPLEXITY = 17

# cv
CV_OUTER = 10
N_REPEATS = 5

# params knn
KNN_K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
KNN_WEIGHTS = "uniform"

# params logistic
# C_GRID = np.logspace(-4, 4, 10)
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
CV_INNER = 5
SCORING_CV = "roc_auc"

# params sparse group lasso logistic
GROUP_REGS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
L1_REGS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# random forest (same grid as deepmicro)
RF_GRID = [
    {
        "n_estimators": [s for s in range(100, 1001, 200)],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "criterion": ["gini", "entropy"],
    }
]

# permutation test
N_PERMUTATIONS = 1000


def main(args, log):
    # -------------------------------------------- Read data --------------------------------------------
    # cluster_catalogue_centroid = get_cluster_catalogue(args.input_path, log) # == list(cluster_catalogue_centroid.keys())

    sample_ids, labels = read_sample_labels(
        args.sample_labels_path, log, split_train_test=False
    )

    cluster_abundances = read_cluster_abundances(args.input_path, sample_ids, log)

    if args.model_name != "vamb":
        hausdorff, hausdorff_clusternames = read_hausdorff(
            os.path.join(
                args.input_path,
                "hausdorff",
                f"{args.model_name}_{args.dataset_name}.npz",
            ),
            log,
        )
        assert all(isinstance(x, str) for x in hausdorff_clusternames)
        assert all(isinstance(x, str) for x in hausdorff_clusternames)
        assert np.array_equal(
            cluster_abundances.columns[1:].values, hausdorff_clusternames
        )

    assert all(isinstance(x, str) for x in cluster_abundances.columns[1:].values)
    assert all(isinstance(x, str) for x in cluster_abundances["sample"].values)
    assert all(isinstance(x, str) for x in sample_ids)

    assert np.array_equal(cluster_abundances["sample"].values, sample_ids)
    assert len(cluster_abundances["sample"]) == len(sample_ids)

    # agglomorative
    if args.model_name != "vamb":
        n_groups = int(hausdorff.shape[0] ** 0.5)
        groups = get_groups_agglomorative(
            hausdorff,
            hausdorff_clusternames,
            n_groups,
            DISTANCE_METRIC_AGG,
            LINKAGE_AGG,
            TSNE_PERPLEXITY,
            args.agglomorative_path,
            log,
        )
        log.append(f"Using {n_groups} groups")

    # -------------------------------------------- CV Evaluate --------------------------------------------
    eval_metrics = {"metrics": []}
    cluster_abundance_features = cluster_abundances.iloc[:, 1:]
    print(cluster_abundance_features.shape)
    global_features = cluster_abundances.columns.drop("sample").tolist()
    coefficients = {"coefs": {}}
    reg_strengths = {"regs": {}}
    skf = StratifiedKFold(n_splits=CV_OUTER, shuffle=True, random_state=42)
    rskf = RepeatedStratifiedKFold(
        n_splits=CV_OUTER, n_repeats=N_REPEATS, random_state=42
    )

    for fold_idx, (train_idx, test_idx) in enumerate(
        rskf.split(cluster_abundance_features, labels)
    ):
        # ------------ split ------------
        cluster_abundances_train, cluster_abundances_test = (
            cluster_abundance_features.iloc[train_idx, :],
            cluster_abundance_features.iloc[test_idx, :],
        )
        labels_train, labels_test = labels[train_idx], labels[test_idx]
        sample_ids_train, sample_ids_test = sample_ids[train_idx], sample_ids[test_idx]

        log.append(
            f"{'-'*20} Fold {fold_idx+1} {'-'*20}\n"
            f"- Train samples: n={len(labels_train)}, 0s={len(labels_train) - np.sum(labels_train)}, 1s={np.sum(labels_train)}\n{sample_ids_train}\n"
            f"- Test samples:  n={len(labels_test)},  0s={len(labels_test) - np.sum(labels_test)}, 1s={np.sum(labels_test)}\n{sample_ids_test}"
        )

        assert (
            len(cluster_abundances_train) == len(sample_ids_train) == len(labels_train)
        )
        assert len(cluster_abundances_test) == len(sample_ids_test) == len(labels_test)

        # ------------ MIL ------------
        for mil_method in args.mil_methods:
            log.append(f"Using MIL method: {mil_method}")

            if mil_method == "knn":
                log.append(f"  → Training KNN'")
                predictions, predictions_proba = fit_predict_knn(
                    X_train=cluster_abundances_train,
                    X_test=cluster_abundances_test,
                    y_train=labels_train,
                    log=log,
                    k_grid=KNN_K,
                    fold=fold_idx + 1,
                    output_path=None,
                    weights=KNN_WEIGHTS,
                    cv=CV_INNER,
                    scoring=SCORING_CV,
                    reg_strengths=reg_strengths,
                )
                eval_metrics = append_eval_metrics(
                    eval_metrics,
                    labels_test,
                    predictions,
                    predictions_proba,
                    mil_method,
                    fold_idx + 1,
                )
                del predictions, predictions_proba

            elif mil_method == "logistic":
                for penalty in [None, "l1", "l2", "elasticnet"]:
                    log.append(
                        f"  → Training logistic regression with penalty='{penalty}'"
                    )
                    (
                        predictions,
                        predictions_proba,
                        coefficients,
                    ) = fit_predict_logistic(
                        X_train=cluster_abundances_train,
                        X_test=cluster_abundances_test,
                        y_train=labels_train,
                        fold=fold_idx + 1,
                        output_path=args.output_path,
                        penalty=penalty,
                        log=log,
                        C_grid=C_GRID,
                        cv=CV_INNER,
                        scoring=SCORING_CV,
                        coefficients=coefficients,
                        global_features=global_features,
                        reg_strengths=reg_strengths,
                    )
                    eval_metrics = append_eval_metrics(
                        eval_metrics,
                        labels_test,
                        predictions,
                        predictions_proba,
                        f"{mil_method}_{penalty}",
                        fold_idx + 1,
                    )
                    del predictions, predictions_proba

            elif mil_method == "logistic_groupsparselasso":
                if args.model_name != "vamb":
                    (
                        predictions,
                        predictions_proba,
                        coefficients,
                    ) = fit_predict_sparsegrouplasso(
                        X_train=cluster_abundances_train,
                        X_test=cluster_abundances_test,
                        y_train=labels_train,
                        groups=groups,
                        fold=fold_idx + 1,
                        output_path=args.output_path,
                        log=log,
                        group_reg_grid=GROUP_REGS,
                        l1_reg_grid=L1_REGS,
                        cv=CV_INNER,
                        scoring=SCORING_CV,
                        coefficients=coefficients,
                        global_features=global_features,
                    )
                    if predictions is None:  # can happen that no features are used.
                        continue

                    eval_metrics = append_eval_metrics(
                        eval_metrics,
                        labels_test,
                        predictions,
                        predictions_proba,
                        mil_method,
                        fold_idx + 1,
                    )
                    del predictions, predictions_proba

            elif mil_method == "rf":
                log.append(f"  → Training RF")
                predictions, predictions_proba, coefficients = fit_predict_rf(
                    X_train=cluster_abundances_train,
                    X_test=cluster_abundances_test,
                    y_train=labels_train,
                    fold=fold_idx + 1,
                    log=log,
                    param_grid=RF_GRID,
                    cv=CV_INNER,
                    scoring=SCORING_CV,
                    coefficients=coefficients,
                    global_features=global_features,
                    reg_strengths=reg_strengths,
                )
                eval_metrics = append_eval_metrics(
                    eval_metrics,
                    labels_test,
                    predictions,
                    predictions_proba,
                    mil_method,
                    fold_idx + 1,
                )
                del predictions, predictions_proba

    coefs_df = coefs_dict_to_df(
        coefficients, os.path.join(args.output_path, f"coefs.csv")
    )

    with open(
        os.path.join(
            args.output_path,
            f"eval_metrics_{args.model_name}_{args.dataset_name}.json",
        ),
        "w",
    ) as f:
        json.dump(eval_metrics, f, indent=4)

    summary_eval = compute_summary_eval(eval_metrics)
    log.append(f"{summary_eval}")
    with open(
        os.path.join(
            args.output_path,
            f"eval_metrics_summary_{args.model_name}_{args.dataset_name}.json",
        ),
        "w",
    ) as f:
        json.dump(summary_eval, f, indent=4)

    # permutation test with majority voting regurilization strengths
    best_regs = choose_hyperparams(reg_strengths, log)

    permutation_results = {"perms": {}}
    for mil_method in args.mil_methods:

        if mil_method == "knn":
            best_k = best_regs.get("knn", None)
            mean_score = summary_eval["knn"].get("auc_roc_mean")
            log.append(f"  → Permutation test for KNN using K: {best_k} '")
            permutation_results = append_permutation_test(
                X=cluster_abundance_features,
                y=labels,
                mil_method=mil_method,
                penalty=None,
                best_reg=None,
                k=best_k,
                rf_params=None,
                permutation_results=permutation_results,
                scoring=SCORING_CV,
                cv=skf,
                n_permutations=N_PERMUTATIONS,
                score=mean_score,
            )

        elif mil_method == "logistic":
            for penalty in [None, "l1", "l2", "elasticnet"]:
                id = f"logistic_{penalty}"
                best_reg = best_regs.get(id, None)
                mean_score = summary_eval[id].get("auc_roc_mean", None)
                log.append(
                    f"  → Permutation test logistic_{penalty} using regurilization strength: {best_reg} '"
                )

                if (
                    penalty == "l1" and args.model_name != "vamb"
                ):  # groupsparselasso, use L1 as baseline
                    permutation_results = append_permutation_test(
                        X=cluster_abundance_features,
                        y=labels,
                        mil_method="logistic_groupsparselasso",
                        penalty=penalty,
                        best_reg=best_reg,
                        k=None,
                        rf_params=None,
                        permutation_results=permutation_results,
                        scoring=SCORING_CV,
                        cv=skf,
                        n_permutations=N_PERMUTATIONS,
                        score=summary_eval["logistic_groupsparselasso"].get(
                            "auc_roc_mean", None
                        ),
                    )

                permutation_results = append_permutation_test(
                    X=cluster_abundance_features,
                    y=labels,
                    mil_method=mil_method,
                    penalty=penalty,
                    best_reg=best_reg,
                    k=None,
                    rf_params=None,
                    permutation_results=permutation_results,
                    scoring=SCORING_CV,
                    cv=skf,
                    n_permutations=N_PERMUTATIONS,
                    score=mean_score,
                )
        elif mil_method == "rf":
            best_rf_params = best_regs.get("rf", None)
            mean_score = summary_eval["rf"].get("auc_roc_mean")
            log.append(f"  → Permutation test for RF using params: {best_rf_params} '")
            permutation_results = append_permutation_test(
                X=cluster_abundance_features,
                y=labels,
                mil_method=mil_method,
                penalty=None,
                best_reg=None,
                k=None,
                rf_params=best_rf_params,
                permutation_results=permutation_results,
                scoring=SCORING_CV,
                cv=skf,
                n_permutations=N_PERMUTATIONS,
                score=mean_score,
            )

    log.append(str(permutation_results))
    with open(
        os.path.join(
            args.output_path,
            f"permutation_{args.model_name}_{args.dataset_name}.json",
        ),
        "w",
    ) as f:
        json.dump(permutation_results, f, indent=4)


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
        choices=MIL_METHODS + ["all"],
        help="MIL method to use",
    )
    parser.add_argument(
        "--agglomorative_path",
        "-ag",
        help="Path to save/load agglomorative clustering",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time()
    args = add_arguments()

    log = Logger(args.log)

    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    if "all" in args.mil_methods:
        args.mil_methods = MIL_METHODS
        log.append(f"Running with {MIL_METHODS}")

    os.makedirs(args.output_path, exist_ok=True)
    main(args, log)

    end_time = time()
    elapsed_time = end_time - start_time
    log.append(f"MIL Phenotype of {args.model_name} ran in {elapsed_time:.2f} Seconds")
