"""Functions to evaluate models given true labels and predictions."""

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import permutation_test_score


def append_permutation_test(
    X,
    y,
    mil_method: str,
    penalty: str,
    best_reg: float,
    k: int,
    permutation_results: dict,
    scoring: str,
    cv: int,
    n_permutations: int,
):

    # knn
    if penalty is None and best_reg is None and k is not None:
        estimator = KNeighborsClassifier(n_neighbors=k, weights="uniform")

    # logistic no penalty
    elif penalty is None and best_reg is not None and k is None:
        estimator = LogisticRegression(
            penalty=penalty,
            solver="saga",
            max_iter=5_000,
            tol=3e-3,
            random_state=42,
        )

    # logistic penalty
    elif penalty is not None and best_reg is not None and k is None:
        estimator = LogisticRegression(
            penalty=penalty,
            C=best_reg,
            solver="saga",
            max_iter=5_000,
            tol=3e-3,
            random_state=42,
            l1_ratio=0.5 if penalty == "elasticnet" else None,
        )

    score, perm_scores, p_value = permutation_test_score(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_permutations=n_permutations,
        random_state=42,
        n_jobs=-1,
    )

    if penalty is None and best_reg is None and k is not None:
        if mil_method not in permutation_results:
            permutation_results["perms"][mil_method] = [score, p_value]

    elif penalty is None and best_reg is not None and k is None:
        if mil_method + "_" + "None" not in permutation_results:
            permutation_results["perms"][f"{mil_method}_None"] = [score, p_value]

    elif penalty is not None and best_reg is not None and k is None:
        if mil_method + "_" + penalty not in permutation_results:
            permutation_results["perms"][f"{mil_method}_{penalty}"] = [score, p_value]

    return permutation_results


def append_eval_metrics(
    eval_metrics: dict, y_true, y_pred, y_predprob, mil_method: str, fold: str
) -> dict[str, float]:
    """Compute evaluation metrics given true labels and predictions."""

    eval_metrics_fold = {
        "fold": fold,
        "mil_method": mil_method,
        "f1_score": float(f1_score(y_true, y_pred)),
        "auc_roc": float(
            roc_auc_score(y_true, y_predprob)
        ),  # uses proba instead of [0,1]
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    eval_metrics["metrics"].append(eval_metrics_fold)

    return eval_metrics


def compute_summary_eval(eval_metrics: dict) -> dict[str, dict[str, float]]:
    """
    Compute mean and std of method/metric
    """

    # Turn the list of per-fold dicts into a DataFrame
    df = pd.DataFrame(eval_metrics["metrics"])

    summary_eval = {}
    for method, group in df.groupby("mil_method"):
        stats = {}
        for metric in ("f1_score", "auc_roc", "accuracy"):
            stats[f"{metric}_mean"] = round(float(group[metric].mean()), 3)
            stats[f"{metric}_std"] = round(
                float(group[metric].std(ddof=0)), 3
            )  # population std
        summary_eval[method] = stats

    return summary_eval
