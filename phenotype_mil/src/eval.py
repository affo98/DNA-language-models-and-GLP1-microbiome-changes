"""Functions to evaluate models given true labels and predictions."""

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


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
            stats[f"{metric}_mean"] = round(float(group[metric].mean(), 2))
            stats[f"{metric}_std"] = round(
                float(group[metric].std(ddof=0)), 2
            )  # population std
        summary_eval[method] = stats

    return summary_eval
