"""Functions to evaluate models given true labels and predictions."""
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def append_eval_metrics(eval_metrics:dict,y_true, y_pred, mil_method:str, fold:str) -> dict[str, float]:
    """Compute evaluation metrics given true labels and predictions."""
    
    eval_metrics_fold = {
        "fold": fold,
        "mil_method": mil_method,
        "f1_score": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred)
    }
    
    eval_metrics['metrics'].append(eval_metrics_fold)
    
    
    return eval_metrics
    }