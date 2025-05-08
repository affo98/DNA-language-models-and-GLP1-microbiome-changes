from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from group_lasso import LogisticGroupLasso



def fit_predict_sparsegrouplasso(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    groups: Sequence[int],
    fold: int,
    output_path: str,
    log,
    group_reg_grid: Sequence[float],
    l1_reg_grid: Sequence[float],
    cv: int,
    scoring: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs manual grid search over group_reg and l1_reg for LogisticGroupLasso,
    logs CV results, transforms data, fits a second-stage LogisticRegression,
    and returns predictions and probabilities on X_test.
    """

    best_score = -np.inf
    best_params = None
    best_gl = None

    inner_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)

    # Grid search loop
    for group_reg in group_reg_grid:
        for l1_reg in l1_reg_grid:
            fold_scores = []

            for train_idx, val_idx in inner_cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx, :], X_train.iloc[val_idx, :]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                gl = LogisticGroupLasso(
                    groups=groups,
                    group_reg=group_reg,
                    l1_reg=l1_reg,
                    scale_reg="inverse_group_size",
                    random_state=42,
                    supress_warning=True,
                )

                try:
                    gl.fit(X_tr.values, y_tr)
                    X_tr_sel = gl.transform(X_tr.values)
                    X_val_sel = gl.transform(X_val.values)
                    

                    if X_tr_sel.shape[1] == 0: #if all coefs=0
                        fold_scores.append(0.0)
                        continue

                    lr = LogisticRegression(
                        solver="lbfgs", max_iter=1000, random_state=0
                    )
                    lr.fit(X_tr_sel, y_tr)
                    
                    score = roc_auc_score(y_val, lr.predict_proba(X_val_sel)[:, 1])
                    fold_scores.append(score)

                except Exception as e:
                    log.append(f"Error in GroupLasso CV: Group-Reg: {group_reg}, L1-Reg {l1_reg}")
                    fold_scores.append(0.0)
                    continue

            mean_score = np.mean(fold_scores)
            log.append(
                f"[Fold {fold}] group_reg={group_reg}, l1_reg={l1_reg}, {scoring}={mean_score:.4f}"
            )

            if mean_score > best_score:
                best_score = mean_score
                best_params = (group_reg, l1_reg)

    # Refit best model on full training data
    group_reg, l1_reg = best_params
     log.append(
        f"[Fold {fold}] Best params: group_reg={group_reg}, l1_reg={l1_reg}, best_score={best_score:.4f}"
    )
    
    best_gl = LogisticGroupLasso(
        groups=groups,
        group_reg=group_reg,
        l1_reg=l1_reg,
        scale_reg="inverse_group_size",
        random_state=42,
        supress_warning=True,
    )
    best_gl.fit(X_train.values, y_train)
    log.append(f"[Fold {fold}] Chosen groups: {best_gl.chosen_groups_}")

    # Transform data, i.e. only keep selected groups
    X_train_sel = best_gl.transform(X_train.values)
    X_test_sel = best_gl.transform(X_test.values)
    log.append(f"[Fold {fold}] Reduced train shape: {X_train_sel.shape}")
    log.append(f"[Fold {fold}] Reduced test shape: {X_test_sel.shape}")

    # Second-stage classifier
    second_lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
    second_lr.fit(X_tr_sel, y_train)

    y_pred = second_lr.predict(X_train_sel)
    y_predprob = second_lr.predict_proba(X_test_sel)[:, 1]

    return y_pred, y_predprob



def fit_predict_logistic(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    output_path: str,
    penalty: str,
    log,
    C_grid: Sequence[float],
    cv: int,
    scoring: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a LogisticRegression with saga+penalty, grid-searching C if needed,
    and log CV results to the provided Logger.

    Returns predicted labels on X_test.
    """

    base_lr = LogisticRegression(
        penalty=penalty,
        solver="saga",
        max_iter=2000,
        random_state=0,
        l1_ratio=0.5 if penalty == "elasticnet" else None,
    )

    if penalty in {"l1", "l2", "elasticnet"}:
        param_grid = {"C": C_grid}
        strat_kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)

        search = GridSearchCV(
            estimator=base_lr,
            param_grid=param_grid,
            cv=strat_kf,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        # Log the CV results
        log.append(f"--- CV results for fold {fold}, penalty={penalty} ---")
        res = search.cv_results_
        for mean, std, C in zip(
            res["mean_test_score"], res["std_test_score"], res["param_C"]
        ):
            log.append(f"C={C:.4g}  {scoring}: {mean:.4f} ± {std:.4f}")
        log.append(
            f"Best C: {search.best_params_['C']:.4g}, best {scoring}: {search.best_score_:.4f}"
        )

        best_lr = search.best_estimator_
        y_pred = best_lr.predict(X_test)
        y_predprob = best_lr.predict(X_test)[:, 1]  # probability of class 1

    else:
        # no tuning needed for penalty='none'
        base_lr.fit(X_train, y_train)
        y_pred = base_lr.predict(X_test)
        y_predprob = best_lr.predict(X_test)[:, 1]  # probability of class 1

    return y_pred, y_predprob
