from typing import Sequence
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import roc_auc_score
from group_lasso import LogisticGroupLasso
from tqdm import tqdm

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*FISTA iterations did not converge.*"
)


def coefs_dict_to_df(coefficients: dict, output_path: str) -> pd.DataFrame:
    """
    Convert coefficients["coefs"] of the form:
      {
        "method1": {
           "featA": [c1, c2, ..., cK],
           "featB": [c1, c2, ..., cK],
           ...
        },
        "method2": { ... },
        ...
      }
    into a long-form DataFrame with columns:
      mil_method, feature, fold, coef
    """
    records = []
    for mil_method, feat_dict in coefficients["coefs"].items():
        for feat, coef_list in feat_dict.items():
            for fold_idx, coef in enumerate(coef_list, start=1):
                records.append(
                    {
                        "mil_method": mil_method,
                        "feature": feat,
                        "fold": fold_idx,
                        "coef": coef,
                    }
                )
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)

    return df


def append_coefs(
    coefficients: dict,
    mil_method: str,
    global_features: list[str],
    local_features: list[str],
    model,
    fold: int,
    gl_model=None,
    groups: list[int] = None,
) -> dict:
    """
    Record the final model’s coefficients for this fold.
    - eval_coefs: dict with key "coefs" holding per-method dicts
    - mil_method: e.g. "logistic_l1", "sparsegrouplasso"
    - feature_names: original column names
    - model: the final LogisticRegression (second-stage for sparsegrouplasso, or plain logistic)
    - fold: the fold index (int)
    - gl_model: if provided, the fitted LogisticGroupLasso used to select groups
    - groups: list of group assignments per original feature (only needed if gl_model is given)
    """

    if mil_method not in coefficients["coefs"]:
        # for each feature, we store a list of length = #folds
        coefficients["coefs"][mil_method] = {feat: [] for feat in global_features}

    feature_to_global_idx = {feat: i for i, feat in enumerate(global_features)}
    full_coefs = np.zeros(len(global_features), dtype=float)

    # ----------------random forest ----------------
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        for imp, feat in zip(importances, local_features):
            global_idx = feature_to_global_idx.get(feat)
            if global_idx is None:
                raise KeyError(f"Local feature {feat} not in global_features list")
            full_coefs[global_idx] = float(imp)

    if isinstance(model, LogisticRegression):
        # --------------sparse group lasso-------------
        if gl_model is not None and groups is not None:

            mask = gl_model.sparsity_mask_
            assert len(mask) == len(local_features)
            sel_coefs = model.coef_.flatten()
            sel_feature_names = [local_features[i] for i, m in enumerate(mask) if m]

            if len(sel_coefs) != len(sel_feature_names):
                raise ValueError(
                    f"Mismatch: {len(sel_coefs)} coefficients vs "
                    f"{len(sel_feature_names)} selected features"
                )

            for coef, feat in zip(sel_coefs, sel_feature_names):
                global_idx = feature_to_global_idx.get(feat)
                if global_idx is None:
                    raise KeyError(f"Local feature {feat} not in global_features list")
                full_coefs[global_idx] = float(coef)

        # --------------Normal logistic-------------
        else:
            coef_vec = model.coef_.flatten()
            if len(full_coefs) != len(local_features):
                raise ValueError(
                    f"Expected {len(local_features)} coefs, got {len(full_coefs)}"
                )

            for coef, feat in zip(coef_vec, local_features):
                global_idx = feature_to_global_idx.get(feat)
                if global_idx is None:
                    raise KeyError(f"Feature {feat} not found in global_features")
                full_coefs[global_idx] = float(coef)

    # append each feature’s coef to its list
    for feat, coef in zip(global_features, full_coefs):
        coefficients["coefs"][mil_method][feat].append(float(coef))

    return coefficients


def append_rf_params(reg_strenghts: dict, mil_method, best_rf):
    if mil_method not in reg_strenghts["regs"]:
        reg_strenghts["regs"][mil_method] = []

    params = {
        "n_estimators": getattr(best_rf, "n_estimators", None),
        "max_features": getattr(best_rf, "max_features", None),
        "min_samples_leaf": getattr(best_rf, "min_samples_leaf", None),
        "criterion": getattr(best_rf, "criterion", None),
    }

    # Append only if all were found (you can relax this if you like)
    if None not in params.values():
        reg_strenghts["regs"][mil_method].append(params)
    else:
        # If any attribute was missing, record None to keep track
        reg_strenghts["regs"][mil_method].append(params)


def append_regurilization_strength(reg_strengths: dict, mil_method: str, best_lr):

    if mil_method not in reg_strengths["regs"]:
        reg_strengths["regs"][mil_method] = []

    reg = best_lr.C if hasattr(best_lr, "C") else None

    # Append the regularization strength if it's available
    if reg is not None:
        reg_strengths["regs"][mil_method].append(reg)

    return reg_strengths


def choose_hyperparams(reg_strengths: dict, log):
    best_regs = {}
    for method, params in reg_strengths.get("regs", {}).items():
        if not params:
            log.append(f"No entries recorded for method '{method}'.")
            best_regs[method] = None
            continue

        if isinstance(params[0], dict):
            tupled = [tuple(sorted(d.items())) for d in params]
            counts = Counter(tupled)
            most_common = counts.most_common()

            log.append(f"\nHyperparameter combinations for '{method}':")
            for combo, cnt in most_common:
                combo_dict = dict(combo)
                log.append(f"  {combo_dict}, Count: {cnt}")

            best_combo = dict(most_common[0][0])
            best_regs[method] = best_combo

        else:
            counts = Counter(params)
            most_common = counts.most_common()
            log.append(f"\nRegularization strengths for method '{method}':")
            for reg, count in most_common:
                log.append(f"  Strength: {reg}, Count: {count}")
            best_regs[method] = most_common[0][0]  # Most frequent strength

    log.append(str(best_regs))
    return best_regs


def fit_predict_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    fold,
    log,
    param_grid: dict,
    cv: int,
    scoring: str,
    coefficients: dict,
    global_features: list,
    reg_strengths: dict,
):
    base_rf = RandomForestClassifier(random_state=42)

    strat_kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=strat_kf,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    log.append(
        f"Best RandomForest params: {search.best_params_}, best {scoring}: {search.best_score_:.4f}"
    )

    best_lr = search.best_estimator_
    y_pred = best_lr.predict(X_test)
    y_predprob = best_lr.predict_proba(X_test)[:, 1]  # probability of class 1

    coefficients = append_coefs(
        coefficients,
        "rf",
        global_features,
        X_train.columns.tolist(),
        best_lr,
        fold,
    )

    reg_strengths = append_rf_params(reg_strengths, f"rf", best_lr)

    return y_pred, y_predprob, coefficients


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
    coefficients: dict,
    global_features: list,
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
    for group_reg in tqdm(group_reg_grid, desc="Grid search on group lasso"):
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
                    n_iter=10_000,
                    tol=1e-3,
                )

                try:
                    gl.fit(X_tr.values, y_tr)
                    X_tr_sel = gl.transform(X_tr.values)
                    X_val_sel = gl.transform(X_val.values)

                    if X_tr_sel.shape[1] == 0:  # if all coefs=0
                        fold_scores.append(0.0)
                        continue

                    lr = LogisticRegression(
                        penalty=None,
                        solver="saga",
                        max_iter=10_000,
                        random_state=0,
                    )
                    lr.fit(X_tr_sel, y_tr)
                    if scoring == "roc_auc":
                        score = roc_auc_score(y_val, lr.predict_proba(X_val_sel)[:, 1])
                        fold_scores.append(score)
                    else:
                        log.append(f"Invalid scoring for Group Sparse Lasso")
                        break

                except Exception as e:
                    log.append(
                        f"Error in GroupLasso CV: Group-Reg: {group_reg}, L1-Reg {l1_reg}"
                    )
                    fold_scores.append(0.0)
                    continue

            mean_score = np.mean(fold_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = (group_reg, l1_reg)

    # Refit best model on full training data
    group_reg, l1_reg = best_params
    log.append(
        f"[Fold {fold}] Best params: group_reg={group_reg}, l1_reg={l1_reg}, best_AUCROC: {best_score:.4f}"
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
    log.append(
        f"[Fold {fold}] Chosen groups (n={len(best_gl.chosen_groups_)}): {best_gl.chosen_groups_}"
    )

    # Transform data, i.e. only keep selected groups
    X_train_sel = best_gl.transform(X_train.values)
    X_test_sel = best_gl.transform(X_test.values)
    log.append(f"[Fold {fold}] Reduced train shape: {X_train_sel.shape}")
    log.append(f"[Fold {fold}] Reduced test shape: {X_test_sel.shape}")

    if X_train_sel.shape[1] == 0:
        log.append(
            f"[Fold {fold}] No features selected after group lasso. Skipping fold."
        )
        return None, None, coefficients

    # Second-stage classifier
    second_lr = LogisticRegression(
        penalty=None,
        solver="saga",
        max_iter=10_000,
        random_state=0,
    )

    second_lr.fit(X_train_sel, y_train)

    y_pred = second_lr.predict(X_test_sel)
    y_predprob = second_lr.predict_proba(X_test_sel)[:, 1]

    coefficients = append_coefs(
        coefficients,
        "sparsegrouplasso",
        global_features,
        X_train.columns.tolist(),
        second_lr,
        fold,
        gl_model=best_gl,
        groups=groups,
    )

    return y_pred, y_predprob, coefficients


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
    coefficients: dict,
    global_features: list,
    reg_strengths: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a LogisticRegression with saga+penalty, grid-searching C if needed,
    and log CV results to the provided Logger.

    Returns predicted labels on X_test.
    """

    base_lr = LogisticRegression(
        penalty=penalty,
        solver="saga",
        max_iter=10_000,
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

        log.append(
            f"Best C: {search.best_params_['C']:.4g}, best {scoring}: {search.best_score_:.4f}"
        )

        best_lr = search.best_estimator_
        y_pred = best_lr.predict(X_test)
        y_predprob = best_lr.predict_proba(X_test)[:, 1]  # probability of class 1

    else:  # no penalty
        best_lr = LogisticRegression(
            penalty=penalty,
            solver="saga",
            max_iter=10_000,
            random_state=0,
        )

        best_lr.fit(X_train, y_train)
        y_pred = best_lr.predict(X_test)
        y_predprob = best_lr.predict_proba(X_test)[:, 1]  # probability of class 1

    coefficients = append_coefs(
        coefficients,
        f"logistic_{penalty}",
        global_features,
        X_train.columns.tolist(),
        best_lr,
        fold,
    )

    reg_strengths = append_regurilization_strength(
        reg_strengths, f"logistic_{penalty}", best_lr
    )

    return y_pred, y_predprob, coefficients
