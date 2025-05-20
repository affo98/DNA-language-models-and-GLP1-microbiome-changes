import os

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


# OTENTIAL USE
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def append_bestk(reg_strengths: dict, mil_method: str, best_lr):

    if mil_method not in reg_strengths["regs"]:
        reg_strengths["regs"][mil_method] = []

    reg = getattr(best_lr, "n_neighbors", None)

    if reg is not None:
        reg_strengths["regs"][mil_method].append(reg)

    return reg_strengths


def fit_predict_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    log,
    k_grid: list[int],
    fold: int,
    output_path: str,
    weights: str,
    cv: str,
    scoring: str,
    reg_strengths: dict,
) -> tuple[np.ndarray, np.ndarray]:

    strat_kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    knn_base = KNeighborsClassifier(weights=weights)

    search = GridSearchCV(
        knn_base,
        param_grid={"n_neighbors": k_grid},
        cv=strat_kf,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    log.append(
        f"Best K: {search.best_estimator_}, best {scoring}: {search.best_score_:.4f}"
    )

    best_knn = search.best_estimator_
    y_pred = best_knn.predict(X_test)
    y_predprob = best_knn.predict_proba(X_test)[:, 1]  # Probability of class 1

    reg_strengths = append_bestk(reg_strengths, f"knn", best_knn)

    if output_path is not None:
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        plot_knn_decision_boundary(
            X=X_train_pca,
            y=y_train,
            k=getattr(best_knn, "n_neighbors", None),
            weights=weights,
            title=f"KNN Decision Boundary (Fold {fold})",
            save_path=os.path.join(output_path, f"knn_boundary_fold{fold}.png"),
        )
    return y_pred, y_predprob


def plot_knn_decision_boundary(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 5,
    weights: str = "uniform",
    title: str = None,
    save_path: str = None,
):
    # Ensure columns are string type if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        X.columns = X.columns.astype(str)
        xlabel, ylabel = X.columns[0], X.columns[1]
    else:
        X = np.array(X)
        xlabel, ylabel = "Feature 1", "Feature 2"

    knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knn.fit(X, y)

    disp = DecisionBoundaryDisplay.from_estimator(
        knn,
        X,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.4,
    )
    scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="viridis")
    disp.ax_.set_xlabel(xlabel)
    disp.ax_.set_ylabel(ylabel)
    disp.ax_.set_title(title or f"KNN (k={k}, weights='{weights}')")
    legend = disp.ax_.legend(*scatter.legend_elements(), title="Classes")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
