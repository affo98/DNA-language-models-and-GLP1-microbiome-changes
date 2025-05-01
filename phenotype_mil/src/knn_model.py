import os

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


# OTENTIAL USE
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def fit_predict_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    fold_idx: int,
    output_path: str,
    weights: str = "uniform",
) -> np.ndarray:

    knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(X_train.shape)
    if X_train.shape[1] == 2 and output_path and fold_idx is not None:
        print("si")
        plot_knn_decision_boundary(
            X=X_train,
            y=y_train,
            k=k,
            weights=weights,
            title=f"KNN Decision Boundary (Fold {fold_idx+1})",
            save_path=os.path.join(output_path, f"knn_boundary_fold{fold_idx+1}.png"),
        )
    return y_pred


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


# class KNNModel:

#     def __init__(
#         self,
#         labels_train: np.array,
#         labels_test: np.array,
#         abundances_train: np.array,
#         abundances_test: np.array,
#         log=Logger,
#         save_path=str,
#     ):

#         self.labels_train = labels_train
#         self.labels_test = labels_test
#         self.abundances_train = abundances_train
#         self.abundances_test = abundances_test
#         self.log = log
#         self.save_path = save_path

#     def predict(self, k: int, distance_metric: str) -> list[str]:
#         """Predict the labels for the test set using KNN."""

#         distances = cdist(
#             self.abundances_test, self.abundances_train, metric=distance_metric
#         )
#         self.log.append(f"KNN distance matrix (test X train) {distances.shape}")
#         top_k_indices = np.argsort(distances, axis=1)[:, :k]
#         top_k_labels = np.array(
#             [np.array(self.labels_train)[indices] for indices in top_k_indices]
#         )
#         self.log.append("topk labels", top_k_labels)

#         predictions = []
#         for labels in top_k_labels:
#             print(labels)
#             majority_label = Counter(labels).most_common(1)[0][
#                 0
#             ]  # majority voting pred
#             print(majority_label)
#             predictions.append(majority_label)

#         self.predictions = predictions
#         return predictions

#     def plot_predictions(self): ...
