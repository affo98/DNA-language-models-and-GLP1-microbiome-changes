import numpy as np
from collections import Counter


from sklearn.neighbors import KNeighborsClassifier

from src.utils import Logger

# OTENTIAL USE
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def fit_predict_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    k: int,
    weights: str = "uniform",
) -> np.ndarray:

    knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred


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
