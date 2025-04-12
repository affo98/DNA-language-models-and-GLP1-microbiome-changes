from skhubness.analysis import Hubness
from skhubness.reduction import MutualProximity
import numpy as np


# pip install https://github.com/VarIr/scikit-hubness/archive/main.tar.gz


# if __name__ == "__main__":

#     X = np.load("dnaberts.npz")
#     X = X["embeddings"]

#     hub = Hubness(k=10, metric="cosine", verbose=2, return_value="all")
#     hub.fit(X)
#     hubness_metrics = hub.score()
#     print("Robinhood Index (Before MP):", hubness_metrics.get("robinhood"))
#     print("k-Skewness (Before MP):", hubness_metrics.get("k_skewness"))


# hub_mp = Hubness(k=10, metric="cosine", hubness="mutual_proximity", verbose=2)
# hub_mp.fit(X)
# k_skew_mp = hub_mp.score()
# print(
#     f"Skewness after MP: {k_skew_mp:.3f} "
#     f"(reduction of {k_skew - k_skew_mp:.3f})"
# )


from skhubness.analysis import Hubness
from skhubness.reduction import MutualProximity
import numpy as np


def calculate_robinhood(k_occurrence):
    """Calculates the Robin Hood index from k-occurrence counts."""
    n = len(k_occurrence)
    total_occurrences = sum(k_occurrence)
    mean_occurrence = total_occurrences / n
    redistribution = sum(max(0, count - mean_occurrence) for count in k_occurrence)
    robinhood_index = (
        redistribution / total_occurrences if total_occurrences > 0 else 0.0
    )
    return robinhood_index


def calculate_k_skewness(k_occurrence):
    """Calculates the k-skewness from k-occurrence counts."""
    mean_k = np.mean(k_occurrence)
    std_k = np.std(k_occurrence)
    if std_k == 0:
        return 0.0
    return np.mean(((k_occurrence - mean_k) / std_k) ** 3)


if __name__ == "__main__":

    X = np.load("dnaberts.npz")
    X = X["embeddings"]

    k_neighbors = 10
    metric = "cosine"

    # Calculate hubness before MP
    hub_before = Hubness(k=k_neighbors, metric=metric, verbose=0, return_value="all")
    hub_before.fit(X)
    hubness_before = hub_before.score()
    print("--- Hubness Before Mutual Proximity ---")
    print("Robinhood Index:", hubness_before.get("robinhood"))
    print("k-Skewness:", hubness_before.get("k_skewness"))
    print("-" * 40)

    # Reduce hubness using Mutual Proximity
    mp = MutualProximity(k=k_neighbors, method="standard")
    mp.fit(X)
    X_reduced_dist = mp.transform(X)

    # Find k-Nearest Neighbors based on reduced distances
    n_samples = X.shape[0]
    neighbor_indices_after_mp = np.zeros((n_samples, k_neighbors), dtype=int)
    for i in range(n_samples):
        distances_i = X_reduced_dist[i]
        nearest_indices = np.argsort(distances_i)[:k_neighbors]
        neighbor_indices_after_mp[i] = nearest_indices

    # Calculate k-occurrence from the neighbor indices
    k_occurrence_after_mp = np.zeros(n_samples, dtype=int)
    for neighbors in neighbor_indices_after_mp:
        for neighbor_idx in neighbors:
            k_occurrence_after_mp[neighbor_idx] += 1

    # Calculate Robinhood and k-Skewness from k-occurrence
    robinhood_after_mp = calculate_robinhood(k_occurrence_after_mp)
    k_skewness_after_mp = calculate_k_skewness(k_occurrence_after_mp)

    print("\n--- Hubness After Standard Mutual Proximity (Manual Calculation) ---")
    print("Robinhood Index:", robinhood_after_mp)
    print("k-Skewness:", k_skewness_after_mp)
    print("-" * 40)

    # You can repeat this for other MP methods if needed
