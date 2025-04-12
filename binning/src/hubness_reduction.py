from skhubness.analysis import Hubness
from skhubness.reduction import MutualProximity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# pip install https://github.com/VarIr/scikit-hubness/archive/main.tar.gz


if __name__ == "__main__":

    X = np.load("dnaberts.npz")
    X = X["embeddings"]

    X = X[:10000]
    X = cosine_similarity(X)

    negative_indices = np.where(X < 0)
    print(negative_indices)

    hub = Hubness(k=10, metric="cosine", verbose=2, return_value="all")
    hub.fit(X)
    hubness_metrics = hub.score()
    print("Robinhood Index (Before MP):", hubness_metrics.get("robinhood"))
    print("k-Skewness (Before MP):", hubness_metrics.get("k_skewness"))


# hub_mp = Hubness(k=10, metric="cosine", hubness="mutual_proximity", verbose=2)
# hub_mp.fit(X)
# k_skew_mp = hub_mp.score()
# print(
#     f"Skewness after MP: {k_skew_mp:.3f} "
#     f"(reduction of {k_skew - k_skew_mp:.3f})"
# )
