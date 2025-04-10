from skhubness import Hubness
import numpy as np


if name == "main":

    X = np.load("dnaberts.npz")

    hub = Hubness(k=10, metric="cosine")
    hub.fit(X)
    k_skew = hub.score()
    print(f"Skewness = {k_skew:.3f}")
    print(f"Robin hood index: {hub.robinhood_index:.3f}")
    print(f"Antihub occurrence: {hub.antihub_occurrence:.3f}")
    print(f"Hub occurrence: {hub.hub_occurrence:.3f}")

    hub_mp = Hubness(k=10, metric="cosine", hubness="mutual_proximity")
    hub_mp.fit(X)
    k_skew_mp = hub_mp.score()
    print(
        f"Skewness after MP: {k_skew_mp:.3f} "
        f"(reduction of {k_skew - k_skew_mp:.3f})"
    )
    print(
        f"Robin hood: {hub_mp.robinhood_index:.3f} "
        f"(reduction of {hub.robinhood_index - hub_mp.robinhood_index:.3f})"
    )
