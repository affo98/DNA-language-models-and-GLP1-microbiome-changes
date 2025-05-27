import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering


def get_groups_agglomorative(
    hausdorff_matrix: np.ndarray,
    hausdorff_clusternames: np.array,
    n_clusters: int,
    distance_metric: str,
    linkage: str,
    perplexity: int,
    save_path: str,
    log,
    random_state: int = 42,
) -> np.ndarray:
    """
    Runs Agglomerative Clustering with different numbers of clusters,
    projects with t-SNE, and plots the results (legend outside).

    Parameters:
        hausdorff_matrix (np.ndarray): 2D (n x n) symmetric distance matrix.
        n_clusters_list (list[int]): Values of n_clusters to try.
        linkage (str): Linkage criterion ('average', 'complete', 'single', etc.).
        perplexity (int): Perplexity for t-SNE.
        random_state (int): Random seed for reproducibility.
    """

    try:
        agg_data = np.load(os.path.join(save_path, "agglomorative_data.npz"))
        labels = agg_data["labels"]
        n_clusters_loaded = agg_data["n_clusters"]
        hausdorff_clusternames_loaded = agg_data["hausdorff_clusternames"]
        print(hausdorff_clusternames_loaded.shape)
        print(hausdorff_clusternames.shape)

        assert len(np.unique(labels)) == n_clusters_loaded == n_clusters
        assert (
            len(labels)
            == len(hausdorff_clusternames)
            == len(hausdorff_clusternames_loaded)
        )
        assert np.array_equal(hausdorff_clusternames, hausdorff_clusternames_loaded)
        log.append(f"Loaded agglmorative clustering from {save_path}")

        return labels

    except FileNotFoundError:
        log.append(f"Agglomorative clustering not found at {save_path}, generating new")

    os.makedirs(save_path, exist_ok=True)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric=distance_metric, linkage=linkage
    )
    labels = model.fit_predict(hausdorff_matrix)

    tsne = TSNE(
        metric=distance_metric,  # or "cosine" on your raw features
        perplexity=perplexity,
        n_components=2,
        random_state=random_state,
        init="random",  # more stable than random init
    )
    embedding = tsne.fit_transform(hausdorff_matrix)

    plt.figure(figsize=(9, 9))
    for lbl in np.unique(labels):
        mask = labels == lbl
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1], label=f"Cluster {lbl}", s=30
        )

        plt.title(f"Agglomerative (n_clusters={n_clusters}, linkage={linkage})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Clusters")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"agglomorative.png"), dpi=300)

    npz_path = os.path.join(save_path, "agglomorative_data.npz")
    np.savez_compressed(
        npz_path,
        labels=labels,
        embedding=embedding,
        n_clusters=n_clusters,
        hausdorff_clusternames=hausdorff_clusternames,
    )

    return labels
