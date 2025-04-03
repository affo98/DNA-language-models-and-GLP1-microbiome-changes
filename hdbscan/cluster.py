import numpy as np

# from sklearn.cluster import HDBSCAN
from cuml.cluster import HDBSCAN
import gzip
import os
from Bio import SeqIO
import sys
import time


def cluster(path_to_embeds: str) -> np.array:
    dnabert_metahit_embeds = np.load(path_to_embeds)

    min_cluster_size = 20
    with open("hdbscan_log.txt", "w") as f:
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, verbose=True)
        f.write(f"TYPE of hdbscan: {type(hdb)}\n")
        start = time.time()
        fmt = time.gmtime(start)
        current_time = time.strftime("%D %T", fmt)
        f.write(f"STARTED AT {current_time}\n")
        hdb.fit(dnabert_metahit_embeds)
        end = time.time()
        elapsed_time = end - start
        cluster_labels = hdb.labels_

        num_clusters = len(np.unique(cluster_labels).tolist())
        num_noicy_contigs = (cluster_labels == -1).sum()
        unassigned_contigs = (cluster_labels < 0).sum()

        f.write("#" * 100 + "\n" * 2)
        f.write("#" * 30 + "\t" * 2 + "HDBSCAN PARAMETERS" + "\t" * 2 + "#" * 30)
        f.write("\n" * 2)
        f.write(f"\t\tmin_cluster_size: {min_cluster_size}")
        f.write("\n" * 2)
        f.write("#" * 100 + "\n")

        f.write(f"Number of Clusters: {num_clusters},\n")
        f.write(f"Noicy Contigs i.e. -1: {num_noicy_contigs},\n")
        f.write(f"Numbr of unassigned Contigs: {unassigned_contigs},\n")
        f.write("\n" * 2)
        f.write(f"Elapsed time fitting HDBSCAN: {elapsed_time:.2f} seconds\n")
        # TODO filter out -1 clusters
    return cluster_labels


def get_contig_names(path_to_contig_catalogue: str) -> list:
    contigs, contig_names, total_contigs = [], [], 0

    with gzip.open(path_to_contig_catalogue, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            total_contigs += 1
            if len(record.seq) < 60000:
                contigs.append(str(record.seq))
                contig_names.append(str(record.id))
    return contig_names


def save_output(contig_names, predictions) -> None:
    """save predictions in save_path in format: clustername \\t contigname"""

    output_file = os.path.join("clusters.tsv")
    with open(output_file, "w") as file:
        file.write("clustername\tcontigname\n")  # header

        for cluster, contig in zip(predictions, contig_names):
            file.write(f"{cluster}\t{contig}\n")


if __name__ == "__main__":
    embeddings_path = sys.argv[1]
    contig_catalogue_path = sys.argv[2]

    clusters = cluster(embeddings_path)
    contig_names = get_contig_names(contig_catalogue_path)

    save_output(contig_names, clusters)
