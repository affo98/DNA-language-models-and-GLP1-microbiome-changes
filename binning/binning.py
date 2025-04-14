import os
from argparse import ArgumentParser

from time import time

from src.clustering import KMediod
from src.get_embeddings import Embedder
from src.threshold import Threshold
from src.utils import read_contigs, Logger, split_contigs_valtest

# uncomment to use hubness reducion
# from src.threshold_hubness import Threshold
# from src.clustering_hubness import KMediod

# data
MAX_CONTIG_LENGTH = 60000  # oom with 70.000
VAL_PROPORTION = 0.1

# threshold calculation
N_BINS = 1000
BLOCK_SIZE = 1000  # might need to chagne in hubness

# kmediod
MIN_BIN_SIZE = 2  # changed from 10 to 2, because bins less than MINSIZE_BINS (250000) will be removed in postprocessing.
NUM_STEPS = 3
MAX_ITER = 2000  # increased from 1000


def main(args, log):

    contigs, contig_names = read_contigs(
        args.contigs, filter_len=MAX_CONTIG_LENGTH, log=log
    )
    # contigs = contigs[0:5000]
    # contig_names = contig_names[0:5000]

    contigs_test, contigs_val, contig_names_test, contig_names_val = (
        split_contigs_valtest(
            contigs, contig_names, log, args.save_path, VAL_PROPORTION
        )
    )

    if args.mode == "val":
        log.append(
            f"{'='*60}\n"
            f"=== Start hyperparameter search for KNN ===\n"
            f"=== K: {args.knnk} ===\n"
            f"=== P: {args.knnp} ===\n"
            f"{'='*60}"
        )
        histogram_dir = os.path.join(args.save_path, "threshold_histograms")
        results_dir = os.path.join(args.save_path, "cluster_results")
        os.makedirs(histogram_dir)
        os.makedirs(results_dir)
        embedder_val = Embedder(
            contigs_val,
            contig_names_val,
            args.batch_sizes,
            args.model_name,
            args.model_path,
            os.path.join(args.save_path, "embeddings", f"{args.model_name}.npz"),
            normalize_embeddings=True,
            log=log,
        )
        embeddings_val = embedder_val.get_embeddings()

        thresholder_val = Threshold(
            embeddings_val,
            N_BINS,
            BLOCK_SIZE,
            histogram_dir,
            args.model_name,
            log,
        )

        kmediod_val = KMediod(
            embeddings_val,
            contig_names_val,
            results_dir,
            log,
            False,
            "val",
            MIN_BIN_SIZE,
            NUM_STEPS,
            MAX_ITER,
            BLOCK_SIZE,
        )

        for knnk in args.knnk:
            for knnp in args.knnp:
                log.append(f"\nRunning k:{knnk} p:{knnp}")
                threshold = thresholder_val.get_knn_threshold(knnk, knnp)
                _, _ = kmediod_val.fit(threshold, knnk, knnp)

    elif args.mode == "test":
        knnk = args.knnk[0]
        knnp = args.knnp[0]
        log.append(
            f"{'='*60}\n"
            f"=== Running Binning Test with K: {knnk} P: {knnp} ===\n"
            f"{'='*60}"
        )

        embedder_test = Embedder(
            contigs_test,
            contig_names_test,
            args.batch_sizes,
            args.model_name,
            args.model_path,
            os.path.join(args.save_path, "embeddings", f"{args.model_name}.npz"),
            normalize_embeddings=True,
            log=log,
        )
        embeddings_test = embedder_test.get_embeddings()

        thresholder_test = Threshold(
            embeddings_test,
            N_BINS,
            BLOCK_SIZE,
            args.save_path,
            args.model_name,
            log,
        )

        kmediod_test = KMediod(
            embeddings_test,
            contig_names_test,
            args.save_path,
            log,
            True,
            "test",
            MIN_BIN_SIZE,
            NUM_STEPS,
            MAX_ITER,
            BLOCK_SIZE,
        )

        threshold = thresholder_test.get_knn_threshold(knnk, knnp)
        _, _ = kmediod_test.fit(threshold, knnp, knnk)


def add_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--contigs",
        "-c",
        help="contig catalogue from multi-split approach",
    )
    parser.add_argument(
        "--model_name",
        "-mn",
        help="Name of the model to use for embedding generation",
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        help="Path to the pretrained model file or directory",
    )
    parser.add_argument(
        "--batch_sizes",
        "-b",
        nargs="+",
        type=int,
        help="batch sizes for embeddings",
    )
    parser.add_argument(
        "--knnk",
        "-k",
        nargs="+",
        type=int,
        help="List of k-values to search",
    )
    parser.add_argument(
        "--knnp",
        "-p",
        nargs="+",
        type=int,
        help="List of p-values to search",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to save the computed embeddings or to load existing ones",
    )
    parser.add_argument(
        "--log",
        "-l",
        help="Path to save logfile",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["val", "test"],
        required=True,
        help="Choose whether to run in validation ('val') or test ('test') mode.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    start_time = time()
    args = add_arguments()

    log = Logger(args.log)

    for arg, value in vars(args).items():
        log.append(f"{arg}: {value}")

    os.makedirs(args.save_path)
    main(args, log)

    end_time = time()
    elapsed_time = end_time - start_time
    log.append(f"Binning of {args.model_name} ran in {elapsed_time:.2f} Seconds")
