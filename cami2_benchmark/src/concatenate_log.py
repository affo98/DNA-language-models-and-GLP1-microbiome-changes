import argparse
import gzip
import numpy as np
from Bio import SeqIO


def compute_summary_stats(lengths):
    if not lengths:
        return 0, 0, 0, 0, 0  # Handle empty case
    return (
        np.min(lengths),
        np.max(lengths),
        np.percentile(lengths, 50),
        np.percentile(lengths, 25),
        np.percentile(lengths, 75),
    )


def get_contig_lengths(paths):
    """Get contig lengths from one or more FASTA files, handling .gz files."""
    contig_lengths = []
    for path in paths:
        if path.endswith(".gz"):  # Check if the file is gzipped
            with gzip.open(path, "rt") as handle:  # Open gzipped file in text mode
                for record in SeqIO.parse(handle, "fasta"):
                    contig_lengths.append(len(record.seq))
        else:  # For regular text files
            with open(path, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    contig_lengths.append(len(record.seq))
    return contig_lengths


def log_statistics(contig_lengths_before, logfile, minlength):
    stats_before = compute_summary_stats(contig_lengths_before)

    with open(logfile, "w") as log_f:
        log_f.write(f"Using minimum contig length of {minlength}\n")
        log_f.write(f"Total contigs before filtering: {len(contig_lengths_before)}\n")

        log_f.write(f"Contig length statistics (before filtering):\n")
        log_f.write(f"  Min length: {stats_before[0]}\n")
        log_f.write(f"  Max length: {stats_before[1]}\n")
        log_f.write(f"  Median length: {stats_before[2]:.2f}\n")
        log_f.write(f"  25th percentile: {stats_before[3]}\n")
        log_f.write(f"  75th percentile: {stats_before[4]}\n\n")

    print(f"Log saved to {logfile}")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Contig length statistics and logging script."
    )
    parser.add_argument("inpaths", help="Paths to input FASTA file(s)", nargs="+")
    parser.add_argument("outpath", help="Paths to output FASTA")
    parser.add_argument("--log", help="Path to log file")
    parser.add_argument(
        "-m",
        "--minlength",
        type=int,
        default=2000,
        help="Discard sequences below this length [2000]",
    )
    args = parser.parse_args()

    # Get contig lengths from input FASTA files
    contig_lengths_before = get_contig_lengths(args.inpaths)
    contig_lengths_after = get_contig_lengths([args.outpath])

    # Log statistics to the provided logfile
    log_statistics(contig_lengths_before, args.logfile, args.minlength)
    log_statistics(contig_lengths_after, args.logfile, args.minlength)


if __name__ == "__main__":
    main()
