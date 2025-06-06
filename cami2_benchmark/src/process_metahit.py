"""Removes contigs shorter than minlength in metahit"""

import argparse
import gzip
import re
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


def sanitize_fasta_header(header: str) -> str:
    """Sanitizes a FASTA header by replacing [ and ] with |"""
    sanitized = re.sub(r"\[", "|", header)  # Replace '[' with '|'
    sanitized = re.sub(r"\]", "|", sanitized)  # Replace ']' with '|'
    return sanitized


def process_contigs(
    inpath: str, outpath: str, minlength: int
) -> tuple[list[int], list[int]]:
    """
    Reads contigs from 'inpath', calculates their lengths, filters out contigs
    shorter than minlength, overwrites the file with the filtered contigs, and returns
    both the original and filtered contig lengths.

    Args:
        inpath (str): Path to the input FASTA file (can be gzipped).
        minlength (int): Minimum length a contig must have to be kept.
        log: Logger object for logging messages.

    Returns:
        tuple: (original_lengths, filtered_lengths)
            original_lengths: List[int] of contig lengths before filtering.
            filtered_lengths: List[int] of contig lengths after filtering.
    """

    open_func = gzip.open if inpath.endswith(".gz") else open

    with open_func(inpath, "rt") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    for record in records:
        record.id = sanitize_fasta_header(record.id)
        record.description = sanitize_fasta_header(record.description)

    contig_lengths_before = [len(record.seq) for record in records]

    filtered_records = [record for record in records if len(record.seq) >= minlength]
    contig_lengths_after = [len(record.seq) for record in filtered_records]

    print(f"Writing processed contigs:")
    out_mode = "wt" if inpath.endswith(".gz") else "w"
    with open_func(outpath, out_mode) as out_handle:
        SeqIO.write(filtered_records, out_handle, "fasta")

    return contig_lengths_before, contig_lengths_after


def main():
    parser = argparse.ArgumentParser(
        description="Contig length statistics and logging script."
    )
    parser.add_argument("-i", "--inpath", help="Path to input FASTA")
    parser.add_argument("-o", "--outpath", help="Path to output FASTA")
    parser.add_argument(
        "--log", help="Path to log file", required=True
    )  # Fixed the logfile argument
    parser.add_argument(
        "-m",
        "--minlength",
        type=int,
        default=2000,
        help="Discard sequences below this length [2000]",
    )
    args = parser.parse_args()

    contig_lengths_before, contig_lengths_after = process_contigs(
        args.inpath, args.outpath, args.minlength
    )

    stats_before = compute_summary_stats(contig_lengths_before)
    stats_after = compute_summary_stats(contig_lengths_after)

    # Write the statistics to the log file
    with open(args.log, "w") as log_f:
        log_f.write(f"Using minimum contig length of {args.minlength}\n")
        log_f.write(f"Total contigs before filtering: {len(contig_lengths_before)}\n")
        log_f.write(f"Total contigs after filtering: {len(contig_lengths_after)}\n")

        # Before statistics
        log_f.write(f"Contig length statistics (before filtering):\n")
        log_f.write(f"  Min length: {stats_before[0]}\n")
        log_f.write(f"  Max length: {stats_before[1]}\n")
        log_f.write(f"  Median length: {stats_before[2]:.2f}\n")
        log_f.write(f"  25th percentile: {stats_before[3]}\n")
        log_f.write(f"  75th percentile: {stats_before[4]}\n\n")

        # After statistics
        log_f.write(f"Contig length statistics (after filtering):\n")
        log_f.write(f"  Min length: {stats_after[0]}\n")
        log_f.write(f"  Max length: {stats_after[1]}\n")
        log_f.write(f"  Median length: {stats_after[2]:.2f}\n")
        log_f.write(f"  25th percentile: {stats_after[3]}\n")
        log_f.write(f"  75th percentile: {stats_after[4]}\n\n")


if __name__ == "__main__":
    main()
