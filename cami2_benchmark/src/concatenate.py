#!/usr/bin/env python

import os
import argparse
import gzip
import numpy as np

import vamb
from Bio import SeqIO

parser = argparse.ArgumentParser(
    description="""Creates the input FASTA file for Vamb.
Input should be one or more FASTA files, each from a sample-specific assembly.
If keepnames is False, resulting FASTA can be binsplit with separator 'C'.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=True,
)

parser.add_argument("outpath", help="Path to output FASTA file")
parser.add_argument("inpaths", help="Paths to input FASTA file(s)", nargs="+")
parser.add_argument(
    "-m",
    dest="minlength",
    metavar="",
    type=int,
    default=2000,
    help="Discard sequences below this length [2000]",
)
parser.add_argument(
    "--keepnames", action="store_true", help="Do not rename sequences [False]"
)
parser.add_argument("--nozip", action="store_true", help="Do not gzip output [False]")
parser.add_argument("--log", help="Path to log file", required=True)


args = parser.parse_args()

# Check inputs
for path in args.inpaths:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

if os.path.exists(args.outpath):
    raise FileExistsError(args.outpath)

outpath = os.path.normpath(args.outpath)
parent = os.path.dirname(outpath)
if parent != "" and not os.path.isdir(parent):
    raise NotADirectoryError(
        f'Output file cannot be created: Parent directory "{parent}" is not an existing directory'
    )

contig_lengths_before = []
for path in args.inpaths:
    with open(path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            contig_lengths_before.append(len(record.seq))


# Run the code. Compressing DNA is easy, this is not much bigger than level 9, but
# many times faster
filehandle = (
    open(outpath, "w") if args.nozip else gzip.open(outpath, "wt", compresslevel=1)
)
try:
    vamb.vambtools.concatenate_fasta(
        filehandle, args.inpaths, minlength=args.minlength, rename=(not args.keepnames)
    )
except:
    filehandle.close()
    raise


contig_lengths_before = []
with vamb.vambtools.Reader(outpath) as file:
    for record in SeqIO.parse(file, "fasta"):
        contig_lengths_before.append(len(record.seq))


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


stats_before = compute_summary_stats(contig_lengths_before)
# stats_after = compute_summary_stats(contig_lengths_after)

# Write log file
with open(args.log, "w") as log_f:
    log_f.write(f"Using minimum contig length of {args.minlength}\n")
    log_f.write(f"Total contigs before filtering: {len(contig_lengths_before)}\n")
    # log_f.write(f"Total contigs after filtering: {len(contig_lengths_after)}\n")
    # log_f.write(
    #     f"Number of contigs removed: {len(contig_lengths_before) - len(contig_lengths_after)}\n\n"
    # )

    log_f.write(f"Contig length statistics (before filtering):\n")
    log_f.write(f"  Min length: {stats_before[0]}\n")
    log_f.write(f"  Max length: {stats_before[1]}\n")
    log_f.write(f"  Median length: {stats_before[2]:.2f}\n")
    log_f.write(f"  25th percentile: {stats_before[3]}\n")
    log_f.write(f"  75th percentile: {stats_before[4]}\n\n")

    # log_f.write(f"Contig length statistics (after filtering):\n")
    # log_f.write(f"  Min length: {stats_after[0]}\n")
    # log_f.write(f"  Max length: {stats_after[1]}\n")
    # log_f.write(f"  Median length: {stats_after[2]:.2f}\n")
    # log_f.write(f"  25th percentile: {stats_after[3]}\n")
    # log_f.write(f"  75th percentile: {stats_after[4]}\n")

print(f"Concatenation complete. Log saved to {args.log}")
