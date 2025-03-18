import sys
import argparse
import vamb
import pathlib

parser = argparse.ArgumentParser(
    description="""Command-line bin creator.
Will read the entire content of the FASTA file into memory - beware.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)

parser.add_argument("fastapath", help="Path to FASTA file")
parser.add_argument("clusterspath", help="Path to clusters.tsv")
parser.add_argument("minsize", help="Minimum size of bin in bp", type=int, default=0)
parser.add_argument("outdir", help="Directory to create")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()


def byte_iterfasta(filehandle):
    """Yields FastaEntries from a binary opened fasta file.

    Usage:
    >>> with Reader('/dir/fasta.fna') as filehandle:
    ...     entries = byte_iterfasta(filehandle, '/dir/fasta/fna') # a generator

    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment

    Output: Generator of FastaEntry-objects from file
    """

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    # prefix = "" if filename is None else f"In file '{filename}', "
    header = next(line_iterator, None)

    # Empty file is valid - we return from the iterator
    if header is None:
        return None
    elif not isinstance(header, bytes):
        raise TypeError(
            f"first line is not binary. Are you sure you are reading the file in binary mode?"
        )

    elif not header.startswith(b">"):
        raise ValueError(f"FASTA file is invalid, first line does not begin with '>'")


with vamb.vambtools.Reader(args.fastapath) as file:
    byte_iterfasta(file)


# Read in FASTA files only to get its length. This way, we can avoid storing
# in memory contigs for sequences that will never get output anyway
# lens: dict[str, int] = dict()
# with vamb.vambtools.Reader(args.fastapath) as file:
#     for record in vamb.vambtools.byte_iterfasta(file, args.fastapath):
#         lens[record.identifier] = len(record)

# with open(args.clusterspath) as file:
#     clusters = vamb.vambtools.read_clusters(file)

# clusters = {
#     cluster: contigs
#     for (cluster, contigs) in clusters.items()
#     if sum(lens[c] for c in contigs) >= args.minsize
# }

# with vamb.vambtools.Reader(args.fastapath) as file:
#     vamb.vambtools.write_bins(pathlib.Path(args.outdir), clusters, file, maxbins=None)
