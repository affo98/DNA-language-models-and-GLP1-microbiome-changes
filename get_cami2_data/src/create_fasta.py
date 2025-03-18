import sys
import argparse
import vamb
import pathlib
import gzip

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


# Function to check if a file is opened in binary mode
def is_binary(file):
    try:
        # Attempt to read a single byte
        byte = file.read(1)
        # If we get a byte object, it's binary
        return isinstance(byte, bytes)
    except Exception as e:
        # If an error occurs, assume it's not binary
        print(f"Error checking file mode: {e}")
        return False


# Open the gzipped FASTA file in binary mode
with gzip.open(args.fastapath, "rb") as file:
    # Check if the file is opened in binary mode
    if is_binary(file):
        print("File is opened in binary mode.")
    else:
        print("File is not opened in binary mode.")
    # Initialize a dictionary to store sequence lengths
    lens = {}
    # Iterate over sequences in the FASTA file
    for record in vamb.vambtools.byte_iterfasta(file, args.fastapath):
        lens[record.identifier] = len(record)

# Read clusters from the clusters file
with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)

# Filter clusters based on minimum size
clusters = {
    cluster: contigs
    for cluster, contigs in clusters.items()
    if sum(lens[c] for c in contigs) >= args.minsize
}

# Write bins to the output directory
with gzip.open(args.fastapath, "rb") as file:
    vamb.vambtools.write_bins(pathlib.Path(args.outdir), clusters, file, maxbins=None)
