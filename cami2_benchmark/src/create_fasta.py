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
parser.add_argument("--log", help="Path to log file", required=True)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

# Read in FASTA files only to get its length. This way, we can avoid storing
# in memory contigs for sequences that will never get output anyway
lens: dict[str, int] = dict()
with vamb.vambtools.Reader(args.fastapath) as file:
    for record in vamb.vambtools.byte_iterfasta(
        file, None
    ):  # changed this line from vamb github code, otherwise causes error
        lens[record.identifier] = len(record)


with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)
num_clusters_before = len(clusters)

clusters = {
    cluster: contigs
    for (cluster, contigs) in clusters.items()
    if sum(lens[c] for c in contigs) >= args.minsize
}
num_clusters_after = len(clusters)

with vamb.vambtools.Reader(args.fastapath) as file:
    vamb.vambtools.write_bins(pathlib.Path(args.outdir), clusters, file, maxbins=None)


with open(args.log, "w") as log_f:
    log_f.write(f"PATH: {args.clusterspath}")
    log_f.write(f"Using minimum binsize of {args.minsize} base-pairs")
    log_f.write(f"Total clusters before filtering: {num_clusters_before}\n")
    log_f.write(f"Total clusters after filtering: {num_clusters_after}\n")
    log_f.write(
        f"Number of clusters removed: {num_clusters_before - num_clusters_after}\n\n\n"
    )

print(f"Postprocess clustering complete. Log saved to {args.log}")
