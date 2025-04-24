from Bio import SeqIO
import gzip

# Define the sample IDs to include
sample_ids_to_keep = {
    "1906463",
    "1906460",
    "1906474",
    "1906451",
    "1906543",
    "1906572",
}

# Input and output file paths
input_fasta = "T2D-EW_PRJEB1786/global_contig_catalogue.fna.gz"
output_fasta = "global_contig_catalogue.fna.gz"

# Filter and write the sequences
with gzip.open(input_fasta, "rt") as in_handle, gzip.open(
    output_fasta, "wt"
) as out_handle:
    for record in SeqIO.parse(in_handle, "fasta"):
        if any(sample_id in record.id for sample_id in sample_ids_to_keep):
            SeqIO.write(record, out_handle, "fasta")

print(f"Filtered contigs written to: {output_fasta}")
