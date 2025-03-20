import argparse
import os
import shutil

from Bio import SeqIO


def postprocess_comebin(input_dir, output_dir, min_total_length, log_file):
    """
    Filters FASTA files in the input directory, retaining only those where the sum of all sequence lengths
    exceeds min_total_length. Writes the filtered files to the output directory with .fna extension.

    Parameters:
    - input_dir (str): Path to the directory containing input FASTA files.
    - output_dir (str): Path to the directory where filtered FASTA files will be saved.
    - min_total_length (int): Minimum total length of sequences required to retain a file. Default is 250,000.
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # overwrite
    os.makedirs(output_dir)

    num_clusters_before = 0
    num_clusters_after = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            input_file_path = os.path.join(input_dir, filename)
            total_length = 0
            num_sequences = 0

            # Read sequences and calculate total length
            with open(input_file_path, "r") as file:
                for record in SeqIO.parse(file, "fasta"):
                    total_length += len(record.seq)
                    num_sequences += 1

            num_clusters_before += 1

            # If the total length exceeds the threshold, write to the output directory
            if total_length > min_total_length:
                num_clusters_after += 1
                output_filename = os.path.splitext(filename)[0] + ".fna"
                output_file_path = os.path.join(output_dir, output_filename)
                with open(input_file_path, "r") as infile, open(
                    output_file_path, "w"
                ) as outfile:
                    for record in SeqIO.parse(infile, "fasta"):
                        SeqIO.write(record, outfile, "fasta")

        with open(log_file, "w") as log_f:
            log_f.write(f"Using minimum binsize of {min_total_length} base-pairs")
            log_f.write(f"Total clusters before filtering: {num_clusters_before}\n")
            log_f.write(f"Total clusters after filtering: {num_clusters_after}\n")
            log_f.write(
                f"Number of clusters removed: {num_clusters_before - num_clusters_after}\n"
            )

    return


def main():
    parser = argparse.ArgumentParser(
        description="Move cluster output files to a specified directory."
    )
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("input_dirs", type=str, help="List of input directories.")
    parser.add_argument(
        "minsize_bins", help="Minimum size of bin in bp", type=int, default=0
    )
    parser.add_argument("--log", help="Path to log file", required=True)

    args = parser.parse_args()
    print(f"Postprocessing comebin cluster outputs:\n")
    print(f"Input: {args.input_dirs}")
    print(f"Output: {args.output_dir}")
    print(f"Minsize Bins: {args.minsize_bins}")
    print(f"Log File: {args.log}")

    postprocess_comebin(args.input_dirs, args.output_dir, args.minsize_bins, args.log)


if __name__ == "__main__":

    main()
