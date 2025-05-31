import sys
import json


def rename_contigs(
    path_to_input_contigs: str, path_to_output_contigs: str, path_to_contigs_lu: str
):
    """
    Renames FASTA headers to integers and stores the lookup in a JSON file.

    Args:
        path_to_contigs: Path to input FASTA file.
        path_to_output: Path to write modified FASTA file.
        path_to_json: Path to write JSON lookup table.
    """

    lookup = {}
    counter = 0

    with open(path_to_input_contigs, "r") as infile, open(
        path_to_output_contigs, "w"
    ) as outfile:
        for line in infile:
            if line.startswith(">"):
                header = line.strip()
                new_header = f">{counter}"
                lookup[new_header[1:]] = header[1:]
                outfile.write(new_header + "\n")
                counter += 1
            else:
                outfile.write(line)

    with open(path_to_contigs_lu, "w") as json_file:
        json.dump(lookup, json_file, indent=2)

    return


if __name__ == "__main__":

    path_to_input_contigs = sys.argv[1]
    path_to_output_contigs = sys.argv[2]
    path_to_contigs_lu = sys.argv[3]

    rename_contigs(path_to_input_contigs, path_to_output_contigs, path_to_contigs_lu)
