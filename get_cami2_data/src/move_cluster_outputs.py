import argparse
import os
import shutil
from pathlib import Path

# Define a mapping of input directories to specific filenames
FILE_MAP = {
    "vamb_output": "vae_clusters_unsplit.tsv",
    "semibin_output": "some_other_clusters.tsv",
    "comebin_output": "model_clusters.tsv",
    # Add more directories and filenames as needed
}


def move_cluster_outputs(input_dirs, output_dir):
    """Move specified files from multiple input directories to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for input_dir in input_dirs:
        input_model = Path(input_dir).name
        if input_model in FILE_MAP:
            filename = FILE_MAP[input_model]
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(output_dir)

            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} -> {dest_path}")
            else:
                print(f"Warning: {src_path} does not exist.")
        else:
            print(f"Warning: No filename mapping for input directory {input_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Move cluster output files to a specified directory."
    )
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "input_dirs", type=str, nargs="+", help="List of input directories."
    )

    args = parser.parse_args()
    print(f"Moving cluster outputs:\n")
    print(f"Input: {args.input_dirs}")
    print(f"Output: {args.output_dir}")

    move_cluster_outputs(args.input_dirs, args.output_dir)


if __name__ == "__main__":

    main()
