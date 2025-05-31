import argparse
import yaml
import subprocess
import shutil
import os


def get_fastqc_path():
    """Find the fastqc binary path inside the active conda environment."""
    fastqc_path = shutil.which("fastqc")  # Finds fastqc in the current environment
    if fastqc_path:
        return os.path.dirname(fastqc_path)  # Get directory
    else:
        raise FileNotFoundError("FastQC not found in the active Conda environment.")


def update_config(config_file):
    """Update the config.yaml file with the specified dataset and output directory."""

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config["PY_SCRIPTS"] = os.path.join("src", "scripts")
    config["CONDA_ENVS"] = os.path.join("src", "envs")
    config["DB"] = os.path.join("src", "databases", "bowtie2-index")
    config["FASTQC_PATH"] = get_fastqc_path()

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_snakemake(snakefile, extra_args):
    """Run the Snakemake pipeline with additional arguments."""
    command = ["snakemake", "-s", snakefile] + extra_args
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Snakemake pipeline with config parameters."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/config.yaml",
        help="Path to the config.yaml file",
    )
    parser.add_argument(
        "-s",
        "--snakefile",
        default="pipeline_download.smk",
        help="Path to the Snakemake pipeline file",
    )
    parser.add_argument(
        "--snakemake-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for Snakemake",
    )

    args = parser.parse_args()

    update_config(args.config)

    snakemake_args = args.snakemake_args if args.snakemake_args else []
    run_snakemake(args.snakefile, snakemake_args)


if __name__ == "__main__":
    main()
