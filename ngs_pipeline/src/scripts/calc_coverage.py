import sys

import pycoverm


def calc_coverage(bams: list[str], output_file: str, threads: int = 4):
    contigs, coverage = pycoverm.get_coverages_from_bam(
        bams,
        trim_upper=0.1,
        trim_lower=0.1,
        threads=threads,
        min_identity=0.001,
    )
    with open(output_file, "w") as f:
        for contig, cov in zip(contigs, coverage):
            f.write(f"{contig}\t{cov[0]}\n")


if __name__ == "__main__":
    bams = sys.argv[1:-2]
    output_file = sys.argv[-2]
    threads = int(sys.argv[-1])
    calc_coverage(bams, output_file, threads)


# VAMB code to get rpkm normalized values and sum to 1 across samples:

# abundance *= 1_000_000 / sample_depths_sum
# total_abundance = abundance.sum(axis=1)

# Normalize abundance to sum to 1
# n_samples = abundance.shape[1]
# zero_total_abundance = total_abundance == 0
# abundance[zero_total_abundance] = 1 / n_samples
# nonzero_total_abundance = total_abundance.copy()
# nonzero_total_abundance[zero_total_abundance] = 1.0
# abundance /= nonzero_total_abundance.reshape((-1, 1))
