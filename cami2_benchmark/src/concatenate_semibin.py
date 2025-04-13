#!/usr/bin/env python

import sys
import os
import argparse
import gzip
from vamb.vambtools import Reader, byte_iterfasta
from typing import IO
from collections.abc import Iterable


def concatenate_fasta(
    outfile: IO[str], inpaths: Iterable[str], minlength: int = 2000, rename: bool = True
):
    """Creates a new FASTA file from input paths, and optionally rename contig headers
    to the pattern "S{sample number}C{contig identifier}".

    Inputs:
        outpath: Open filehandle for output file
        inpaths: Iterable of paths to FASTA files to read from
        minlength: Minimum contig length to keep [2000]
        rename: Rename headers

    Output: None
    """

    identifiers: set[str] = set()
    for inpathno, inpath in enumerate(inpaths):
        print(inpath)
        sample_barcode = inpath.split("/")[-1].split("_")[0]
        print(sample_barcode)
        try:
            with Reader(inpath) as infile:
                # If we rename, seq identifiers only have to be unique for each sample
                if rename:
                    identifiers.clear()

                for entry in byte_iterfasta(infile):
                    if len(entry) < minlength:
                        continue

                    if rename:
                        entry.rename(f"S{sample_barcode}C{entry.identifier}".encode())

                    if entry.identifier in identifiers:
                        raise ValueError(
                            "Multiple sequences would be given "
                            f'identifier "{entry.identifier}".'
                        )
                    identifiers.add(entry.identifier)
                    print(entry.format(), file=outfile)
        except Exception as e:
            print(f"Exception occured when parsing file {inpath}", file=sys.stderr)
            raise e from None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Creates the input FASTA file for Vamb.
    Input should be one or more FASTA files, each from a sample-specific assembly.
    If keepnames is False, resulting FASTA can be binsplit with separator 'C'.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
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
    parser.add_argument(
        "--nozip", action="store_true", help="Do not gzip output [False]"
    )

    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Check inputs
    for path in args.inpaths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    if os.path.exists(args.outpath):
        raise FileExistsError(args.outpath)

    parent = os.path.dirname(args.outpath)
    if parent != "" and not os.path.isdir(parent):
        raise NotADirectoryError(
            f'Output file cannot be created: Parent directory "{parent}" is not an existing directory'
        )

    # Run the code. Compressing DNA is easy, this is not much bigger than level 9, but
    # many times faster
    filehandle = (
        open(args.outpath, "w")
        if args.nozip
        else gzip.open(args.outpath, "wt", compresslevel=1)
    )
    concatenate_fasta(
        filehandle, args.inpaths, minlength=args.minlength, rename=(not args.keepnames)
    )
    filehandle.close()
