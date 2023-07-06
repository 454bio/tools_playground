#!/usr/bin/env python

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import ziontools

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog=''
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        action='store',
        dest='color_transformed_csv',
        help="color transformed .csv"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        action='store',
        type=argparse.FileType('w'),
        dest='outputcsv',
        help="output filepath for .csv file"
    )

    args = parser.parse_args()

    ziontools.dephase(
        args.color_transformed_csv,
        args.outputcsv
    )
