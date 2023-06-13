#!/usr/bin/env python

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import ziontools

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extracts pixel data of ONE cycle',
        epilog='help'
    )

    parser.add_argument(
        "-i", "--input", action='store',
        required=True,
        dest='input_raw_path',
        help="Input folder with .tif files"
    )

    parser.add_argument(
        "-r", "--roi", action='store',
        required=True,
        dest='roiset_zip_filename',
        help="roiset zip filename"
    )

    parser.add_argument(
        "-o", "--output", action='store',
        type=argparse.FileType('w'),
        required=True,
        dest='output_csv_filename',
        help="output filename e.g. /tmp/spot_pixel_data.csv"
    )

    parser.add_argument(
        "-n", action='store',
        type=int,
        dest='max_number_of_pixel_per_spot',
        default=200,
        help="Maximum number of pixel per spot in the csv file"
    )

    parser.add_argument(
        "-e", action='store',
        type=int,
        default=0,  # all
        dest='start_645_image_number',
        help="Start image number of 5 .tif image block (645, 590, 525, 445, 365) \ne.g.: -e 7  (...0007_645_C001...tif)"
    )

    args = parser.parse_args()

    ziontools.extract_roiset_pixel_data(
        args.input_raw_path,
        args.roiset_zip_filename,
        args.output_csv_filename,
        args.max_number_of_pixel_per_spot,
        args.start_645_image_number
    )
