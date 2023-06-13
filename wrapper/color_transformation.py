#!/usr/bin/env python

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import argparse
import ziontools

if __name__ == '__main__':

    test = 0

    if not test:
        parser = argparse.ArgumentParser(
            description='Color Transformation',
            epilog=''
        )

        parser.add_argument(
            "-i", "--input",
            required=True,
            action='store',
            dest='input_raw_path',
            help="Directory with .tif files, e.g.: /tmp/S001/raws/"
        )

        parser.add_argument(
            "-p", "--spot-pixel-data",
            required=True,
            action='store',
            dest='spot_data_filename',
            help="Spot pixel data file (with at least: C G A T BG), e.g.: /tmp/spot_pixel_data.csv"
        )

        parser.add_argument(
            "-r", "--roiset",
            required=True,
            action='store',
            dest='roiset_file_path',
            help="ImageJ RoiSet file, e.g.: /tmp/RoiSet.zip"
        )

        parser.add_argument(
            "-c", "--channel_subset",
            action='store',
            type=str,
            nargs='+',
            dest='channel_subset',
            help="Channel subset e.g. -c G445 G525 R590 B445, Default: all 15 channels"
        )

        parser.add_argument(
            "-s", "--spot_subset",
            action='store',
            type=str,
            nargs='+',
            dest='spot_subset',
            help="Spot subset e.g. -s C G A T BG, Default: all spots in the RoiSet"
        )

        # Output
        parser.add_argument(
            "-o", "--output",
            required=True,
            action='store',
            dest='output_directory_path',
            help="Output directory for .png and .csv files, e.g.: /tmp/S001/analysis/"
        )

        parser.add_argument(
            '-g', '--graph',
            action='store_true',
            dest='show_graph',
            default=False,
            help="plot into browser"
        )

        args = parser.parse_args()

        input_raw_path = args.input_raw_path
        print(f"input_raw_path: {input_raw_path}")

        output_directory_path = args.output_directory_path
        print(f"output_directory_path: {output_directory_path}")
        if not os.path.exists(output_directory_path):
            print(f"ERROR: output path {output_directory_path} doesn't exist")
            exit(-1)

        roiset_file_path = args.roiset_file_path
        print(f"roiset_file_path: {roiset_file_path}")

        spot_data_filename = args.spot_data_filename
        print(f"spot_data_filename: {spot_data_filename}")


        spot_names_subset = None
        if args.spot_subset:
            spot_names_subset = list(set(args.spot_subset))

    else:
        input_directory_path = ''
        spot_data_filename = ''
        roiset_file_path = ''
        output_directory_path = ''
        spot_names_subset = ['R365', 'G365', 'B365', 'R445']

    #
    fig = ziontools.calculate_and_apply_transformation(
        spot_data_filename,
        roiset_file_path,
        input_raw_path,
        output_directory_path,
        args.channel_subset,
        spot_names_subset
    )

    fig.write_image(os.path.join(output_directory_path, "bar.png"), scale=1.5)

    if args.show_graph:
        fig.show()

