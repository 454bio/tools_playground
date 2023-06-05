#!/usr/bin/env python

import os
import argparse

import sys
sys.path.insert(0, "/home/domibel/454_Bio/tools_playground/")
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
            dest='input_directory_path',
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

        args = parser.parse_args()
        input_directory_path = args.input_directory_path
        print(f"input_directory_path: {input_directory_path}")

        output_directory_path = args.output_directory_path
        print(f"output_directory_path: {output_directory_path}")
        if not os.path.exists(output_directory_path):
            print(f"ERROR: output path {output_directory_path} doesn't exist")
            exit(-1)

        roiset_file_path = args.roiset_file_path
        print(f"roiset_file_path: {roiset_file_path}")

        spot_data_filename = args.spot_data_filename
        print(f"spot_data_filename: {spot_data_filename}")


        if args.channel_subset:
            assert len(args.channel_subset) >= 2, "Please provide at least 2 channels"
            for ch in args.channel_subset:
                pattern = "^[R|G|B](\d{3})$"
                match = re.search(pattern, ch)
                if not match:
                    print(f"{ch} doesn't match format, e.g. R365")
                    exit(-1)

            channel_names = args.channel_subset
        else:
            channel_names = ['R365', 'G365', 'B365', 'R445', 'G445', 'B445', 'R525', 'G525', 'B525', 'R590', 'G590', 'B590', 'R645', 'G645', 'B645']

        spot_names_subset = None
        if args.spot_subset:
            spot_names_subset = list(set(args.spot_subset))

    else:

        input_directory_path = ''
        spot_data_filename = ''
        roiset_file_path = ''
        output_directory_path = ''
        channel_names = ['R365', 'G365', 'B365', 'R445']

    ziontools.calculate_and_apply_transformation(spot_data_filename, roiset_file_path, input_directory_path, output_directory_path, channel_names, spot_names_subset)
