#!/usr/bin/env python

import argparse
import ziontools

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog=''
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        action='store',
        type=argparse.FileType('w'),
        dest='outputcsv',
#        default='roiset_means.csv',
        help="output filepath for .csv file"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        action='store',
        dest='input_raw_path',
        help="Input folder with .tif files"
    )

    parser.add_argument(
        "-r", "--roi",
        required=True,
        action='store',
        dest='roizipfilepath',
        default='RoiSet.zip',
        help="roiset zipfile"
    )

    args = parser.parse_args()
    input_raw_path = args.input_raw_path
    print(f"input_raw_path: {input_raw_path}")

    outputfilename = args.outputcsv
    print(f"outputfilename: {outputfilename}")

    roizipfilepath = args.roizipfilepath
    print(f"roizipfilepath: {roizipfilepath}")

    ziontools.extract_roiset_metrics(
        input_raw_path,
        roizipfilepath,
        outputfilename
    )

    # test
#    inputpath = "/home/domibel/454_Bio/runs/20230419_2037_S0079_0001/raws"
#    roizipfilepath = "/tmp/RoiSet.zip"
#    outputfilename = "/tmp/test.csv"
#    extract_roiset_metrics(inputpath, roizipfilepath, outputfilename)

