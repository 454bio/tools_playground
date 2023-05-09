#!/usr/bin/env python

import roifile
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import os
import glob
import argparse
from scipy import ndimage
import pathlib

# TODO, subtract 4096?
#S0 background, S1: S2: S3: S4:   S5 scatter

# # TODO use last few images in cycle
im1=2
im5=6

def main(inputpath: str, roizipfilepath : str, outputfilename: str):

    file_names = sorted(glob.glob(inputpath + "/*tif", recursive=False))
    print(f"Found {len(file_names)} tif files.")

    # check if roifile exists
    if not pathlib.Path(roizipfilepath).is_file():
        print(f"Cannot open {roizipfilepath}")
        exit(-1)

    rois = roifile.ImagejRoi.fromfile(roizipfilepath)
    print(rois)

    rows_list = []

    # read image size from first image
    image0 = cv.imread(file_names[2], cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    print(f"Image shape {image0.shape}")

    # mask for one RGB channel
    mask = np.zeros(image0.shape[:2], dtype=np.uint8)
    print(f"mask: {type(mask)}, {mask.dtype}, {mask.shape}")
    print(f"mask shape {mask.shape[:2]}")

    for idx, roi in enumerate(rois):
        if __debug__:
            print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)
#            print(roi)

        yc = np.uint16(roi.top + (roi.bottom - roi.top) / 2)
        xc = np.uint16(roi.left + (roi.right - roi.left) / 2)
        x_axis_length = int((roi.right - roi.left)/2.0)
        y_axis_length = int((roi.bottom - roi.top)/2.0)

        label_id = idx + 1
        mask = cv.ellipse(mask, (xc, yc), [x_axis_length, y_axis_length], angle=0, startAngle=0, endAngle=360, color=label_id, thickness=-1)

    # how many regions?
    nb_labels = len(rois)
    label_ids = np.arange(1, nb_labels + 1) # range(1, nb_labels + 1)
    print(f"labels: {nb_labels}")

    sizes = ndimage.sum_labels(mask, mask, range(nb_labels + 1))
    print(f"number of pixels per roi: {sizes}")

    plt.imshow(mask)
    plt.show()


    for idx, filename in enumerate(file_names):  # TODO use only last images in cycle

        print(f"{idx}  {os.path.basename(filename)}", end="\n")
        # extract file info
        file_info = os.path.basename(filename).rstrip(".tif").split("_")

        # small protection against additional files
        if not file_info[0].startswith('000'):
            continue

        file_info_number = file_info[0]
        #         if not int(file_info_number) in range(7,12):
        if not int(file_info_number) in range(im1,im5+1):
            continue
        file_info_wl = file_info[3]

        if '_C' in filename:
            file_info_cy = int(file_info[4].lstrip("C"))
            file_info_ts = file_info[5]
        else:
            file_info_cy = 1 #int(file_info[4].lstrip("C"))
            file_info_ts = file_info[4]

        print(f"WL:{file_info_wl}  CY:{file_info_cy}  TS:{file_info_ts}")

        image = cv.imread(filename, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
#        image[image<4096]=4096
#        image -= 4096

        # FOR each label/spot

        print(f"sizes shape {sizes.shape}")
        # pixel per roi

        # apply mean mask
        for idx, roi in enumerate(rois):
            i = 0
            for rgb in image[mask == idx + 1]:

                dict_entry = {
                    'spot': roi.name.lstrip('spot'), 'cycle': file_info_cy, 'TS': file_info_ts, 'WL': file_info_wl,
                    'i':i, 'R': rgb[0], 'G': rgb[1], 'B': rgb[2],
                }
                i += 1
                if i > 500:
                    break
#                print(dict_entry)

                rows_list.append(dict_entry)


    # create final dataframe
    df = pd.DataFrame(rows_list)
    df.sort_values(by=['spot','cycle', 'TS'], inplace=True)
    print(f"Writing {outputfilename}")
    df.to_csv(outputfilename, index=False)
#    print(df.to_csv(index=False))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Extracts pixel data',
        epilog='help'
    )

    parser.add_argument(
        "-o", "--output", action='store',
        type=argparse.FileType('w'),
        dest='output',
        default='.',
        help="output folder for spot_pixel.csv"
    )

    parser.add_argument(
        "-i", "--input", action='store',
        dest='input',
        default='.',
        help="Input folder with .tif files"
    )

    parser.add_argument(
        "-r", "--roi", action='store',
        dest='roizipfilepath',
        default='RoiSet.zip',
        help="roiset zipfile"
    )

    args = parser.parse_args()
    inputpath = args.input
    print(f"inputpath: {inputpath}")

    outputfilename = args.output
    print(f"outputfilename: {outputfilename}")

    roizipfilepath = args.roizipfilepath
    print(f"roizipfilepath: {roizipfilepath}")

    # main
    main(inputpath, roizipfilepath, outputfilename)


