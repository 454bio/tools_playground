#!/usr/bin/env python

import roifile
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import os
import argparse
from scipy import ndimage
import pathlib

from common import *

'''
Extracts pixel data for each spot.
Produces a compact csv file format, but the timestamps are slightly inaccurate, they are taken from the 645 image.
'''

# TODO, subtract 4096?

new_format = 1

def main(inputpath: str, roizipfilepath: str, outputfilename: str, max_number_of_pixel_per_spot: int, image_number: int):

    df_files = get_cycle_files(inputpath)
    print(df_files)
    print(type(df_files))


    # check if roifile exists
    if not pathlib.Path(roizipfilepath).is_file():
        print(f"Cannot open {roizipfilepath}")
        exit(-1)

    rois = roifile.ImagejRoi.fromfile(roizipfilepath)
    print(rois)

    rows_list = []

    # read image size from first image
    image0 = cv.imread(df_files.iloc[0]['filenamepath'], cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    print(f"Image shape {image0.shape}")

    # mask for one RGB channel
    mask = np.zeros(image0.shape[:2], dtype=np.uint8)
    print(f"mask: {type(mask)}, {mask.dtype}, {mask.shape}")

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

    sizes = ndimage.sum_labels(np.ones(mask.shape), mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per roi: {sizes}")

    plt.imshow(mask)
    plt.show()

    images = {}

    # filter df, use the 5 images starting from the provided image number
    df_files = df_files.loc[(df_files['file_info_nb'] >= image_number) & (df_files['file_info_nb'] < image_number+5)]

    for index, row in df_files.iterrows():

        filenamepath = row['filenamepath']
        file_info_nb = row['file_info_nb']
        file_info_wl = row['wavelength']
        file_info_cy = row['cycle']
        file_info_ts = row['timestamp']

        filename = os.path.basename(filenamepath)
        print(f"{index:4}   {filename:43}   WL:{file_info_wl:4}   CY:{file_info_cy:3}   TS:{file_info_ts:9}")


        image = cv.imread(filenamepath, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
#        image[image<4096]=4096
#        image -= 4096
        images[file_info_wl] = {'image': image, 'cycle': int(file_info_cy), 'timestamp_ms': file_info_ts}


    label_counter = [-1]*(nb_labels+1) # +1 for 0 background
    label_counter_in_subset = [-1]*(nb_labels+1) # +1 for 0 background

    ratio = (sizes // max_number_of_pixel_per_spot)+1
    print(sizes)
    print(ratio)

    for r, c in np.ndindex(mask.shape):

        label = mask[r, c]
        if label == 0:  # no spot
            continue

        label_counter[label] += 1

        if label_counter[label] % ratio[label] != 0:
            continue

        label_counter_in_subset[label] += 1

        roi_index = label - 1

        if new_format:
            dict_entry = {
                'spot': rois[roi_index].name.lstrip('spot'),
                'pixel_i': label_counter_in_subset[label],
                'timestamp_ms': images[645]['timestamp_ms'],  # slightly inaccurate
                'cycle': images[645]['cycle']  # should be correct
            }
            for wl in [365, 445, 525, 590, 645]:
                for i, color in enumerate(['R', 'G', 'B']):
                    dict_entry[color+str(wl)] = images[wl]['image'][r][c][i]

            rows_list.append(dict_entry)

        else:
            for wl in [365, 445, 525, 590, 645]:
                dict_entry = {
                    'spot': rois[roi_index].name.lstrip('spot'),
                    'pixel_i': label_counter_in_subset[label],
                    'cycle': images[wl]['cycle'],
                    'timestamp_ms': images[wl]['timestamp_ms'],
                    'WL': wl,
                }
                for i, color in enumerate(['R', 'G', 'B']):
                    dict_entry[color] = images[wl]['image'][r][c][i]

                rows_list.append(dict_entry)

    # create final dataframe
    df = pd.DataFrame(rows_list)
    df.sort_values(by=['spot', 'cycle', 'timestamp_ms'], inplace=True)
    print(f"Writing {outputfilename}")
    df.to_csv(outputfilename, index=False)
#    print(df.to_csv(index=False))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
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

    parser.add_argument(
        "-p", action='store',
        type=int,
        dest='max_number_of_pixel_per_spot',
        default=5000,
        help="Maximum number of pixel per spot in the csv file"
    )

    parser.add_argument(
        "-s", action='store',
        required=True,
        type=int,
        dest='start_645_image_number',
        help="image number of 645 image, 5 images will be used"
    )

    args = parser.parse_args()
    inputpath = args.input
    print(f"inputpath: {inputpath}")

    outputfilename = args.output
    print(f"outputfilename: {outputfilename}")

    roizipfilepath = args.roizipfilepath
    print(f"roizipfilepath: {roizipfilepath}")

    max_number_of_pixel_per_spot = args.max_number_of_pixel_per_spot
    print(f"max_number_of_pixel_per_spot: {max_number_of_pixel_per_spot}")

    image_number = args.start_645_image_number
    print(f"image_number: {image_number}")

    # main
    main(inputpath, roizipfilepath, outputfilename, max_number_of_pixel_per_spot, image_number)


