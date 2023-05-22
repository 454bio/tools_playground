#!/usr/bin/env python

import roifile
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import os
import argparse
from scipy import ndimage

from common import *

def main(inputpath: str, roizipfilepath : str, outputfilename: str):

    df_files = get_cycle_files(inputpath)
    print(type(df_files))

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
    print(f"number of pixels per label: {sizes}")

    plt.imshow(mask)
    plt.show()


    for index, row in df_files.iterrows():

        filenamepath = row['filenamepath']
        file_info_nb = row['file_info_nb']
        file_info_wl = row['wavelength']
        file_info_cy = row['cycle']
        file_info_ts = row['timestamp']

        filename = os.path.basename(filenamepath)
        print(f"{index:4}   {filename:43}   WL:{file_info_wl:4}   CY:{file_info_cy:3}   TS:{file_info_ts:9}")

        image = cv.imread(filenamepath, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data


        # apply mean mask

        R_mean = ndimage.labeled_comprehension(image[:,:,0], mask, label_ids, np.mean, int, 0)
        G_mean = ndimage.labeled_comprehension(image[:,:,1], mask, label_ids, np.mean, int, 0)
        B_mean = ndimage.labeled_comprehension(image[:,:,2], mask, label_ids, np.mean, int, 0)

        R_min = ndimage.labeled_comprehension(image[:,:,0], mask, label_ids, np.min, int, 0)
        G_min = ndimage.labeled_comprehension(image[:,:,1], mask, label_ids, np.min, int, 0)
        B_min = ndimage.labeled_comprehension(image[:,:,2], mask, label_ids, np.min, int, 0)

        R_max = ndimage.labeled_comprehension(image[:,:,0], mask, label_ids, np.max, int, 0)
        G_max = ndimage.labeled_comprehension(image[:,:,1], mask, label_ids, np.max, int, 0)
        B_max = ndimage.labeled_comprehension(image[:,:,2], mask, label_ids, np.max, int, 0)

        R_std = ndimage.labeled_comprehension(image[:,:,0], mask, label_ids, np.std, int, 0)
        G_std = ndimage.labeled_comprehension(image[:,:,1], mask, label_ids, np.std, int, 0)
        B_std = ndimage.labeled_comprehension(image[:,:,2], mask, label_ids, np.std, int, 0)

        for i, roi in enumerate(rois):

#            if __debug__:
#                print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)

            dict_entry = {
                'image_number': file_info_nb,
                'spot': roi.name.lstrip('spot'), 'cycle': file_info_cy, 'WL': file_info_wl, 'TS': file_info_ts,
                'Ravg': R_mean[i], 'Gavg': G_mean[i], 'Bavg': B_mean[i],
                'Rmin': R_min[i], 'Gmin': G_min[i], 'Bmin': B_min[i],
                'Rmax': R_max[i], 'Gmax': G_max[i], 'Bmax': B_max[i],
                'Rstd': R_std[i], 'Gstd': G_std[i], 'Bstd': B_std[i],
            }
            rows_list.append(dict_entry)

    # create final dataframe
    df = pd.DataFrame(rows_list)
    df.sort_values(by=['spot','cycle', 'TS'], inplace=True)
    print(f"Writing {outputfilename}")
    df.to_csv(outputfilename, index=False, line_terminator='\n')
#    print(df.to_csv(index=False))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help'
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
#        required=True,
        action='store',
        dest='input',
        default='.',
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
    inputpath = args.input
    print(f"inputpath: {inputpath}")

    outputfilename = args.outputcsv
    print(f"outputfilename: {outputfilename}")

    roizipfilepath = args.roizipfilepath
    print(f"roizipfilepath: {roizipfilepath}")

    # main
    main(inputpath, roizipfilepath, outputfilename)

    # test
#    inputpath = "/home/domibel/454_Bio/runs/20230419_2037_S0079_0001/raws"
#    roizipfilepath = "/tmp/RoiSet.zip"
#    outputfilename = "/tmp/test.csv"
#    main(inputpath, roizipfilepath, outputfilename)

