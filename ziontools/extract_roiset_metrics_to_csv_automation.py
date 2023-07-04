import torch
import torchvision
import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import roifile
import glob
import os
import re
import pandas as pd
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from sklearn import linear_model
import argparse

from common import get_cycle_files, default_base_color_map, default_spot_colors

def main(inputpath: str, roifilepath : str, outputfilename: str):
    print("Extracting ROI names")
    analysis = os.listdir(roifilepath)
    
    roi_csvs = os.listdir(roifilepath + '/rois')
    roi_csvs = sorted(roi_csvs, key=sort_key)
    
    roi_names = []
    
    for roi_csv in roi_csvs:
        curr_csv_path = os.path.join((roifilepath + '/rois'), roi_csv)
        file_name = os.path.splitext(roi_csv)[0]  # Extract file name without extension
        if file_name.isdigit():  # Check if the file name is all digits
            file_name = int(file_name)  # Convert the numeric file name to an integer
        roi_names.append(file_name)
        
    print("Extracting sam_mask")
    sam_file = roifilepath + '/sam_mask.csv'
    sam_mask = np.genfromtxt(sam_file, delimiter=',')
    
    nb_labels = len(roi_names)
    label_ids = np.arange(1, nb_labels + 1) # range(1, nb_labels + 1)
    print(f"labels: {nb_labels}")
    
    sizes = ndimage.sum_labels(np.ones(sam_mask.shape), sam_mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per label: {sizes}")
    
    df_files = get_cycle_files(inputpath)
    
    print("Extracting ROI metrics to CSV")
    for index, row in df_files.iterrows():
        filenamepath = row['filenamepath']
        file_info_nb = row['file_info_nb']
        file_info_wl = row['wavelength']
        file_info_cy = row['cycle']
        file_info_ts = row['timestamp']
    
        filename = os.path.basename(filenamepath)
        image = cv2.imread(filenamepath, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    
        R_mean = ndimage.labeled_comprehension(image[:,:,0], sam_mask, label_ids, np.mean, int, 0)
        G_mean = ndimage.labeled_comprehension(image[:,:,1], sam_mask, label_ids, np.mean, int, 0)
        B_mean = ndimage.labeled_comprehension(image[:,:,2], sam_mask, label_ids, np.mean, int, 0)
    
        R_min = ndimage.labeled_comprehension(image[:,:,0], sam_mask, label_ids, np.min, int, 0)
        G_min = ndimage.labeled_comprehension(image[:,:,1], sam_mask, label_ids, np.min, int, 0)
        B_min = ndimage.labeled_comprehension(image[:,:,2], sam_mask, label_ids, np.min, int, 0)
    
        R_max = ndimage.labeled_comprehension(image[:,:,0], sam_mask, label_ids, np.max, int, 0)
        G_max = ndimage.labeled_comprehension(image[:,:,1], sam_mask, label_ids, np.max, int, 0)
        B_max = ndimage.labeled_comprehension(image[:,:,2], sam_mask, label_ids, np.max, int, 0)
    
        R_std = ndimage.labeled_comprehension(image[:,:,0], sam_mask, label_ids, np.std, int, 0)
        G_std = ndimage.labeled_comprehension(image[:,:,1], sam_mask, label_ids, np.std, int, 0)
        B_std = ndimage.labeled_comprehension(image[:,:,2], sam_mask, label_ids, np.std, int, 0)
    
        rows_list = []
        
        for i, roi in enumerate(roi_names):
            dict_entry = {
                'image_number': file_info_nb,
                'spot': roi, 'cycle': file_info_cy, 'WL': file_info_wl, 'TS': file_info_ts,
                'Ravg': R_mean[i], 'Gavg': G_mean[i], 'Bavg': B_mean[i],
                'Rmin': R_min[i], 'Gmin': G_min[i], 'Bmin': B_min[i],
                'Rmax': R_max[i], 'Gmax': G_max[i], 'Bmax': B_max[i],
                'Rstd': R_std[i], 'Gstd': G_std[i], 'Bstd': B_std[i],
            }
            rows_list.append(dict_entry)
    
    df = pd.DataFrame(rows_list)
    df.sort_values(by=['spot','cycle', 'TS'], inplace=True)
    print(f"Writing {outputfilename}")
    df.to_csv(outputfilename, index=False, lineterminator='\n')
        
def sort_key(file_name):
    match = re.match(r'(\d*)(.*?)\.csv$', file_name)  # Match numeric and string portions
    numeric_part = match.group(1)
    string_part = match.group(2)
    if numeric_part.isdigit():  # Check if the numeric portion is all digits
        return int(numeric_part), string_part
    return float('inf'), file_name  # Return infinity for purely alphabetical file names

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