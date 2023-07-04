import torch
import torchvision
import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2 as cv
import roifile
import glob
import os
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
import csv
import argparse

from common import get_cycle_files, default_base_color_map, default_spot_colors

def sort_key(file_name):
    match = re.match(r'(\d*)(.*?)\.csv$', file_name)  # Match numeric and string portions
    numeric_part = match.group(1)
    string_part = match.group(2)
    if numeric_part.isdigit():  # Check if the numeric portion is all digits
        return int(numeric_part), string_part
    return float('inf'), file_name  # Return infinity for purely alphabetical file names

def csv_to_dict(csv_file):
    result_dict = {}
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = row[0]
            value = row[1]
            
            # Check if the key is a numeric string
            key = int(key)  # Convert numeric string to integer
            
            # Check if the value is a numeric string
            if value.isdigit():
                value = int(value)  # Convert numeric string to integer
            elif '.' in value and all(char.isdigit() for char in value.replace('.', '', 1)):
                value = float(value)  # Convert numeric string to float
            
            result_dict[key] = value
    
    return result_dict

def main(inputpath, roifilepath, outputfilename):
    max_number_of_pixel_per_spot = 300
    image_number = 0
    new_format = 1
    
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
    
    print(roi_names)
    
    df_files = get_cycle_files(inputpath)
    
    if image_number > 0:
        print("use user provided 645 image")
        df_files = df_files.loc[(df_files['file_info_nb'] >= image_number) & (df_files['file_info_nb'] < image_number + 5)]
    else:
        df_files = df_files.loc[(df_files['cycle'] == 1)].tail(5)
    
    print(df_files)
    
    # read image size from first image
    image0 = cv.imread(df_files.iloc[0]['filenamepath'], cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    print(f"Image shape {image0.shape}")
    
    ref_file = roifilepath + '/ref_mask.csv'
    ref_mask = np.genfromtxt(ref_file, delimiter=',')
    
    spot_file = roifilepath + '/spot_mask.csv'
    spot_mask = np.genfromtxt(spot_file, delimiter=',')
    
    sam_file = roifilepath + '/sam_mask.csv'
    sam_mask = np.genfromtxt(sam_file, delimiter=',')
    
    mask_shape = ref_mask.shape
    print(f"mask: {type(ref_mask)}, {ref_mask.dtype}, {ref_mask.shape}")
    
    
    # Example usage
    pixel_discovery_dict_path = roifilepath + '/pixeldiscoverydict.csv'
    pixel_discovery_dict = csv_to_dict(pixel_discovery_dict_path)
    
    # how many regions?
    nb_labels = len(roi_names)
    label_ids = np.arange(1, nb_labels + 1) # range(1, nb_labels + 1)
    print(f"labels: {nb_labels}")
    
    ref_sizes = ndimage.sum_labels(np.ones(mask_shape), ref_mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per reference spot:\n {ref_sizes}")
    spot_sizes = ndimage.sum_labels(np.ones(mask_shape), spot_mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per non-reference spot:\n {spot_sizes}")
    sizes = np.array(list(map(max, ref_sizes, spot_sizes)))
    print(f"number of pixels per spot:\n {sizes}")
    
    if __debug__:
        plt.imshow(ref_mask)
        plt.show()
        plt.imshow(spot_mask)
        plt.show()
    
    images = {}
    
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
    
    label_counter = [-1]*(nb_labels+1)  # +1 for 0 background
    label_counter_in_subset = [-1]*(nb_labels+1)  # +1 for 0 background
    
    ratio = (sizes // max_number_of_pixel_per_spot)+1
    print(sizes)
    print(ratio)
    
    ref_mask = ref_mask.astype(int)
    spot_mask = spot_mask.astype(int)
    sam_mask = sam_mask.astype(int)
    
    ref_values = np.unique(ref_mask)
    spot_values = np.unique(spot_mask)
    
    spot_pixel_list = []
    
    print("Extracting roi pixel data to csv")
    
    for r, c in np.ndindex(mask_shape):    
        if sam_mask[r, c] == 0:  # no reference and no spot
            label = len(label_counter) - 1 #color pixel
        
            label_counter[label] += 1
    
            if label_counter[label] % ratio[label] != 0:
                continue
    
            label_counter_in_subset[label] += 1
    
            roi_index = label - 1 #used to index using color pixel
    
            if new_format:
                dict_entry = {
                    'spot_index': label + 1, #aka the color
                    'spot_name': 'BG',
                    'pixel_i': label_counter_in_subset[label],
                    'r': r,
                    'c': c,
                    'timestamp_ms': images[645]['timestamp_ms'],  # slightly inaccurate
                    'cycle': images[645]['cycle']  # should be correct
                }
                for wl in [365, 445, 525, 590, 645]:
                    for i, color in enumerate(['R', 'G', 'B']):
                        dict_entry[color+str(wl)] = images[wl]['image'][r][c][i]
    
                spot_pixel_list.append(dict_entry)
    
            else:
                for wl in [365, 445, 525, 590, 645]:
                    dict_entry = {
                        'spot_index': 100, #aka the color
                        'spot_name': 'BG',
                        'pixel_i': label_counter_in_subset[label],
                        'r': r,
                        'c': c,
                        'cycle': images[wl]['cycle'],
                        'timestamp_ms': images[wl]['timestamp_ms'],
                        'WL': wl,
                    }
                    for i, color in enumerate(['R', 'G', 'B']):
                        dict_entry[color] = images[wl]['image'][r][c][i]
    
                    spot_pixel_list.append(dict_entry)
        
        
        else:
            label = max(ref_mask[r, c], spot_mask[r, c]) #color pixel
        
            label_counter[label] += 1
    
            if label_counter[label] % ratio[label] != 0:
                continue
    
            label_counter_in_subset[label] += 1
    
            roi_index = label - 1 #used to index using color pixel
    
            if new_format:
                dict_entry = {
                    'spot_index': roi_index+1, #aka the color
                    'spot_name': pixel_discovery_dict.get(label),
                    'pixel_i': label_counter_in_subset[label],
                    'r': r,
                    'c': c,
                    'timestamp_ms': images[645]['timestamp_ms'],  # slightly inaccurate
                    'cycle': images[645]['cycle']  # should be correct
                }
                for wl in [365, 445, 525, 590, 645]:
                    for i, color in enumerate(['R', 'G', 'B']):
                        dict_entry[color+str(wl)] = images[wl]['image'][r][c][i]
    
                spot_pixel_list.append(dict_entry)
    
            else:
                for wl in [365, 445, 525, 590, 645]:
                    dict_entry = {
                        'spot_index': roi_index+1, #aka the color
                        'spot_name': pixel_discovery_dict.get(label),
                        'pixel_i': label_counter_in_subset[label],
                        'r': r,
                        'c': c,
                        'cycle': images[wl]['cycle'],
                        'timestamp_ms': images[wl]['timestamp_ms'],
                        'WL': wl,
                    }
                    for i, color in enumerate(['R', 'G', 'B']):
                        dict_entry[color] = images[wl]['image'][r][c][i]
    
                    spot_pixel_list.append(dict_entry)
    
    # create final dataframe
    df = pd.DataFrame(spot_pixel_list)
    df.sort_values(by=['spot_index', 'cycle', 'timestamp_ms'], inplace=True)
    print(f"Writing {outputfilename}")
    df.to_csv(outputfilename, index=False, lineterminator='\n')
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
        dest='roifilepath',
        default='RoiSet.zip',
        help="roiset zipfile"
    )

    args = parser.parse_args()
    inputpath = args.input
    print(f"inputpath: {inputpath}")

    outputfilename = args.outputcsv
    print(f"outputfilename: {outputfilename}")

    roifilepath = args.roifilepath
    print(f"roifilepath: {roifilepath}")

    # main
    main(inputpath, roifilepath, outputfilename)