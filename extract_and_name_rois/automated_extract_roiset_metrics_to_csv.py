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
import pandas as pd
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from sklearn import linear_model

from common import get_cycle_files, default_base_color_map, default_spot_colors


# #### Find and display the first 365 image, from which the spots will be extracted
inputpath = '/Users/akshitapanigrahi/Documents/data/1458/raws'
pattern = '*.tif'
file_path_pattern = inputpath + '/' + pattern
filenames = glob.glob(file_path_pattern)


if len(filenames) < 1:
    filenames = glob.glob('*.png')
    
filenames = sorted(filenames)

num = len(filenames)
spot_file = -1

for i in range(num):
    if spot_file == -1 and '_365_' in filenames[i]:
        spot_file = i

num_spots = 0
spot_image = cv2.imread(filenames[spot_file])

image = cv2.imread(filenames[spot_file])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('off')
plt.show()


# #### Load and run SAM model, find all masks
sys.path.append("/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#remove background, sort by area, remove nests
def process_masks(masks):
    #remove background, sort by area, remove nests
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    masks = masks[1:]
    
    points = []
    for mask in masks:
        x = mask['point_coords'][0][0]
        y = mask['point_coords'][0][1]
        points.append((x, y))
    
    valid_points = []
    valid_masks = []
    
    for i in range(len(masks)):
        if points[i] not in valid_points:
            valid_points.append(points[i])
            valid_masks.append(masks[i])
    
    return valid_masks    

def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
   
    for ann in anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = [1, 1, 1, 1]
        #img[m] = color_mask
    ax.imshow(img)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_overlap_ratio=0.9,
    crop_n_points_downscale_factor=4,
    min_mask_region_area= 15000,  # Requires open-cv to run post-processing
    output_mode = "binary_mask",
)


masks2 = mask_generator_2.generate(image)
new_masks = process_masks(masks2)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(new_masks)
plt.axis('off')
plt.show() 


# #### Convert each mask to ROI array representation, overlay all masks to form overall ROI image array
#Map pixel value to that ROI's mask array, get sam mask -> returns sam mask and pixel mappings to their indivdual mask array
def get_mask_arr_dict_and_sam_mask(masks):
    pixel_mask_dict = {}
                                   
    mask = np.empty((1520, 2028))
    to_sum = []
                                   
    for i, mask in enumerate(masks):
        pixel_mask_dict[i + 1] = mask['segmentation'].astype(int) * (i + 1)
        to_sum.append(pixel_mask_dict[i + 1])
    
    mask = np.sum(to_sum, axis = 0)
                                   
    return mask, pixel_mask_dict

sam_mask, pixel_mask_dict = get_mask_arr_dict_and_sam_mask(new_masks)


# #### Assign ROIs natural numerical ordering

#Get leftmost coordinate of a mask array
def find_roi_leftmost_coord(roi_mask_arr):
    rows, cols = roi_mask_arr.shape
    for i in range(cols):
        for j in range(rows):
            curr = roi_mask_arr[j, i]
            if (curr != 0):
                return (i, j) 

#Define dictionary of pixel value and its left most coordinate
masks_leftmost_mapping = {}

for pixel_val in pixel_mask_dict.keys():
    masks_leftmost_mapping[pixel_val] = find_roi_leftmost_coord(pixel_mask_dict.get(pixel_val))


discovered_rois = [] #list of discovered rois
discovered_dict = {} #list of pixel value and its coord of discovery

# Get the dimensions of the array
rows, cols = sam_mask.shape
to_add_curr_row = []

for i in range(rows):
    for j in range(cols):
        curr = sam_mask[i, j]
        if curr != 0 and not(curr in to_add_curr_row) and not(curr in discovered_rois):
            to_add_curr_row.append(curr)
            discovered_dict[curr] = (j, i)
    discovered_rois.extend(to_add_curr_row)
    #print(to_add_curr_row)
    to_add_curr_row = []

pixel_discovery_order_dict = {}

for i, pixel in enumerate(discovered_rois):
    pixel_discovery_order_dict[pixel] = i + 1

def get_ycoords(pixel):
    mask_arr = pixel_mask_dict.get(pixel)
    
    y_coords = []
    
    rows, cols = mask_arr.shape

    for i in range(rows):
        for j in range(cols):
            curr = mask_arr[i, j]
            if curr != 0 and not (i in y_coords):
                y_coords.append(i)
    
    return y_coords

def check_if_two_overlap_row(p_1, p_2):
    y_1 = get_ycoords(p_1)
    y_2 = get_ycoords(p_2)
    
    set_1 = set(y_1)
    set_2 = set(y_2)
    
    if len(set_1.intersection(set_2)) > 0:
        return True
    
    return False

undiscovered_rois = discovered_rois[:]  # Create a shallow copy
ordered_discovered_rois = []

for i in discovered_rois:
    curr_overlap = []
    sorted_curr = []
    
    for j in undiscovered_rois:
        if check_if_two_overlap_row(i, j):
            curr_overlap.append(j)
    
    sorted_curr = sorted(curr_overlap, key=lambda k: masks_leftmost_mapping[k][0])
    ordered_discovered_rois.extend(sorted_curr)
    undiscovered_rois = [ele for ele in undiscovered_rois if ele not in sorted_curr]
    curr_overlap = []

for i, pixel in enumerate(ordered_discovered_rois):
    pixel_discovery_order_dict[pixel] = (i + 1)


# #### Find base spots

#Get first column of spots

spot_A = ordered_discovered_rois[0]

def get_xcoords(pixel):
    mask_arr = pixel_mask_dict.get(pixel)
    
    x_coords = []
    
    rows, cols = mask_arr.shape

    for i in range(rows):
        for j in range(cols):
            curr = mask_arr[i, j]
            if curr != 0 and not (i in x_coords):
                x_coords.append(j)
    
    return x_coords

def check_if_two_overlap_col(p_1, p_2):
    x_1 = get_xcoords(p_1)
    x_2 = get_xcoords(p_2)
    
    set_1 = set(x_1)
    set_2 = set(x_2)
    
    if len(set_1.intersection(set_2)) > 0:
        return True
    
    return False

base_spots = []

for i in ordered_discovered_rois:
    if check_if_two_overlap_col(spot_A, i):
        base_spots.append(i)

#Update names
bases = ['A', 'C', 'T', 'G']
for i, base in enumerate(base_spots):
    pixel_discovery_order_dict[base] = bases[i]


# #### Order and name remaining spots
undiscovered_rois = ordered_discovered_rois[:]
undiscovered_rois = [ele for ele in ordered_discovered_rois if ele not in base_spots]

for i, spot in enumerate(undiscovered_rois):
    pixel_discovery_order_dict[spot] = i + 1

roi_ordering = np.zeros(image.shape[:2], dtype=np.uint8)
for i, val in enumerate(discovered_rois):
    text = str(pixel_discovery_order_dict.get(val))
    loc = masks_leftmost_mapping[val]
    roi_ordering = cv2.putText(roi_ordering, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)

plt.figure(figsize=(10,10))
plt.imshow(sam_mask)
plt.imshow(roi_ordering, alpha = 0.75)
plt.show()

# # Extract ROI set metrics to csv

outputfilename = '/Users/akshitapanigrahi/Desktop/roiset_metrics_automated.csv'
df_files = get_cycle_files(inputpath)

# how many regions?
nb_labels = len(ordered_discovered_rois)
label_ids = np.arange(1, nb_labels + 1) # range(1, nb_labels + 1)
print(f"labels: {nb_labels}")

sizes = ndimage.sum_labels(np.ones(sam_mask.shape), sam_mask, range(nb_labels + 1)).astype(int)
print(f"number of pixels per label: {sizes}")

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
    
    for i, roi in enumerate(ordered_discovered_rois):
        dict_entry = {
            'image_number': file_info_nb,
            'spot': pixel_discovery_order_dict.get(roi), 'cycle': file_info_cy, 'WL': file_info_wl, 'TS': file_info_ts,
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


# ## Extract ROI set pixel data
output_csv_filename = '/Users/akshitapanigrahi/Desktop/spot_pixel_data_automated.csv'

max_number_of_pixel_per_spot = 300
image_number = spot_file

# read image size from first image
image0 = cv2.imread(df_files.iloc[0]['filenamepath'], cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
print(f"Image shape {image0.shape}")

# mask for one RGB channel
ref_mask = np.zeros(image0.shape[:2], dtype=np.uint8)
spot_mask = np.zeros(image0.shape[:2], dtype=np.uint8)

mask_shape = ref_mask.shape
print(f"mask: {type(ref_mask)}, {ref_mask.dtype}, {ref_mask.shape}")

ref_mask_to_sum = []
spot_mask_to_sum = []

for roi in pixel_mask_dict.keys():
    if pixel_discovery_order_dict.get(roi) in ['A', 'C', 'G', 'T']:
        ref_mask_to_sum.append(pixel_mask_dict.get(roi))
    else:
        spot_mask_to_sum.append(pixel_mask_dict.get(roi))

ref_mask = np.sum(ref_mask_to_sum, axis = 0)
spot_mask = np.sum(spot_mask_to_sum, axis = 0)

ref_sizes = ndimage.sum_labels(np.ones(mask_shape), ref_mask, range(nb_labels + 1)).astype(int)
print(f"number of pixels per reference spot:\n {ref_sizes}")

spot_sizes = ndimage.sum_labels(np.ones(mask_shape), spot_mask, range(nb_labels + 1)).astype(int)
print(f"number of pixels per non-reference spot:\n {spot_sizes}")

sizes = np.array(list(map(max, ref_sizes, spot_sizes)))
print(f"number of pixels per spot:\n {sizes}")

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(ref_mask)
plt.show()

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(spot_mask)
plt.show()

images = {}
image_number = 62
df_files = df_files.loc[(df_files['file_info_nb'] >= image_number) & (df_files['file_info_nb'] < image_number+5)]

for index, row in df_files.iterrows():
    filenamepath = row['filenamepath']
    file_info_nb = row['file_info_nb']
    file_info_wl = row['wavelength']
    file_info_cy = row['cycle']
    file_info_ts = row['timestamp']

    filename = os.path.basename(filenamepath)
    print(f"{index:4}   {filename:43}   WL:{file_info_wl:4}   CY:{file_info_cy:3}   TS:{file_info_ts:9}")

    image = cv2.imread(filenamepath, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    images[file_info_wl] = {'image': image, 'cycle': int(file_info_cy), 'timestamp_ms': file_info_ts}

label_counter = [-1]*(nb_labels+1)  # +1 for 0 background
label_counter_in_subset = [-1]*(nb_labels+1)  # +1 for 0 background
new_format = 1

ratio = (sizes // max_number_of_pixel_per_spot)+1
print(sizes)
print(ratio)

spot_pixel_list = []

for r, c in np.ndindex(mask_shape):
    label = max(ref_mask[r, c], spot_mask[r, c])
    if label == 0:  # no reference and no spot
        continue

    label_counter[label] += 1

    if label_counter[label] % ratio[label] != 0:
        continue

    label_counter_in_subset[label] += 1

    roi_index = label - 1

    if new_format:
        dict_entry = {
            'spot': pixel_discovery_order_dict.get(roi_index),
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
                'spot': pixel_discovery_order_dict.get(roi_index),
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
df.sort_values(by=['spot', 'cycle', 'timestamp_ms'], inplace=True)
print(f"Writing {output_csv_filename}")
df.to_csv(output_csv_filename, index=False, lineterminator='\n')