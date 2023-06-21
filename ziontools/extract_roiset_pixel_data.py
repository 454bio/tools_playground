import roifile
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import os
from scipy import ndimage
import pathlib

from .common import get_cycle_files

'''
Extracts pixel data for each spot.
Produces a compact csv file format, but the timestamps are slightly inaccurate, they are taken from the 645 image.
'''

# TODO, subtract 4096?

new_format = 1

def extract_roiset_pixel_data(
        input_raw_path: str,
        roiset_zip_filename: str,
        output_csv_filename: str,
        max_number_of_pixel_per_spot: int = 300,
        image_number: int = 0
):

    df_files = get_cycle_files(input_raw_path)
    print(df_files)

    # check if roifile exists
    if not pathlib.Path(roiset_zip_filename).is_file():
        print(f"Cannot open {roiset_zip_filename}")
        exit(-1)

    rois = roifile.ImagejRoi.fromfile(roiset_zip_filename)
    print(rois)

    # read image size from first image
    image0 = cv.imread(df_files.iloc[0]['filenamepath'], cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data
    print(f"Image shape {image0.shape}")

    # mask for one RGB channel
    ref_mask = np.zeros(image0.shape[:2], dtype=np.uint8)
    spot_mask = np.zeros(image0.shape[:2], dtype=np.uint8)
    mask_shape = ref_mask.shape
    print(f"mask: {type(ref_mask)}, {ref_mask.dtype}, {ref_mask.shape}")

    for idx, roi in enumerate(rois):
        if __debug__:
            print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)
#            print(roi)

        yc = np.uint16(roi.top + (roi.bottom - roi.top) / 2)
        xc = np.uint16(roi.left + (roi.right - roi.left) / 2)
        x_axis_length = int((roi.right - roi.left)/2.0)
        y_axis_length = int((roi.bottom - roi.top)/2.0)

        label_id = idx + 1
        if roi.name in ['A', 'C', 'G', 'T', 'BG']:
            ref_mask = cv.ellipse(ref_mask, (xc, yc), [x_axis_length, y_axis_length], angle=0, startAngle=0, endAngle=360, color=label_id, thickness=-1)
        else:
            spot_mask = cv.ellipse(spot_mask, (xc, yc), [x_axis_length, y_axis_length], angle=0, startAngle=0, endAngle=360, color=label_id, thickness=-1)

    # how many regions?
    nb_labels = len(rois)
    label_ids = np.arange(1, nb_labels + 1) # range(1, nb_labels + 1)
    print(f"labels: {nb_labels}")

    ref_sizes = ndimage.sum_labels(np.ones(mask_shape), ref_mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per reference spot:\n {ref_sizes}")
    spot_sizes = ndimage.sum_labels(np.ones(mask_shape), spot_mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per non-reference spot:\n {spot_sizes}")
    sizes = np.array(list(map(max, ref_sizes, spot_sizes)))
    print(f"number of pixels per spot:\n {sizes}")

    plt.imshow(ref_mask)
    plt.show()
    plt.imshow(spot_mask)
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

    label_counter = [-1]*(nb_labels+1)  # +1 for 0 background
    label_counter_in_subset = [-1]*(nb_labels+1)  # +1 for 0 background

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
                'spot_index': roi_index+1,
                'spot_name': rois[roi_index].name,
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
                    'spot_index': roi_index + 1,
                    'spot_name': rois[roi_index].name,
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
    print(f"Writing {output_csv_filename}")
    df.to_csv(output_csv_filename, index=False, lineterminator='\n')
#    print(df.to_csv(index=False))


