import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import csv
import argparse
from tqdm import tqdm

def main(inputpath: str, outputfilename: str):
    
    print("Finding and displaying the first 365 image, from which the spots will be extracted")
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
    
    image = cv2.imread(filenames[spot_file])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    sys.path.append("/segment-anything")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    device = "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
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
    
    valid_masks = []
    for mask in new_masks:
        array = [mask['bbox'][2], mask['bbox'][3]]

        X = max(array)
        Y = array[array.index(X) - 1]  # Get the other value in the array
    
        if X/Y < 1.5:
            valid_masks.append(mask)
    
    new_masks = valid_masks
    
    sam_mask, pixel_mask_dict, pixel_bbox_dict = get_mask_arr_dict_and_sam_mask(new_masks)   
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(new_masks)
    plt.axis('off')
    plt.show()
    
    removeNest = input("Do you want to remove any nested ROIs? (Y/N)")
    nest = False
    if removeNest == 'Y':
        nest = True
    
    if (nest):
        
        keep_mask, new_pixel_mask_dict = remove_nested(pixel_mask_dict, image)
        sam_mask = keep_mask
        pixel_mask_dict = new_pixel_mask_dict
        
        plt.figure(figsize=(10,10))
        plt.imshow(sam_mask)
        plt.axis('off')
        plt.show()
    
   
    natural = input("Do you want to assign a natural numerical ordering to the ROIs? (Y/N)")
    wantNatural = False
    if (natural == 'Y'):
        wantNatural = True
        
    if not wantNatural:
        pixel_values = np.unique(sam_mask[sam_mask != 0])
        
        pixel_discovery_order_dict = {}
        
        for i, pixel in enumerate(pixel_values):
            pixel_discovery_order_dict[pixel] = i + 1
            
        roi_ordering = np.zeros(image.shape[:2], dtype=np.uint8)
    
        for i, val in enumerate(pixel_discovery_order_dict.keys()):
            text = str(pixel_discovery_order_dict.get(val))
           
            bbox = pixel_bbox_dict.get(val)
            x, y, w, h = bbox
            midpoint_x = x + (w / 2)
            midpoint_y = y + (h / 2)
            
            loc = (int(midpoint_x), int(midpoint_y))
            
            roi_ordering = cv2.putText(roi_ordering, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
            
        plt.figure(figsize=(10,10))
        plt.imshow(sam_mask)
        plt.imshow(roi_ordering, alpha = 0.75)
        plt.axis('off')
        plt.show()
        
        bases = ['A', 'C', 'T', 'G']
        user_base_spots = []
        A = int(input("Which number spot corresponds to base A"))
        C = int(input("Which number spot corresponds to base C"))
        T = int(input("Which number spot corresponds to base T"))
        G = int(input("Which number spot corresponds to base G"))
        
        user_base_spots.extend([A, C, T, G])

        for key, discovery in pixel_discovery_order_dict.items():
            if discovery in user_base_spots:
                base_index = user_base_spots.index(discovery)
                base = bases[base_index]
                pixel_discovery_order_dict[key] = base
        
        roi_ordering = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for i, val in enumerate(pixel_discovery_order_dict.keys()):
            text = str(pixel_discovery_order_dict.get(val))
           
            bbox = pixel_bbox_dict.get(val)
            x, y, w, h = bbox
            midpoint_x = x + (w / 2)
            midpoint_y = y + (h / 2)
            
            loc = (int(midpoint_x), int(midpoint_y))
            
            roi_ordering = cv2.putText(roi_ordering, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
            
        plt.figure(figsize=(10,10))
        plt.imshow(sam_mask)
        plt.imshow(roi_ordering, alpha = 0.75)
        plt.axis('off')
        plt.show()
        
        print("Storing segmentation masks to CSV")
        store_arrays_as_csv(pixel_mask_dict, pixel_discovery_order_dict, outputfilename)
        
        print("Storing pixel color-name mapping dictionary to CSV")
        store_dictionary_to_csv(pixel_discovery_order_dict, outputfilename)
        
        print("Storing pixel color-name mapping dictionary to CSV")
        store_dictionary_to_csv(pixel_discovery_order_dict, outputfilename)
        
        print("Storing reference and spot masks to CSV")
        #Store reference and spot masks

        ref_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        spot_mask = np.zeros(image.shape[:2], dtype=np.uint8)
      
        mask_shape = ref_mask.shape
        
        ref_mask_to_sum = []
        spot_mask_to_sum = []
        
        for roi in pixel_mask_dict.keys():
            if pixel_discovery_order_dict.get(roi) in ['A', 'C', 'G', 'T']:
                ref_mask_to_sum.append(pixel_mask_dict.get(roi))
            else:
                spot_mask_to_sum.append(pixel_mask_dict.get(roi))
        
        ref_mask = np.sum(ref_mask_to_sum, axis = 0)
        spot_mask = np.sum(spot_mask_to_sum, axis = 0)

        store_mask_arr_as_csv(sam_mask, outputfilename, 'sam_mask.csv')
        store_mask_arr_as_csv(ref_mask, outputfilename, 'ref_mask.csv')
        store_mask_arr_as_csv(spot_mask, outputfilename, 'spot_mask.csv')
        
    else:   
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
            
        undiscovered_rois = discovered_rois[:]  # Create a shallow copy
        ordered_discovered_rois = []
    
        from tqdm import tqdm
        
        undiscovered_rois = discovered_rois[:]  # Create a shallow copy
        ordered_discovered_rois = []
        
        # Create a progress bar with the length of the outer loop
        progress_bar = tqdm(discovered_rois, desc="Determining natural ordering of ROIs", unit="ROI")
        
        for i in progress_bar:
            curr_overlap = []
            sorted_curr = []
        
            for j in undiscovered_rois:
                if check_if_two_overlap_row(i, j, pixel_mask_dict):
                    curr_overlap.append(j)
        
            sorted_curr = sorted(curr_overlap, key=lambda k: masks_leftmost_mapping[k][0])
            ordered_discovered_rois.extend(sorted_curr)
            undiscovered_rois = [ele for ele in undiscovered_rois if ele not in sorted_curr]
            curr_overlap = []
        
        # Close the progress bar
        progress_bar.close()

            
        for i, pixel in enumerate(ordered_discovered_rois):
            pixel_discovery_order_dict[pixel] = (i + 1)
            
        roi_ordering = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, val in enumerate(discovered_rois):
            text = str(pixel_discovery_order_dict.get(val))
            loc = masks_leftmost_mapping[val]
            roi_ordering = cv2.putText(roi_ordering, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
            
        plt.figure(figsize=(10,10))
        plt.imshow(sam_mask)
        plt.imshow(roi_ordering, alpha = 0.75)
        plt.axis('off')
        plt.show()
        
        bases = ['A', 'C', 'T', 'G']
        user_base_spots = []
        A = int(input("Which number spot corresponds to base A"))
        C = int(input("Which number spot corresponds to base C"))
        T = int(input("Which number spot corresponds to base T"))
        G = int(input("Which number spot corresponds to base G"))
        
        print("Re-ordering remaining spots")
        
        user_base_spots.extend([A, C, T, G])
        
        for key, discovery in pixel_discovery_order_dict.items():
            if discovery in user_base_spots:
                base_index = user_base_spots.index(discovery)
                base = bases[base_index]
                pixel_discovery_order_dict[key] = base
                
        undiscovered_rois = ordered_discovered_rois[:]
        undiscovered_rois = [ele for ele in ordered_discovered_rois if pixel_discovery_order_dict.get(ele) not in bases]
        
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
        plt.axis('off')
        plt.show()
        
        print("Storing segmentation masks to CSV")
        store_arrays_as_csv(pixel_mask_dict, pixel_discovery_order_dict, outputfilename)
        
        print("Storing pixel color-name mapping dictionary to CSV")
        store_dictionary_to_csv(pixel_discovery_order_dict, outputfilename)
        
        print("Storing reference and spot masks to CSV")
        ref_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        spot_mask = np.zeros(image.shape[:2], dtype=np.uint8)
               
        ref_mask_to_sum = []
        spot_mask_to_sum = []
        
        for roi in pixel_mask_dict.keys():
            if pixel_discovery_order_dict.get(roi) in ['A', 'C', 'G', 'T']:
                ref_mask_to_sum.append(pixel_mask_dict.get(roi))
            else:
                spot_mask_to_sum.append(pixel_mask_dict.get(roi))
                
        ref_mask = np.sum(ref_mask_to_sum, axis = 0)
        spot_mask = np.sum(spot_mask_to_sum, axis = 0)
        
        store_mask_arr_as_csv(sam_mask, outputfilename, 'sam_mask.csv')
        store_mask_arr_as_csv(ref_mask, outputfilename, 'ref_mask.csv')
        store_mask_arr_as_csv(spot_mask, outputfilename, 'spot_mask.csv')

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
            
    valid2_masks = []
    
    for mask in valid_masks:
        y = mask['bbox'][1]
        if (y > 50 ):
            valid2_masks.append(mask)
            
    return valid2_masks   

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
        img[m] = color_mask
    ax.imshow(img)

#Map pixel value to that ROI's mask array, get sam mask -> returns sam mask and pixel mappings to their indivdual mask array
def get_mask_arr_dict_and_sam_mask(masks):
    pixel_mask_dict = {}
    pixel_bbox_dict = {}
                                   
    mask = np.empty((1520, 2028))
    to_sum = []
                                   
    for i, mask in enumerate(masks):
        pixel_mask_dict[i + 1] = mask['segmentation'].astype(int) * (i + 1)
        pixel_bbox_dict[i + 1] = mask['bbox']
        
        to_sum.append(pixel_mask_dict[i + 1])
    
    mask = np.sum(to_sum, axis = 0)
                                   
    return mask, pixel_mask_dict, pixel_bbox_dict

#Get leftmost coordinate of a mask array
def find_roi_leftmost_coord(roi_mask_arr):
    rows, cols = roi_mask_arr.shape
    for i in range(cols):
        for j in range(rows):
            curr = roi_mask_arr[j, i]
            if (curr != 0):
                return (i, j) 

def get_ycoords(pixel, pixel_mask_dict):
    mask_arr = pixel_mask_dict.get(pixel)
    
    y_coords = []
    
    rows, cols = mask_arr.shape

    for i in range(rows):
        for j in range(cols):
            curr = mask_arr[i, j]
            if curr != 0 and not (i in y_coords):
                y_coords.append(i)
    
    return y_coords

def check_if_two_overlap_row(p_1, p_2, pixel_mask_dict):
    y_1 = get_ycoords(p_1, pixel_mask_dict)
    y_2 = get_ycoords(p_2, pixel_mask_dict)
    
    set_1 = set(y_1)
    set_2 = set(y_2)
    
    if len(set_1.intersection(set_2)) > 0:
        return True
    
    return False

def store_arrays_as_csv(pixel_mask_dict, pixel_discovery_order_dict, folder_path):
    array_folder = str(folder_path) + '/rois'
    # Create the folder if it doesn't exist
    os.makedirs(array_folder, exist_ok=True)

    # Create a progress bar with the length of the dictionary
    progress_bar = tqdm(pixel_mask_dict.items(), desc="Processing", unit="array")

    for key, array in progress_bar:
        # Get the discovery value for the key
        discovery_value = pixel_discovery_order_dict.get(key)

        # Generate the filename based on the discovery value
        filename = f"{discovery_value}.csv"

        # Convert the 2D numpy array to a CSV file
        csv_path = os.path.join(array_folder, filename)
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(array)

    # Close the progress bar
    progress_bar.close()

def store_mask_arr_as_csv(roi_mask, folder_path, name):
    if roi_mask is not None:
        csv_filename = os.path.join(folder_path, name)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(roi_mask)
            
def store_dictionary_to_csv(dictionary, output_filename):
    output_filename = output_filename + '/pixeldiscoverydict.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dictionary.items())

def get_dictionary_from_csv(filename):
    dictionary = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key, value = row
            dictionary[key] = value
    return dictionary

def get_xcoords(pixel, pixel_mask_dict):
    mask_arr = pixel_mask_dict.get(pixel)
    
    x_coords = []
    
    rows, cols = mask_arr.shape

    for i in range(rows):
        for j in range(cols):
            curr = mask_arr[i, j]
            if curr != 0 and not (i in x_coords):
                x_coords.append(j)
    
    return x_coords

def check_if_two_overlap_col(p_1, p_2, pixel_mask_dict):
    x_1 = get_xcoords(p_1, pixel_mask_dict)
    x_2 = get_xcoords(p_2, pixel_mask_dict)
    
    set_1 = set(x_1)
    set_2 = set(x_2)
    
    if len(set_1.intersection(set_2)) > 0:
        return True
    
    return False

def remove_nested(pixel_mask_dict, image):
    keep_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    keep_pixels = []
    pixels_to_remove = []

    progress_bar = tqdm(pixel_mask_dict.keys(), desc="Detecting and removing nested ROIs", total=len(pixel_mask_dict.keys()))

    for pixel in progress_bar:
        for keep_pixel in keep_pixels:
            if (check_if_two_overlap_row(pixel, keep_pixel, pixel_mask_dict) and check_if_two_overlap_col(pixel, keep_pixel, pixel_mask_dict)):
                pixels_to_remove.append(pixel)
                break
        else:
            keep_pixels.append(pixel)

    keep_pixels.sort() #sorts keep pixels from least to greatest

    new_pixel_mask_dict = {} #maps the New pixels, like straight up new, to their mask array which we also transform
    
    old_to_new = {} #maps the old pixel value to its translated new one

    for i, pixel in enumerate(keep_pixels):
        old_to_new[pixel] = i + 1
        
    for old_pixel in old_to_new.keys(): #basically iterating through the old pixels aka the og keep_pixels
        old_arr = pixel_mask_dict.get(old_pixel) #original mask corresponding to the old pixel
        new_arr = np.where(old_arr == old_pixel, old_to_new.get(old_pixel), old_arr)
        new_pixel_mask_dict[old_to_new.get(old_pixel)] = new_arr
            
    keep_mask_to_sum = []
    new_pixels = new_pixel_mask_dict.keys()
    
    for new in new_pixels:
        keep_mask_to_sum.append(new_pixel_mask_dict.get(new))

    keep_mask = np.sum(keep_mask_to_sum, axis=0)

    return keep_mask, new_pixel_mask_dict

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

    args = parser.parse_args()
    inputpath = args.input
    print(f"inputpath: {inputpath}")

    outputfilename = args.outputcsv
    print(f"outputfilename: {outputfilename}")

    # main
    main(inputpath, outputfilename)
