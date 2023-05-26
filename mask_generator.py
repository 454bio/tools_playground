import torch
import torchvision
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('test.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Original image
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

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(masks[0].keys())

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
        if points[i] not in valid_points and masks[i]['area'] > 15000:
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
        #img[m] = [1, 1, 1, 1]
        img[m] = color_mask
    ax.imshow(img)

#Masks before any open cv processing
masks = process_masks(masks)
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


#Mask generation using open cv processing
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area= 15000,  # Requires open-cv to run post-processing
    output_mode = "binary_mask",
)

masks2 = mask_generator_2.generate(image)

new_masks = process_masks(masks2)

#Display masks in color
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(new_masks)
plt.axis('off')
plt.savefig('color_mask')
plt.show() 


