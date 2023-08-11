import matplotlib.pyplot as plt
import numpy as np
import cv2
from pylab import array 
import pandas as pd
import argparse

def main(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    plt.figure(figsize=(20, 20))
    plt.title('Original Image')
    plt.imshow(image)
    plt.show()
    
    # Assuming grayhighlow is a 3-channel BGR image, convert it to grayscale
    grayhighlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image with black and white
    ret, binary_image = cv2.threshold(grayhighlow, 28, 255, cv2.THRESH_BINARY)

    # Apply a Gaussian blur to find the superspots
    blur = cv2.GaussianBlur(binary_image,(15,15),cv2.BORDER_DEFAULT)

    # Threshold the blurred image to BW
    ret, binary_image_blurred = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    min_area_threshold = 2000  # Adjust as needed
    max_area_threshold = 100000  # Adjust as needed

    min_component_size = 10  # Adjust as needed to remove small regions
    min_circularity = 0  # Adjust as needed to keep only very circular components

    def get_spot_intensity(super_spot_number, test_point, gray_region):
        radius = 3
        #original_gray_region = original_gray_regions.get(super_spot_number)
        
        x = test_point[0]
        y = test_point[1]
            
        # Create a mask with a circular region of interest
        mask = np.zeros_like(gray_region)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Calculate the mean pixel intensity within the circular region
        masked_pixels = cv2.bitwise_and(gray_region, mask)
        num_pixels = np.sum(masked_pixels > 0)
        intensity_sum = np.sum(masked_pixels)
        
        #print('num_pixels: ' + str(num_pixels))
        #print('intensity sum: ' + str(intensity_sum))
        
        average_intensity = intensity_sum / num_pixels
        
        return average_intensity

    def is_contour_circular(contour):
        # Function to determine if a contour is circular based on its area and perimeter
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > 0.6  # Adjust the threshold for circularity as needed

    def process_binary_image(binary_image, original_image):
        # Define the contrast control parameter (alpha) and brightness control parameter (beta)
        alpha = 1  # Increase contrast (you can adjust this value as needed)
        beta = 1    # No brightness adjustment (you can adjust this value as needed)
        
        final_contours = []
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new black image with the same size as the original image
        result_image = np.zeros_like(binary_image)    

        # Create a mask to hold the selected circular regions
        super_spot_mask = np.zeros_like(binary_image)
        
        curr_super_spot = 0
        
        # Create a blank image to hold the labels
        label_image = np.zeros_like(original_image)
        
        # Create pixel intensity table
        df_data = []
        
        # Dictionary mapping super spot to its mask
        super_spot_sam_masks = {}
        
        # Dictionary mapping super spot to its original gray region
        original_gray_regions = {}
        
        # Dictionary mapping super spot to a list of tuples of the center points of its ROIs- keyed by its super spot number, curr_super_spot
        super_spot_center_points = {}
        
        # Dictionary mapping super spot to its width (for later use in filling of rows)
        super_spot_widths = {}
        
        # Dictionary mapping super spot to its average region intensity (for later use in filling missing spots)
        super_spot_intensities = {}
        
        # Dictionary mapping super spot to its bounding box
        super_spot_bboxes = {}
            
        # Iterate through the contours and filter based on size and circularity
        for idx, contour in enumerate(contours):        
            if cv2.contourArea(contour) > min_area_threshold and cv2.contourArea(contour) < max_area_threshold and is_contour_circular(contour):
                final_contours.append(contour)
                curr_super_spot += 1

                # Fill the contour in the mask
                cv2.drawContours(super_spot_mask, [contour], -1, 255, cv2.FILLED)

                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                super_spot_bboxes[curr_super_spot] = (x, y, w, h)
                            
                super_spot_widths[curr_super_spot] = w
                
                # Get the region from the original image corresponding to the contour
                region = original_image[y:y+h, x:x+w]
                
                total_intensity = np.sum(region)
                total_pixels = region.size
                intensity = total_intensity / total_pixels         
                
                df_data.append({'Superspot': curr_super_spot, 'Total Intensity': intensity})
                
                if (intensity < 27):
                    alpha = 2
                
                # Convert the region to grayscale
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                gray_region = cv2.convertScaleAbs(gray_region, alpha=alpha, beta=beta)
                
                alpha = 1

                # Create a mask of the same size as the region
                contour_mask = np.zeros_like(gray_region)

                # Draw the contour on the mask
                cv2.drawContours(contour_mask, [contour - (x, y)], -1, 255, cv2.FILLED)
                
                if (intensity < 27):
                    gray_region = cv2.GaussianBlur(gray_region,(3, 3),cv2.BORDER_DEFAULT)
                
                # Apply the adaptive threshold only within the contour using the mask
                adaptive_thresholded = cv2.adaptiveThreshold(gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -1)
                adaptive_thresholded[contour_mask == 0] = 0
                        
                # Label connected components in the thresholded image
                num_labels, labeled_image, stats, _ = cv2.connectedComponentsWithStats(adaptive_thresholded)
                            
                # Remove small regions based on the minimum component size and circularity threshold
                for label in range(1, num_labels):
                    if stats[label, cv2.CC_STAT_AREA] < min_component_size:
                        labeled_image[labeled_image == label] = 0
                    else:
                        contour_points = np.argwhere(labeled_image == label)
                        area = stats[label, cv2.CC_STAT_AREA]
                        perimeter = 2 * (stats[label, cv2.CC_STAT_WIDTH] + stats[label, cv2.CC_STAT_HEIGHT])
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity < min_circularity:
                            labeled_image[contour_points[:, 0], contour_points[:, 1]] = 0
                                                    
                labeled_image = labeled_image.astype(np.uint8)
                num_labels, cumulative_labeled_image, stats, centroids = cv2.connectedComponentsWithStats(labeled_image)
                
                center_points = centroids[1:]
                        
                super_spot_center_points[curr_super_spot] = center_points
                                                                
                # Create a copy of the cumulative_labeled_image to draw the circles
                ellipses = np.zeros_like(cumulative_labeled_image)
                
                num_ellipses = 0

                # Draw circles at the center points of each connected component
                
                avg_intensity = 0
                
                for i, center_point in enumerate(center_points):
                    # Skip the background component (index 0)
                    label_index = i + 1
                    x_c, y_c = center_point.astype(int)

                    # Draw a circle with a radius of 3 at the center point with the associated color
                    cv2.circle(ellipses, (x_c, y_c), 3, label_index, -1)  # -1 fills the circle
                    
                    super_spot_intensity = get_spot_intensity(curr_super_spot, (x_c, y_c), gray_region)
                    avg_intensity += super_spot_intensity
                    
                    num_ellipses += 1
                    
                avg_intensity /= num_ellipses
                super_spot_intensities[curr_super_spot] = avg_intensity
                    
                super_spot_sam_masks[curr_super_spot] = ellipses
                original_gray_regions[curr_super_spot] = gray_region
                                                    
                # Create a mask for the filtered region
                mask = np.zeros_like(gray_region)
                mask[ellipses != 0] = curr_super_spot
                
                # Draw the filtered region on the result image using the same color
                result_image[y:y+h, x:x+w][ellipses != 0] = curr_super_spot

                # Label each super spot in the label_image
                text_position = (x + 5, y + 20)  # Adjust the position of the text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 0, 0)  # White color
                line_type = 4
                cv2.putText(label_image, str(curr_super_spot), text_position, font, font_scale, font_color, line_type)
                
        df = pd.DataFrame(df_data)

        return result_image, super_spot_mask, final_contours, super_spot_sam_masks, label_image, df, super_spot_center_points, original_gray_regions, super_spot_widths, super_spot_intensities, super_spot_bboxes

    # Process the binary image and display each region's computed thresholded region as an individual figure
    result_image, super_spot_mask, final_contours, super_spot_sam_masks, label_image, df, super_spot_center_points, original_gray_regions, super_spot_widths, super_spot_intensities, super_spot_bboxes = process_binary_image(binary_image_blurred, image)

    def group_centerpoints_by_row(super_spot_number, y_tolerance=1.5):
        centerpoints = super_spot_center_points.get(super_spot_number)

        # Create an empty dictionary to store the grouped centerpoints
        grouped_rows = {}

        # Loop through each centerpoint
        for idx, point in enumerate(centerpoints):
            # Get the Y-coordinate (2nd value) of the centerpoint
            y_coord = point[1]
            
            # Check if there's a row with a similar y_coord in the dictionary
            found_row = False
            for row_num, row_data in grouped_rows.items():
                if abs(y_coord - row_num) <= y_tolerance:
                    row_data['points'].append(point)
                    row_data['total_y'] += y_coord
                    row_data['count'] += 1
                    found_row = True
                    break
            
            # If no similar row found, create a new row in the dictionary
            if not found_row:
                grouped_rows[y_coord] = {'points': [point], 'total_y': y_coord, 'count': 1}
        
        row_dict = {}
        for i, row_data in enumerate(grouped_rows.values()):
            avg_y = row_data['total_y'] / row_data['count']
            row_dict[i + 1] = {'points': row_data['points'], 'avg_y': avg_y}
        
        # Filter out any rows that are not within a certain amount
        row_keys = sorted(row_dict.keys(), key=lambda x: row_dict[x]['avg_y'])  # Sorting keys based on avg_y
        
        filtered_row_dict = {}
        
        if (row_dict.get(2).get('avg_y') - row_dict.get(1).get('avg_y') >= 6):
            filtered_row_dict[1] = {'points': row_dict[1]['points']}
        
        for i in range(1, len(row_keys) - 1):
            curr_row_key = row_keys[i]
            prev_row_key = row_keys[i - 1]
            next_row_key = row_keys[i + 1]
            
            if not (((row_dict[next_row_key]['avg_y'] - row_dict[curr_row_key]['avg_y']) < 6) and
                    ((row_dict[curr_row_key]['avg_y'] - row_dict[prev_row_key]['avg_y']) < 6)):
                filtered_row_dict[curr_row_key] = {'points': row_dict[curr_row_key]['points']}
                
        if (row_dict.get(len(row_dict.keys())).get('avg_y') - row_dict.get(len(row_dict.keys()) - 1).get('avg_y') >= 6):
            filtered_row_dict[len(row_dict.keys())] = {'points': row_dict[len(row_dict.keys())]['points']}
        
        ordred_filtered_row_dict = {}
        for key, value in enumerate(filtered_row_dict.items()):
            # Sort the list of tuples based on the X-coordinate (index 0)
            sorted_points = sorted(value[1]['points'], key=lambda point: point[0])
            ordred_filtered_row_dict[key + 1] = sorted_points
            
        return ordred_filtered_row_dict

    def convert_dict_values_to_list(row_dict):
        centerpoints = []
        for row_num, entry in row_dict.items():
            points  = entry
            for point in points:
                x = point[0]
                y = point[1]
                centerpoints.append(([x, y]))
        
        return  np.array(centerpoints)

    def check_and_fill_row(row_num, super_spot_number):
        width = super_spot_widths.get(super_spot_number)
        row_dict = group_centerpoints_by_row(super_spot_number)
        centerpoints = row_dict.get(row_num)
                
        avg_intensity = super_spot_intensities.get(super_spot_number)
        
        new_row_points = []
        
        curr_exist = 0
        
        starting_x = int(centerpoints[0][0])
        starting_y = int(centerpoints[0][1])
        
        # Start at left-most found point, iterate to the right
        x = starting_x
            
        remaining = True
        other = False
        
        #print(width)
                    
        while x <= width:
            #print('x: ' + str(x))
            if (remaining):
                if (abs(x - centerpoints[curr_exist][0]) < 2):
                    new_row_points.append(centerpoints[curr_exist])
                    curr_exist += 1
                    other = True
                                    
                    if (curr_exist == len(centerpoints)):
                        remaining = False
                else:
                    other = False
            
            #print('intensity: ' + str(get_spot_intensity(super_spot_number, (x, starting_y), original_gray_regions.get(super_spot_number))))
            
            if not other:
                if (abs(get_spot_intensity(super_spot_number, (x, starting_y), original_gray_regions.get(super_spot_number)) >= avg_intensity)):
                    new_row_points.append(array([x, starting_y]))
                elif (abs(get_spot_intensity(super_spot_number, (x, starting_y), original_gray_regions.get(super_spot_number)) - avg_intensity) < 3):
                    new_row_points.append(array([x, starting_y]))
                    
            x += 8
            
        # Start at left-most point, iterate to the left
        x = starting_x - 8
        
        while x >= 0:
            if (abs(get_spot_intensity(super_spot_number, (x, starting_y), original_gray_regions.get(super_spot_number))) >= avg_intensity):
                    new_row_points.append(array([x, starting_y]))
            elif (abs(get_spot_intensity(super_spot_number, (x, starting_y), original_gray_regions.get(super_spot_number)) - avg_intensity) < 3):
                new_row_points.append(array([x, starting_y]))
            
            x -= 8    
                    
        return new_row_points

    def fill_super_spot(super_spot_number):
            row_dict = group_centerpoints_by_row(super_spot_number)
            all_rows = list(row_dict.keys())
                
            row_center_points = []
            
            for row in all_rows:
                row_center_points.extend(check_and_fill_row(row, super_spot_number))
            
            return row_center_points

    def test_fill_align_visualize(test_spot):
        points = fill_super_spot(test_spot)
        og_gray = original_gray_regions.get(test_spot)
        
        fig, ax = plt.subplots()

        for point in points:    
            x = point[0]
            y = point[1]

            ax.add_patch(plt.Circle((x, y), radius=3, color='b', alpha=0.75))

        ax.set_aspect('equal', adjustable='datalim')

        plt.imshow(og_gray, cmap='gray')    
        plt.axis("off")
        plt.show()

    def refill_and_get_center_mappings(intensity_thresh=28):
        # If super spot intensity is below a certain threshold, apply refilling
        new_super_spot_center_points = super_spot_center_points
        
        for super_spot in super_spot_intensities.keys():
            df_intensity = df[df['Superspot'] == super_spot]['Total Intensity'].values[0]
            
            if df_intensity <= intensity_thresh:
                new_super_spot_center_points[super_spot] = fill_super_spot(super_spot)
                
        return new_super_spot_center_points

    def redraw_filtered_centers(result_image, super_spot_mask, 
                             super_spot_sam_masks, label_image, super_spot_center_points, original_gray_regions, super_spot_bboxes):    
        
        new_super_spot_center_points = refill_and_get_center_mappings()
        
        for curr_super_spot in super_spot_intensities.keys():

            center_points = new_super_spot_center_points.get(curr_super_spot)
            super_spot_center_points[curr_super_spot] = center_points

            # Create a copy of the cumulative_labeled_image to draw the circles
            ellipses = np.zeros_like(original_gray_regions.get(curr_super_spot))
                
            for i, center_point in enumerate(center_points):
                # Skip the background component (index 0)
                label_index = i + 1
                x_c, y_c = center_point.astype(int)

                # Draw a circle with a radius of 3 at the center point with the associated color
                cv2.circle(ellipses, (x_c, y_c), 3, label_index, -1)  # -1 fills the circle
                
            super_spot_sam_masks[curr_super_spot] = ellipses

            # Create a mask for the filtered region
            mask = np.zeros_like(original_gray_regions[curr_super_spot])
            mask[ellipses != 0] = curr_super_spot
            
            x, y, w, h = super_spot_bboxes.get(curr_super_spot)

            # Draw the filtered region on the result image using the same color
            result_image[y:y+h, x:x+w][ellipses  != 0] = 255

            # Label each super spot in the label_image
            text_position = (x + 5, y + 20)  # Adjust the position of the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 0, 0)  # White color
            line_type = 4
            cv2.putText(label_image, str(curr_super_spot), text_position, font, font_scale, font_color, line_type)
            
            original_region = image[y:y+h, x:x+w]
            
            # Create a single figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Plot the original region in the first subplot
            axes[0].imshow(cv2.cvtColor(original_region, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Superspot " + str(curr_super_spot) + ", Original Region")
            axes[0].axis("off")

            # Plot the filtered thresholded result in the second subplot
            axes[1].imshow(original_gray_regions[curr_super_spot], cmap='gray')
            axes[1].set_title("Superspot " + str(curr_super_spot) + ", Segmented ROIs")

            for point in center_points:
                x_c = point[0]
                y_c = point[1]

                axes[1].add_patch(plt.Circle((x_c, y_c), radius=3, color='r', alpha=0.75))

            axes[1].set_aspect('equal', adjustable='datalim')
            axes[1].axis("off")

            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.1)
            plt.show()
                
        return result_image, super_spot_sam_masks, label_image, super_spot_center_points
    
    final_result_image, final_super_spot_sam_masks, final_label_image, final_super_spot_center_points = redraw_filtered_centers(result_image, super_spot_mask, super_spot_sam_masks, label_image, super_spot_center_points, original_gray_regions, super_spot_bboxes)
    plt.figure(figsize=(20, 20))
    plt.title('Extracted ROIs')
    plt.imshow(final_result_image, cmap='gray')
    plt.imshow(final_label_image, alpha=0.5, cmap='gray')
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image and display results.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    main(args.image_path)