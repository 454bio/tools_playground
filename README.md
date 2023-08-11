# 57k_image_roi_extraction_automation.py

This is a python script for segmentation of ROIs in a 57K mask. Each superspot is segmented, and the ROIs within each of those are extracted. 

The ROIs are represented as a dictionary of segmentation masks, key representing superspot number, value representing a 2D array of similar format used in pipeline code (each ROI represented by a distinct pixel value, ROI pixel values increment by 1 from one ROI to another). 

In theory, this dictionary can be fed to rest of pipeline, which would run downstream image analysis on each of the superspot mask arrays. Total time to run is < 1 minute.

## Installation

Before using the script, you need to have the required Python libraries installed. You can install them using the following command:

```bash
pip install opencv-python matplotlib numpy pandas
```

## Usage

To run the script and analyze an image, follow these steps:

1. Open a command-line interface (terminal or command prompt).

2. Navigate to the directory where the script is located.

3. Execute the script by providing the path to the input image as a command-line argument. For example:

   **On Windows:**
   ```bash
   python 57k_image_roi_extraction_automation.py path/to/your/image.jpg
   ```

   **On macOS/Linux:**
   ```bash
   python3 57k_image_roi_extraction_automation.py path/to/your/image.jpg
   ```

Replace `path/to/your/image.jpg` with the actual path to the 57K image you want to analyze. Reccomended use is on a cycle 1, 445 wavelength image.  

## Sample Input and Output

The script will generate visualizations of the extracted ROIs for each of the super spots, as well as a visualization of all extracted ROIs. 

The main function will also output a dictionary of the mask arrays for each of the super spots, as well as an overall mask array, for later downstream use in automated image to basecall pipeline. 

Information on super spot pixel intensities and dimensions is also generated. 

**Sample Input**

Input Image: /Users/akshita/S0205_raws/00000022_001A_00022_445_C003_001811562.tif

![Example Input Image](445_input_raw_image.jpg)

## Methodology

## Limitations, Future Directions, Notes


