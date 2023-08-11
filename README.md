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

### Sample Input

**Input Image: ex. 445_input_raw_image.tif**
![originasdlfjads](https://github.com/454bio/tools_playground/assets/129779339/278403ae-03ce-43e5-8cea-6b10810023bc)

**Output: Super Spots ROI Segmentation**
![super spot](https://github.com/454bio/tools_playground/assets/129779339/6d10f59e-563e-4da9-90c6-26272b2b9fb7)

**Output: ROI Segmentations Within Each Super Spot**
![1](https://github.com/454bio/tools_playground/assets/129779339/2e668148-cc49-4176-abb3-e515291b333b)
![5](https://github.com/454bio/tools_playground/assets/129779339/c4132cb3-91aa-40c2-a240-8063dc8b7a8f)
![17](https://github.com/454bio/tools_playground/assets/129779339/ab454ad4-0bee-43e5-a833-b008649cfa22)

...

**Output: All Extracted ROIs Mask**
![extracted rois](https://github.com/454bio/tools_playground/assets/129779339/46889682-4695-42e6-af68-6dd2328842c0)

**Output: Mask arrays for all super spots, containing segmentations for ROIs**
![image](https://github.com/454bio/tools_playground/assets/129779339/cb6652d2-15dd-4214-bbed-a6a70924b45f)

## Methodology

## Limitations, Future Directions, Notes


