# extract_and_name_rois.py

Python tool to extract and name ROIs using Segment-Anything model (SAM).

## Installation

Before installing the packages, ensure that you have the following prerequisites installed on your system:

Python (version 3.3 or higher)
pip (Python package installer)

### Packages and Dependencies

This code requires pytorch>=1.7 and torchvision>=0.8. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Prerequisites

Before installing the packages, ensure that you have the following prerequisites installed on your system:

- Python (version 3.6 or higher)
- pip (Python package installer)

### Installation Steps

1. Open a terminal or command prompt.

2. Create a new virtual environment (optional but recommended):

```bash
python3 -m venv myenv
```

3. Activate the virtual environment:

**For Windows:**

```bash
myenv\Scripts\activate
```

**For macOS and Linux:**

```bash
source myenv/bin/activate
```

4. Install the remaining required packages and their dependencies using the following command:

```bash
pip install numpy scipy matplotlib opencv-python roifile pandas plotly scikit-learn
```

### Pulling the SAM Repository

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Additional Notes

- Place the extract_and_name_rois and all other files in this repository in the main segment-anything folder that was installed via the pip install git repository command above
- The above command installs the necessary packages, including their dependencies, in your virtual environment.

- If you already have some of these packages installed, you can use the `--upgrade` flag to ensure you have the latest versions:

```bash
pip install --upgrade torch torchvision numpy scipy matplotlib opencv-python roifile pandas plotly scikit-learn
```

### Model Checkpoint

Install the vit_h version of the model [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Usage

The input -i is the directory of raw images

The output -o is the directory where the CSVs for the ROIs, entire ROI mask, reference mask, and spot mask are to be stored, as well as the dictionary mapping pixel value and spot name. 

Performs phase correction and calls bases for each spot, with default parameters

```bash
python extract_and_name_rois.py 
    -i /Users/akshitapanigrahi/Documents/data/193/raws_aligned_images 
    -o /Users/akshitapanigrahi/data/193/analysis
```
# Remaining pipeline

Below are command line instructions to run the remaining tools in Dom's pipeline, using the outputs from extract_and_name_rois.py. All of these downstream tools were modified from the most recent version as of July 4, 2023. 

The inputs to these scripts are defined the exact same as they are in the main branch, except for the -r roiset input, which in this pipeline is the directory created by extract_and_name_rois.py, containing the mask csvs and pixel naming dictionary. In the examples below, the input -r is simply defined as the exact same as the output -o for extract_and_name_rois above. 

```bash
extract_roiset_metrics_to_csv_automation.py
python extract_roiset_metrics_to_csv_automation.py 
    -i /Users/akshitapanigrahi/Documents/data/193/raws_aligned_images 
    -r /Users/akshitapanigrahi/Documents/data/193/analysis 
    -o /Users/akshitapanigrahi/Desktop/roiset_metrics_automated.csv
```

```bash
extract_roi_pixel_data_automation.py
python extract_roi_pixel_data_automation.py 
    -i /Users/akshitapanigrahi/Documents/data/193/raws_aligned_images 
    -r /Users/akshitapanigrahi/Documents/data/193/analysis 
    -o /Users/akshitapanigrahi/Desktop/spot_pixel_data_automated.csv
```

```bash
color_transform_automation.py
python color_transform_automation.py 
    -p /Users/akshitapanigrahi/Desktop/spot_pixel_data_automated.csv 
    -r /Users/akshitapanigrahi/Documents/data/193/analysis 
    -i /Users/akshitapanigrahi/Documents/data/193/raws_aligned_images 
    -o /Users/akshitapanigrahi/Desktop
```

```bash
dephased_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\color_transformed_spots_automated.csv
    -o \Users\akshitapanigrahi\Desktop
    -grid
```
Disclaimer: Basecalling output is as expected not great, as the ROIs are extracted for the entire spot, and the background defined as the entire area that is not a spot. These scripts were just added to see the entire downstream pipeline with the automated ROI extraction. 
