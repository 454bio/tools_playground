# dephased_basecalls

dephased_basecalls.py - the basic app to perform base calling, by Mel Davey, 2023

Does the following for each spot
1. loads the measured signals for each dye at the start of each cycle, from output of color_transform.py
2. instantiates a model to calculate expected signals based on phase params
3. applies the model to predict signals using a what-if strategy at each cycle, trying each base and seeing which was best

## Usage

Performs a grid-search over a hard-coded range of cf/ie/dr params, then performs  phase correction, calls bases, shows plots, on best cf/ie/dr params, for each spot

The input 'spots' is the output file from color_transform.py

The output -o is the directory where the dephased basecalls, predicted signal intensities, and basecalls histogram are stored. 

```python
python dephased_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\S0189_color_transformed_spots.csv
    -o \Users\akshitapanigrahi\Desktop
    -grid
```

Displays base call histograms and measured and predicted signal over cycle graphs for each spot

```python
python dephased_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\S0189_color_transformed_spots.csv
    -o \Users\akshitapanigrahi\Desktop
    -grid
    -plots
```

## Sample Basecall Output

The program outputs a dataframe of the post color transform base calls, post dephased base calls for each spot, and the cumulative predicted error. The table is stored to a CSV.



The program generates the base call histogram for each of the spots:
![newplot](https://github.com/454bio/tools_playground/assets/129779339/9af874f5-c434-4808-849e-2afadbcdf1f8)

