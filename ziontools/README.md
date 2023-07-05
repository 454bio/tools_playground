# dephasing_basecaller

dephasing_basecaller.py - the basic app to perform base calling, by Mel Davey, 2023. 

Does the following for each spot
1. loads the measured signals for each dye at the start of each cycle, from output of color_transform.py
2. instantiates a model to calculate expected signals based on phase params
3. applies the model to predict signals using a what-if strategy at each cycle, trying each base and seeing which was best

## Usage

The input 'spots' is the output csv file from color_transform.py

The output -o is the directory where the dephased basecalls, predicted signal intensities, and basecalls histogram are stored. 

Performs phase correction and calls bases for each spot, with default parameters

```python
python dephasing_basecaller.py 
    -ie 0.11 
    -cf 0.080 
    -dr 0.025 
    -spots \Users\akshitapanigrahi\Desktop\color_transformed_spots.csv 
    -o \Users\akshitapanigrahi\Desktop
```

Performs a comparison of basecalls with ground truth basecalls. If spot name is not an oligo number in oligo_sequences (from common.py), it will ask the user for the oligo number corresponding to spot. Inputted oligo number must be in the oligo_sequences map.

```python
python dephasing_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\S0228_color_transformed_spots.csv
    -o \Users\akshitapanigrahi\Desktop
    -compare
```

Performs a grid-search over a hard-coded range of cf/ie/dr params, then performs phase correction and calls bases, on best cf/ie/dr params, for each spot

```python
python dephasing_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\S0228_color_transformed_spots.csv
    -o \Users\akshitapanigrahi\Desktop
    -grid
```

Add the -plots parameter to display measured and predicted signal over cycle graphs for each spot

```python
python dephasing_basecaller.py 
    -ie 0.11 
    -cf 0.080 
    -dr 0.025 
    -spots \Users\akshitapanigrahi\Desktop\S0228_color_transformed_spots.csv 
    -o \Users\akshitapanigrahi\Desktop
    -plots
```

Any combination of these parameters can be sent/not sent. 

## Sample Basecall Output

The program outputs a dataframe of the post color transform base calls, post dephased base calls for each spot, and the cumulative predicted error. It also compares the color transform/dephased basecalls with the ground truth basecall. 

**All comparisons with ground truth are done POST running of dephasing model**

The table is stored to a CSV, 'dephased_basecalls.csv'

| Spot Index | Spot Name | NumCycles | Ground Truth Basecalls | Basecalls Post Color Transformation | Basecalls Post Dephasing | Read Length: Color Transformation | Read Length: Post Dephasing | Greater Read Length | #Differences: Ground Truth vs Color Transform | #Differences: Ground Truth vs Dephased | Cumulative Error |
|------------|-----------|-----------|-----------------------|-----------------------------------|-------------------------|-----------------------------------|-------------------------------|---------------------|-----------------------------------------|---------------------------------------|-----------------|
| 1          | A         | 16        | AAATGCAGTCGACGTA      | AAATTCAATTTCCCCC                 | AAATGCATCATCGTCA        | 4                                 | 7                             | D                   | 8                                       | 8                                     | 1.485814783     |
| 2          | C         | 16        | CCCGTATCGACTATGC      | CCCCTTTTTTCTTTTT                 | CCCGTATCGTCATCGT        | 3                                 | 9                             | D                   | 8                                       | 5                                     | 1.634730564     |
| 3          | G         | 16        | GGGACTCGATGCTCAG      | GGGGCTTCTTTTTTTT                 | GGGACTCGATCTGTAC        | 3                                 | 10                            | D                   | 9                                       | 5                                     | 1.298512122     |
| 4          | T         | 16        | TTTCAGTACGATGACT      | TTTTTTTTTTTTTTTT                 | TTTTCTATCGTCATGC        | 3                                 | 3                             | E                   | 10                                      | 11                                    | 2.065647962     |
| 5          | BG        | 16        | N/A                   | CCCCCCCCCCCCCCCC                 | CTGCTACGTCAGTCAG        | N/A                               | N/A                           | N/A                 | N/A                                     | N/A                                   | 6.23221263      |
| 6          | 370       | 16        | ACGTGACTAGTGCATC      | ACGTGCCTTTTTCCCC                 | ACGTGCTACTGCTACG        | 5                                 | 5                             | E                   | 6                                       | 10                                    | 2.862011622     |
| 7          | 373       | 16        | CGTATCGACTATGCAG      | CGTTTCGCCTTTTTTC                 | CGTATCGACTACTGCA        | 3                                 | 11                            | D                   | 7                                       | 5                                     | 2.039761671     |
| 8          | 377       | 16        | GTCAGCTACGACTGAT      | GTCCCCTTCCCCCCTT                 | GTCAGCTACTGCTAGC        | 3                                 | 9                             | D                   | 8                                       | 5                                     | 3.045921692     |
| 9          | 379       | 16        | TCAGTACGATGACTGC      | TCAGTTCTTTTTTTTT                 | TCAGTACGTACTGCTA        | 5                                 | 8                             | D                   | 8                                       | 8                                     | 2.301325666     |
....

If -compare was not sent as a parameter, the above table is produced without the ground truth, read lengths, and #differences columns. 

The program outputs a dataframe of the predicted signal intesities for each of the four dyes over all cycles for all spots. The table is stored to a CSV, 'dephased_intensities.csv'

| spot_index | spot_name | cycle | G          | C          | A           | T          |
|------------|-----------|-------|------------|------------|-------------|------------|
| 1          | A         | 1     | 0          | 0          | 1           | 0          |
| 1          | A         | 2     | 0          | 0          | 0.9694464   | 0.0055536  |
| 1          | A         | 3     | 0.012946128| 0.000709377| 0.833805322 | 0.103164173|
| 1          | A         | 4     | 0.126277486| 0.020621558| 0.238016786 | 0.541943545|
| 1          | A         | 5     | 0.459470095| 0.139130296| 0.076140402 | 0.228947098|
| 1          | A         | 6     | 0.232025657| 0.401793616| 0.153477448 | 0.093798973|
| 1          | A         | 7     | 0.073133448| 0.270834447| 0.355775175 | 0.159325231|
| 1          | A         | 8     | 0.018317515| 0.229395858| 0.270289324 | 0.319588888|
| 1          | A         | 9     | 0.005966614| 0.310981257| 0.232974086 | 0.266729564|
| 1          | A         | 10    | 0.013632799| 0.265810696| 0.279155068 | 0.237632057|
| 1          | A         | 11    | 0.050497072| 0.23888204 | 0.210063564 | 0.276835471|
| 1          | A         | 12    | 0.125578349| 0.26462214 | 0.112799144 | 0.253562998|
| 1          | A         | 13    | 0.193739496| 0.247142324| 0.064471694 | 0.230910798|
| 1          | A         | 14    | 0.174056304| 0.227086167| 0.085218506 | 0.227226231|
| 1          | A         | 15    | 0.108622077| 0.218426399| 0.177343918 | 0.182378596|
| 1          | A         | 16    | 0.052647081| 0.176516111| 0.314867322 | 0.113015864|
| 2          | C         | 1     | 0          | 1          | 0           | 0          |
| 2          | C         | 2     | 0.0055536  | 0.9694464  | 0           | 0          |
| 2          | C         | 3     | 0.103164173| 0.833774479| 0.000709
....

The program generates the predicted dye intensity histogram for each of the spots:
![dephased_intensities](https://github.com/454bio/tools_playground/assets/129779339/5062a876-7e75-4719-a8cf-ba62150c6b33)
