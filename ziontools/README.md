# dephasing_basecaller

dephasing_basecaller.py - the basic app to perform base calling

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

Performs a grid-search over a hard-coded range of cf/ie/dr params, then performs phase correction and calls bases, on best cf/ie/dr params, for each spot

```python
python dephasing_basecaller.py \
    -spots \Users\akshitapanigrahi\Desktop\S0189_color_transformed_spots.csv
    -o \Users\akshitapanigrahi\Desktop
    -grid
```

Add the -plots parameter to display measured and predicted signal over cycle graphs for each spot

```python
python dephasing_basecaller.py 
    -ie 0.11 
    -cf 0.080 
    -dr 0.025 
    -spots \Users\akshitapanigrahi\Desktop\color_transformed_spots.csv 
    -o \Users\akshitapanigrahi\Desktop
    -plots
```

## Sample Basecall Output

The program outputs a dataframe of the post color transform base calls, post dephased base calls for each spot, and the cumulative predicted error. The table is stored to a CSV, 'dephased_basecalls.csv'

| Spot | NumCycles | Basecalls Post Color Transformation | Basecalls Post Dephasing | CumulativeError |
|------|-----------|------------------------------------|-------------------------|-----------------|
| G    | 10        | GTCAACTAAA                         | GTCAGCTACT              | 1.469654        |
| C    | 10        | CGTATCGACT                         | CGTATCGACT              | 0.752127        |
| A    | 10        | ACGTGCCAAT                         | ACGTGCTAGT              | 1.613753        |
| T    | 10        | TCAGTAAAAT                         | TCAGTACGAT              | 0.980266        |
| S1   | 10        | AAATAAAAAA                         | AAATGCAGTC              | 2.636655        |
| S2   | 10        | CCCGTATTAA                         | CCCGTATCAT              | 0.595073        |
| S3   | 10        | GGGACTCCAT                         | GGGACTCGAT              | 1.719469        |
| S4   | 10        | TTTTAATAAA                         | TTTCATATAC              | 1.058731        |
| S5   | 10        | GGGGGGGGGG                         | GGGGGGGGGG              | 13.814532       |
| S6   | 10        | AAAAAAAAAA                         | AAAAAAAAAA              | 12.618895       |
| S7   | 10        | CAACCCCTTT                         | CACTACTACT              | 4.540741        |
| S8   | 10        | TTTTTTTTTT                         | TTTATATCTA              | 2.725622        |
| S9   | 10        | ACGTGCCAAA                         | ACGTGCATAG              | 1.686096        |
| S10  | 10        | CGTATCGAAA                         | CGTATCGACT              | 0.745794        |
| S11  | 10        | GTCAAAAAAA                         | GTCAGCATAC              | 1.479951        |
| S12  | 10        | TCAAAAAAAA                         | TCAGTACATA              | 1.720662        |
| S13  | 10        | AAAAAAAAAA                         | AAATACAGAC              | 2.744884        |
| S14  | 10        | CCCGTAAAAA                         | CCCGTATCAT              | 0.791508        |
| S15  | 10        | GGGACTCAAA                         | GGGACTACAT              | 1.368890        |
| S16  | 10        | TTTAAAAAAA                         | TTTCAATACA              | 1.557294        |
| BG   | 10        | AAAATTTTTT                         | ATACTACTGC              | 4.939585        |


The program outputs a dataframe of the predicted intesities for each of the four bases over all cycles for all spots. The table is stored to a CSV, 'dephased_spots.csv'

| Spot | Cycle |    G    |    C    |    A    |    T    |
|------|-------|---------|---------|---------|---------|
|   G  |   10  | 0.000330| 0.226150| 0.086265| 0.488670|
|   G  |   7   | 0.057969| 0.261093| 0.153212| 0.413241|
|   G  |   6   | 0.222047| 0.468038| 0.072849| 0.140967|
|   G  |   5   | 0.523786| 0.163852| 0.211022| 0.023708|
|   G  |   4   | 0.116176| 0.200553| 0.603818| 0.020644|
|   G  |   3   | 0.017000| 0.705247| 0.091051| 0.147102|
|   G  |   2   | 0.088200| 0.054199| 0.003768| 0.833833|
|   G  |   1   | 1.000000| 0.000000| 0.000000| 0.000000|
|   G  |   8   | 0.011849| 0.215843| 0.366746| 0.270948|
|   G  |   9   | 0.002082| 0.346454| 0.227488| 0.262165|
|   C  |   1   | 0.000000| 1.000000| 0.000000| 0.000000|
|   C  |   7   | 0.407923| 0.260461| 0.153212| 0.063919|
|   C  |   6   | 0.140604| 0.463257| 0.072849| 0.227191|
|   C  |   9   | 0.078333| 0.346445| 0.227488| 0.185923|
|   C  |   10  | 0.020384| 0.226149| 0.086265| 0.468617|
|   C  |   5   | 0.023698| 0.131311| 0.211022| 0.556337|
|   C  |   3   | 0.147102| 0.008188| 0.091051| 0.714059|
|   C  |   4   | 0.020644| 0.016004| 0.603818| 0.300726|
|   C  |   2   | 0.833833| 0.088200| 0.003768| 0.054199|
|   C  |   8   | 0.229421| 0.215765| 0.366746| 0.053454|
|   A  |   9   | 0.332511| 0.016025| 0.227314| 0.262339| 
....

The program generates the base call histogram for each of the spots:
![newplot](https://github.com/454bio/tools_playground/assets/129779339/9af874f5-c434-4808-849e-2afadbcdf1f8)

