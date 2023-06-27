# basecall_spots

basecall_spots.py - the basic app to perform base calling, by Mel Davey, 2023

Does the following for each spot
1. loads the measured signals for each dye at the start of each cycle, from output of color_transform.py
2. instantiates a model to calculate expected signals based on phase params
3. applies the model to predict signals using a what-if strategy at each cycle, trying each base and seeing which was best

## Usage

Performs phase correction, calls bases, shows plots, for each spot

The input 'spots' is the output file from color_transform.py

```python
python basecall_spots.py 
-ie 0.09 
-cf 0.065 
-dr 0.02 
-spots \Users\akshitapanigrahi\Desktop\color_transformed_spots.csv 
-o \Users\akshitapanigrahi\Desktop -plots
```

Performs a grid-search over a hard-coded range of cf/ie/dr params, then performs base calling on best cf/ie/dr params, for each spot

```python
python basecall_spots.py -grid -plots
```

## Sample Basecall Output

The program outputs a dataframe of the ground truth basecalls, post color transform base calls, and post dephased base calls for each spot, and its cumulative error. It also displays the total number of differences. The table is stored to a CSV.

| Spot | NumCycles | Ground Truth Basecalls | Basecalls Post Color Transformation | Basecalls Post Dephasing | #Differences: Ground Truth vs Color Transform | #Differences: Ground Truth vs Dephased | Cumulative Error |
|------|-----------|-----------------------|-------------------------------------|-------------------------|---------------------------------------------|---------------------------------------|-----------------|
| A    | 10        | ATGCAGTCGA            | ATGCAGTCGC                          | ATGCAGTCGA              | 1                                           | 0                                     | 1.659113        |
| BG   | 10        | N/A                   | AGGTTTCCCC                          | AGTCGTCGTC              | N/A                                         | N/A                                   | 5.252732        |
| C    | 10        | CGTATCGACT            | CGTATCGACT                          | CGTATCGACT              | 0                                           | 0                                     | 1.255084        |
| G    | 10        | GACTCGATGC            | GACTCGTTTT                          | GACTCGATCT              | 3                                           | 2                                     | 1.076973        |
| S1   | 10        | ATGCAGTCGA            | ATGCAGTCGC                          | ATGCAGTCGA              | 1                                           | 0                                     | 1.587409        |
| S2   | 10        | CGTATCGACT            | CGTATCGCCT                          | CGTATCGACT              | 1                                           | 0                                     | 1.415804        |
| S3   | 10        | GACTCGATGC            | GACTCGTTTT                          | GACTCGATCT              | 3                                           | 2                                     | 1.122222        |
| S4   | 10        | TCAGTACGAT            | TCAGTACTTT                          | TCAGTACGAT              | 2                                           | 0                                     | 1.717042        |
| S5   | 10        | CGTATCGACT            | CCCCCCCCCC                          | CCTCTCCACT              | 7                                           | 3                                     | 1.864640        |
| S6   | 10        | ATGCAGTCGA            | ATGCAGTCCC                          | ATGCAGTCGC              | 2                                           | 1                                     | 1.931233        |
| S7   | 10        | TCAGTACGAT            | TCAGTACCTT                          | TCAGTACGTC              | 2                                           | 2                                     | 1.915852        |
| S8   | 10        | GACTCGATGC            | GACTCGTTTC                          | GACTCGTATC              | 2                                           | 3                                     | 1.392970        |
| T    | 10        | TCAGTACGAT            | TCAGTATTTT                          | TCAGTACGAT              | 3                                           | 0                                     | 1.635886        |

Total #Errors, Color Transform Basecalls:  27

Total #Errors, Dephased Basecalls:  10
