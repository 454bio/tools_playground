import model
import model_dark
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from pathlib import Path  
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import oligo_sequences, get_cycle_files, default_base_color_map, default_spot_colors
import os

import math

verbose = 0
wantPlots = False
gridsearch = False
correctLoss = False
compare = False

bases = ['G', 'C', 'A', 'T']
base_colors = ['green', 'yellow', 'blue', 'red']

ie = 0.09
cf = 0.065
dr = 0.02

spot_data = '/Users/akshitapanigrahi/Desktop/S0228color_transformed_spots.csv'
output_directory_path = '/Users/akshitapanigrahi/Desktop'
state_model = 'default' # [default,dark]

argc = len(sys.argv)
argcc = 1
while argcc < argc:
    if sys.argv[argcc] == '-ie':
        argcc += 1
        ie = float(sys.argv[argcc])
    if sys.argv[argcc] == '-cf':
        argcc += 1
        cf = float(sys.argv[argcc])
    if sys.argv[argcc] == '-dr':
        argcc += 1
        dr = float(sys.argv[argcc])
    if sys.argv[argcc] == '-v':
        verbose += 1
    if sys.argv[argcc] == '-plots':
        wantPlots = True
    if sys.argv[argcc] == '-grid':
        gridsearch = True
    if sys.argv[argcc] == '-loss':
        correctLoss = True
    if sys.argv[argcc] == '-compare':
        compare = True
    if sys.argv[argcc] == '-model':
        argcc += 1
        state_model = sys.argv[argcc]
    if sys.argv[argcc] == '-spots':
        argcc += 1
        spot_data = sys.argv[argcc]
    if sys.argv[argcc] == '-o':
        argcc += 1
        output_directory_path = sys.argv[argcc]
    argcc += 1

def CallBases(ie, cf, dr, numCycles, measuredSignal):
    dnaTemplate = ''
    if state_model == 'dark':
        m = model_dark.ModelDark()
    else:
        m = model.Model()
    m.SetParams(ie=ie, cf=cf, dr=dr)
    cumulativeError = 0
    dyeIntensities = np.zeros((numCycles, 4))
    totalSignal = np.zeros(numCycles)
    errorPerCycle = np.zeros(numCycles)

    # perform base calling
    numIterations = 3
    for iteration in range(numIterations):
        m.Reset()
        for cycle in range(numCycles):
            best_error = 0
            best_base = -1
            best_signal = 0

            for base in bases:
                # insert a "what-if" base at the current position, and predict what the signal looks like
                testTemplate = dnaTemplate[:cycle] + base + dnaTemplate[cycle+1:]
                
                # the model will return the predicted signal across all 4 dyes
                # plus an unknown and extra component that the model is also tracking
                # for example when we are beyond the end of the template, signals get bucketed up differently at the end
                signal = m.GetSignal(testTemplate)
                signalSum = np.sum(signal[:4]) # total intensity of the 4 dyes

                # compare to measured at this cycle across all 4 dyes
                error = 0
                for i in range(4):
                    delta = (measuredSignal[cycle][i] - signal[i])/signalSum
                    error += delta*delta

                # keep track of the lowest error, this is the best predition
                if error < best_error or best_base == -1:
                    best_base = base
                    best_error = error
                    best_signal = signal
            
                        # append/replace with best base at current position (cycle)
            dnaTemplate = dnaTemplate[:cycle] + best_base + dnaTemplate[cycle+1:]
            dyeIntensities[cycle] = best_signal[:4]
            totalSignal[cycle] = np.sum(best_signal[:4])
            errorPerCycle[cycle] = best_error

            # update the model - note that we do this after getting the measured signals, because this matches the physical
            # system where the first base is exposed to nucleotides prior to UV cleavage
            m.ApplyUV(numCycles)

        if verbose > 0:
            print('iteration %d basecalls: %s' % (iteration, dnaTemplate))

    print('basecalls: %s' % dnaTemplate)
    cumulativeError = np.sum(errorPerCycle)
    return {'err':cumulativeError, 'basecalls':dnaTemplate, 'intensites':dyeIntensities, 'signal':totalSignal, 'error':errorPerCycle}

def CorrectSignalLoss(measuredSignal):
    totalMeasuredSignal = np.sum(measuredSignal, axis=1)
    loss_dim = 2 # 1 for linear, 2 for quadratic, etc
    X = np.arange(len(totalMeasuredSignal))
    coef = np.polyfit(X, totalMeasuredSignal, loss_dim)
    print('measured loss: %s' % coef)
    fit_fn = np.poly1d(coef)
    lossCorrectedSignal = np.copy(measuredSignal)
    for cycle in range(len(totalMeasuredSignal)):
        lossCorrectedSignal[cycle] /= fit_fn(cycle)
    return lossCorrectedSignal

def GridSearch(data, spot_name):
    cf1 = 0.05
    cf2 = 0.08
    cfnum = 11
    ie1 = 0.07
    ie2 = 0.11
    ienum = 11
    dr1 = 0.01
    dr2 = 0.025
    drnum = 4

    minerr = 99999
    bestie = 0
    bestcf = 0
    bestdr = 0
    
    for cf in np.linspace(cf1, cf2, cfnum):
        for ie in np.linspace(ie1, ie2, ienum):
            for dr in np.linspace(dr1, dr2, drnum):
                res = CallBases(ie, cf, dr, numCycles, data)
                if res['err'] < minerr:
                    minerr = res['err']
                    bestie = ie
                    bestcf = cf
                    bestdr = dr
                    
    print(spot_name + ': best err:%f ie:%f cf:%f dr:%f' % (minerr, bestie, bestcf, bestdr))
    return bestie, bestcf, bestdr

#Break up color transformed spot data into arrays of spot dye counts
df = pd.read_csv(spot_data, sep=',')
unique_spots = df.drop_duplicates(subset='spot_index')[['spot_index', 'spot_name']]
spot_indices = unique_spots['spot_index'].tolist()
spot_names = unique_spots['spot_name'].tolist()

spot_arrays = []
spot_dataframes = []

undephased_basecalls = []

for spot_index in spot_indices:
    spot_df = df[df['spot_index'] == spot_index]
    counts = spot_df[['G', 'C', 'A', 'T']].values
    
    undephased_basecall = ''
    for i in counts: 
        undephased_basecall += (bases[np.argmax(i)])
    
    spot_arrays.append(counts)
    undephased_basecalls.append(undephased_basecall)

# Remove NaN values using the math.isnan() function
spot_names = [x for x in spot_names if not (isinstance(x, float) and math.isnan(x))]
spot_indices = [x for x in spot_indices if not (isinstance(x, float) and math.isnan(x))]
spot_arrays = spot_arrays[:len(spot_names)]

# Perform base call for each spot

# Create an empty data frame
df = pd.DataFrame(columns=['spot_index', 'spot_name', 'cycle', 'G', 'C', 'A', 'T'])

for i, spot_data in enumerate(spot_arrays):
    numCycles = spot_data.shape[0]
    print('Spot Index: ' + str(spot_indices[i]) + ', Spot Name: ' + str(spot_names[i]) + ', #cycles: %d' % numCycles)

    if gridsearch:
        ie,cf,dr = GridSearch(spot_data, spot_names[i])
    
    if correctLoss:
        measuredSignal = CorrectSignalLoss(spot_data)
        dr = 0.0
    else:
        measuredSignal = spot_data
           
    results = CallBases(ie, cf, dr, numCycles, measuredSignal)
    
    print('cumulative error: %f' % results['err'])
    print('')
    
    if (compare):
        correct = ''
        if (spot_names[i] == 'BG'):
                    correct = ''
        elif (spot_names[i] not in list(oligo_sequences.keys())):
            oligo_num = input('What is oligo number for base spot ' + spot_names[i] + ' ')
            correct = oligo_sequences.get(oligo_num)[:numCycles]
        else:
            correct = oligo_sequences.get(spot_names[i])[:numCycles]

        # Calculate read length for color transformation
        color_transform_read_length = numCycles
        for j, (char1, char2) in enumerate(zip(correct, undephased_basecalls[i])):
            if char1 != char2:
                color_transform_read_length = j
                break

        # Calculate read length for post dephasing
        dephased_read_length = numCycles
        for j, (char1, char2) in enumerate(zip(correct, results['basecalls'])):
            if char1 != char2:
                dephased_read_length = j
                break

        # Determine the greater read length
        if color_transform_read_length > dephased_read_length:
            greater_read_length = 'CT'
        elif color_transform_read_length < dephased_read_length:
            greater_read_length = 'D'
        else:
            greater_read_length = 'E'
        
    if (compare):
        # Create a new DataFrame for the spot
        if spot_names[i] == 'BG':
            spot_row = pd.DataFrame({'Spot Index': [spot_indices[i]],
                                     'Spot Name': [spot_names[i]],
                                      'NumCycles': [numCycles],
                                      'Ground Truth Basecalls': 'N/A',
                                      'Basecalls Post Color Transformation': undephased_basecalls[i],
                                      'Basecalls Post Dephasing': [results['basecalls']],
                                      'Read Length: Color Transformation': 'N/A',
                                      'Read Length: Post Dephasing': 'N/A',
                                      'Greater Read Length': 'N/A',
                                      '#Differences: Ground Truth vs Color Transform': 'N/A',                                 
                                      '#Differences: Ground Truth vs Dephased': 'N/A',                                
                                      'Cumulative Error': [results['err']]})
        else:
            spot_row = pd.DataFrame({'Spot Index': [spot_indices[i]],
                         'Spot Name': [spot_names[i]],
                          'NumCycles': [numCycles],
                          'Ground Truth Basecalls': correct,
                          'Basecalls Post Color Transformation': undephased_basecalls[i],
                          'Basecalls Post Dephasing': [results['basecalls']],
                          'Read Length: Color Transformation': [color_transform_read_length],
                          'Read Length: Post Dephasing': [dephased_read_length],
                          'Greater Read Length': [greater_read_length],
                          '#Differences: Ground Truth vs Color Transform': sum(char1 != char2 for char1, char2 in zip(correct, undephased_basecalls[i])),
                          '#Differences: Ground Truth vs Dephased': sum(char1 != char2 for char1, char2 in zip(correct, [results['basecalls']][0])),
                          'Cumulative Error': [results['err']]})
    else:
        # Create a new DataFrame for the spot
        spot_row = pd.DataFrame({'Spot Index': [spot_indices[i]],
                                 'Spot Name': [spot_names[i]],
                                  'NumCycles': [numCycles],
                                  'Basecalls Post Color Transformation': undephased_basecalls[i],
                                  'Basecalls Post Dephasing': [results['basecalls']],
                                  'CumulativeError': [results['err']]})

    
    # Append the spot DataFrame to the spot_dataframes list
    spot_dataframes.append(spot_row)
        
    # Create a data frame for the spot's cycle data
    cycle_df = pd.DataFrame({
        'spot_index': [spot_indices[i]] * numCycles,
        'spot_name': [spot_names[i]] * numCycles,
        'cycle': np.arange(1, numCycles + 1),
        'G': results['intensites'][:, 0],
        'C': results['intensites'][:, 1],
        'A': results['intensites'][:, 2],
        'T': results['intensites'][:, 3]
    })
       
    # Append the cycle data to the main data frame
    df = pd.concat([df, cycle_df], ignore_index=True)
    
        # plots
    if wantPlots:
        fig, ax = plt.subplots()
        fig.suptitle('Spot ' + str(spot_indices[i])+ ': ' + str(spot_names[i]) + ' - predicted signals per cycle')
        for cycle in range(numCycles):
            for base in range(4):
                ax.bar(cycle + base*0.1, results['intensites'][cycle, base], color = base_colors[base], width = 0.1)
        plt.plot(range(numCycles), results['signal'], label='total intensity')
        plt.plot(range(numCycles),  results['error'], label='error')
        plt.legend(loc="upper right")
    
    
        totalMeasuredSignal = np.sum(measuredSignal, axis=1)
        fig, ax = plt.subplots()
        fig.suptitle('Spot ' + str(spot_indices[i])+ ': ' + str(spot_names[i]) + ' - measured signals per cycle')
        for cycle in range(numCycles):
            for base in range(4):
                ax.bar(cycle + base*0.1, measuredSignal[cycle, base], color = base_colors[base], width = 0.1)
        plt.plot(range(numCycles), totalMeasuredSignal, label='total intensity')
        plt.show()

cols = 4
fig = make_subplots(
    rows=math.ceil(len(spot_names)/cols), cols=cols
)

dye_bases = ["G", "C", "A", "T"]

# Reorder the rows based on spot_names list
df['spot_index'] = pd.Categorical(df['spot_index'], categories=spot_indices, ordered=True)
df['cycle'] = df['cycle'].astype(int)  # Convert cycle column to integer for proper sorting
df = df.sort_values(['spot_index', 'cycle']).reset_index(drop=True)

# Save the reordered data frame to a CSV file
output_file_path = Path(output_directory_path) / "dephased_spots.csv"
df.to_csv(output_file_path, index=False)

cols = 4
fig = make_subplots(
    rows=math.ceil(len(spot_names)/cols), cols=cols
)

# Iterate over each subplot and plot the data
for i, spot_index in enumerate(spot_indices):
    r = (i // cols) + 1
    c = (i % cols) + 1

    df_spot = df.loc[df['spot_index'] == spot_index]

    # Add traces
    for base_spot_name in dye_bases:
        fig.add_trace(
            # Scatter, Bar
            go.Bar(
                x=df_spot['cycle'],
                y=df_spot[base_spot_name],
                name=base_spot_name,
                marker_color=default_base_color_map[base_spot_name],
                legendgroup=base_spot_name,
                showlegend=(i == 0)
            ),
            row=r, col=c
        )

    fig.update_xaxes(
        title_text='Spot ' + str(spot_indices[i]) + ': ' + str(spot_names[i]),
        title_font={"size": 24},
        row=r, col=c
    )

    fig.update_yaxes(range=[-0.2, 1.2], row=r, col=c)

# Configure layout and save the figure
fig.update_layout(height=3000, width=3000, title_text='Predicted Dye Intensities')
fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size=40)))
fig.write_image(os.path.join(output_directory_path, "dephased_intensities.png"), scale=1.5)
fig.show()

# Concatenate all spot DataFrames into a single DataFrame
result_df = pd.concat(spot_dataframes, ignore_index=True)

# Save the reordered data frame to a CSV file
output_file_path = Path(output_directory_path) / "dephased_basecalls.csv"
result_df.to_csv(output_file_path, index=False)

print(result_df.to_string(index=False))

if (compare):
    numeric_color_transform_errors = pd.to_numeric(result_df['#Differences: Ground Truth vs Color Transform'], errors='coerce')
    color_transform_error_sum = int(numeric_color_transform_errors.sum())

    numeric_dephased_errors = pd.to_numeric(result_df['#Differences: Ground Truth vs Dephased'], errors='coerce')
    dephased_error_sum = int(numeric_dephased_errors.sum())

    greater_read_length_count = result_df[result_df['Greater Read Length'] == 'D'].shape[0]
    equal_length_count = result_df[result_df['Greater Read Length'] == 'E'].shape[0]
    colt_length_count = result_df[result_df['Greater Read Length'] == 'CT'].shape[0]
    
    print("Summary:")
    print(greater_read_length_count, " spots where dephasing produced greater read length")
    print(colt_length_count, " spots where non-dephasing produced greater read length")

    print("Total #Errors, Color Transform Basecalls: ", color_transform_error_sum)
    print("Total #Errors, Dephased Basecalls: ", dephased_error_sum)
