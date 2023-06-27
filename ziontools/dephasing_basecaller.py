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
from common import get_cycle_files, default_base_color_map, default_spot_colors
import os

import math

verbose = 0
wantPlots = False
gridsearch = False

# if set to true, we will perform a fit across the measured data, and correct out any signal loss
# note that the dr (droop) param is then set to 0.0 for the model predictions
correctLoss = False

bases = ['G', 'C', 'A', 'T']
base_colors = ['green', 'yellow', 'blue', 'red']

# ie - the percent of product that does not incorporate
# cf - the percent of product that can incorpporate subsequent positions during UV cleavage
# dr - ther percent of product lost at each UV cycle (modeled as a system-wide param)
ie = 0.09
cf = 0.065
dr = 0.02

spot_data = ''
output_directory_path = ''

# easy to create alternate models to represent the physical system
# state_model selects which one to use
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

#
# CallBases
#
# Uses a physical model to predict what the measured signal would be after a UV cycle is applied
# performs predictions with an initially blank DNA template, so on the first pass only
# incompletion and signal loss (droop) can be accounted for.  On the second and subsequent
# passes, the model is able to improve signal predictions because it has a rough idea of the DNA
# template being sequenced, carry-forward in particual can now be accounted for correctly
#

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
spots = df['spot'].unique()

spot_arrays = []
spot_dataframes = []

undephased_basecalls = []

for spot in spots:
    spot_df = df[df['spot'] == spot]
    counts = spot_df[['G', 'C', 'A', 'T']].values
    
    undephased_basecall = ''
    for i in counts: 
        undephased_basecall += (bases[np.argmax(i)])
    
    spot_arrays.append(counts)
    undephased_basecalls.append(undephased_basecall)

# Perform base call for each spot
for i, spot_data in enumerate(spot_arrays):
    numCycles = spot_data.shape[0]
    print('Spot: ' + spots[i] + ', #cycles: %d' % numCycles)

    if gridsearch:
        ie,cf,dr = GridSearch(spot_data, spots[i])
    
    if correctLoss:
        measuredSignal = CorrectSignalLoss(spot_data)
        dr = 0.0
    else:
        measuredSignal = spot_data
    
    results = CallBases(ie, cf, dr, numCycles, measuredSignal)
    
    print('cumulative error: %f' % results['err'])
    print('')
        
    # Create a new DataFrame for the spot
    spot_row = pd.DataFrame({'Spot': [spots[i]],
                              'NumCycles': [numCycles],
                              'Basecalls Post Color Transformation': undephased_basecalls[i],
                              'Basecalls Post Dephasing': [results['basecalls']],
                              'CumulativeError': [results['err']]})
    
    if (spots[i] == 'BG'):
        spot_row = pd.DataFrame({'Spot': [spots[i]],
                                  'NumCycles': [numCycles],
                                  'Basecalls Post Color Transformation': undephased_basecalls[i],
                                  'Basecalls Post Dephasing': [results['basecalls']],
                                  'CumulativeError': [results['err']]})

    
    # Append the spot DataFrame to the spot_dataframes list
    spot_dataframes.append(spot_row)
    
    # plots
    if wantPlots:
        fig, ax = plt.subplots()
        fig.suptitle(spots[i] + ': predicted signals per cycle')
        for cycle in range(numCycles):
            for base in range(4):
                ax.bar(cycle + base*0.1, results['intensites'][cycle, base], color = base_colors[base], width = 0.1)
        plt.plot(range(numCycles), results['signal'], label='total intensity')
        plt.plot(range(numCycles),  results['error'], label='error')
        plt.legend(loc="upper right")
    
    
        totalMeasuredSignal = np.sum(measuredSignal, axis=1)
        fig, ax = plt.subplots()
        fig.suptitle(spots[i] + ': measured signals per cycle')
        for cycle in range(numCycles):
            for base in range(4):
                ax.bar(cycle + base*0.1, measuredSignal[cycle, base], color = base_colors[base], width = 0.1)
        plt.plot(range(numCycles), totalMeasuredSignal, label='total intensity')
        plt.show()

# Create an empty data frame
df = pd.DataFrame(columns=['spot', 'cycle', 'G', 'C', 'A', 'T'])

# Perform base call for each spot
for i, spot_data in enumerate(spot_arrays):
    numCycles = spot_data.shape[0]
    spot_name = spots[i]
    print('Spot: ' + spot_name + ', #cycles: %d' % numCycles)

    if gridsearch:
        ie, cf, dr = GridSearch(spot_data, spot_name)
    
    if correctLoss:
        measuredSignal = CorrectSignalLoss(spot_data)
        dr = 0.0
    else:
        measuredSignal = spot_data
    
    results = CallBases(ie, cf, dr, numCycles, measuredSignal)
    print('')
    
    # Create a data frame for the spot's cycle data
    cycle_df = pd.DataFrame({
        'spot': [spot_name] * numCycles,
        'cycle': np.arange(1, numCycles + 1),
        'G': results['intensites'][:, 0],
        'C': results['intensites'][:, 1],
        'A': results['intensites'][:, 2],
        'T': results['intensites'][:, 3]
    })
    
    # Append the cycle data to the main data frame
    df = pd.concat([df, cycle_df], ignore_index=True)
    
# Print the final data frame
#print(df.to_string(index=False))

unique_spot_names = list(df['spot'].unique())

spot_names = []
spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('T')))
spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('A')))
spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('C')))
spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('G')))

s_list = [a for a in unique_spot_names if a.startswith('S')]
s_list.sort(key=lambda v: int(v.strip('S')))
x_list = [a for a in unique_spot_names if a.startswith('X')]
x_list.sort(key=lambda v: int(v.strip('X')))
spot_names.extend(s_list)
spot_names.extend(x_list)
spot_names.append(unique_spot_names.pop(unique_spot_names.index('BG')))

# fixed order
'''
spot_names = [
    'G', 'C', 'A', 'T',
    'S1', 'S2', 'S3', 'S4',
    'S5', 'S6', 'S7', 'S8',
    'S9', 'S10', 'S11', 'S12',
    'S13', 'S14', 'S15', 'S16',
    'S17', 'S18', 'S19', 'S20',
    'X1', 'X2', 'X3', 'BG'
]
'''

cols = 4
fig = make_subplots(
    rows=math.ceil(len(spot_names)/cols), cols=cols
)

dye_bases = ["G", "C", "A", "T"]

for i, spot_name in enumerate(spot_names):

    r = (i // cols)+1
    c = (i % cols)+1
    
    df_spot = df.loc[(df['spot'] == spot_name)]

    # Add traces
    for base_spot_name in dye_bases:
        fig.add_trace(
            # Scatter, Bar
            go.Bar(
                x=df_spot['cycle'],
                y=df_spot[base_spot_name],
                name=base_spot_name,
                marker_color=default_base_color_map[base_spot_name],
                legendgroup=base_spot_name, showlegend=(i == 0)
            ),
            row=r, col=c
        )

    fig.add_trace(
        # Scatter, Bar
        go.Scatter(
            x=df_spot['cycle'],
            y=df_spot['G']/1000000+1,
            text=df_spot[dye_bases].idxmax(axis=1),  # column with the highest value
            marker_color="black",
            mode="text",
            textposition="top center",
            textfont_size=26,
            showlegend=False
        ),
        row=r, col=c
    )


    fig.update_xaxes(
        title_text=spot_name,
        title_font={"size": 24},
        row=r, col=c)

    fig.update_yaxes(range=[-0.2, 1.2], row=r, col=c)

fig.update_layout(height=3000, width=3000,
                  title_text='Dephased Basecalls')

fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size=40)
                              ))

fig.write_image(os.path.join(output_directory_path, "dephased_basecalls.png"), scale=1.5)

fig.show()

# Reorder the rows based on spot_names list
df['spot'] = pd.Categorical(df['spot'], categories=spot_names, ordered=True)
df['cycle'] = df['cycle'].astype(int)  # Convert cycle column to integer for proper sorting
df = df.sort_values(['spot', 'cycle']).reset_index(drop=True)

# Save the reordered data frame to a CSV file
output_file_path = Path(output_directory_path) / "dephased_spots.csv"
df.to_csv(output_file_path, index=False)

# Concatenate all spot DataFrames into a single DataFrame
result_df = pd.concat(spot_dataframes, ignore_index=True)

# Reorder the rows based on spot_names list
result_df['Spot'] = pd.Categorical(result_df['Spot'], categories=spot_names, ordered=True)
result_df = result_df.sort_values('Spot').reset_index(drop=True)

# Save the reordered data frame to a CSV file
output_file_path = Path(output_directory_path) / "dephased_basecalls.csv"
result_df.to_csv(output_file_path, index=False)

print(result_df.to_string(index=False))
