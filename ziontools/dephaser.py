from .model import Model
from .model_dark import ModelDark
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import oligo_sequences

verbose = 0

bases = ['A', 'C', 'G', 'T']

# easy to create alternate models to represent the physical system
# state_model selects which one to use
state_model = 'default'  # [default,dark]

#
# CallBases
#
# Uses a physical model to predict what the measured signal would be after a UV cycle is applied
# performs predictions with an initially blank DNA template, so on the first pass only
# incompletion and signal loss (droop) can be accounted for.  On the second and subsequent
# passes, the model is able to improve signal predictions because it has a rough idea of the DNA
# template being sequenced, carry-forward in particular can now be accounted for correctly
#

def CallBases(ie, cf, dr, numCycles, measuredSignal):
    dnaTemplate = ''
    if state_model == 'dark':
        m = ModelDark()
    else:
        m = Model()
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

                # keep track of the lowest error, this is the best prediction
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

#    print('basecalls: %s' % dnaTemplate) # ie, cf, dr dependent
    cumulativeError = np.sum(errorPerCycle)
    return {'err': cumulativeError, 'basecalls': dnaTemplate, 'intensites': dyeIntensities, 'signal': totalSignal, 'error': errorPerCycle}

def CorrectSignalLoss(measuredSignal):
    totalMeasuredSignal = np.sum(measuredSignal, axis=1)
    loss_dim = 2  # 1 for linear, 2 for quadratic, etc
    X = np.arange(len(totalMeasuredSignal))
    coef = np.polyfit(X, totalMeasuredSignal, loss_dim)
    print('measured loss: %s' % coef)
    fit_fn = np.poly1d(coef)
    lossCorrectedSignal = np.copy(measuredSignal)
    for cycle in range(len(totalMeasuredSignal)):
        lossCorrectedSignal[cycle] /= fit_fn(cycle)
    return lossCorrectedSignal


def grid_search(numCycles, data):
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
    print('best err:%f ie:%f cf:%f dr:%f' % (minerr, bestie, bestcf, bestdr))
    return bestie, bestcf, bestdr


def dephase(color_transformed_filename: str, output_csv: str):

    df = pd.read_csv(color_transformed_filename)
    print(df)

    spot_indizes = df.spot_index.unique()
    print(spot_indizes)

    results_list = []
    for i, spot_index in enumerate(spot_indizes):

        print("--------------------------")
        df_spot = df.loc[(df['spot_index'] == spot_index)]
        spot_name = df_spot.spot_name.unique()[0]
        print(f"spot: {i}, idx: {spot_index}, name: {spot_name}")
        text = df_spot[bases].idxmax(axis=1)
#        print(text)
        max_call = text.str.cat(sep='')
#        print(df_spot)
#        print(f'max_call: {max_call}')

        A = df_spot[bases].to_numpy()
        A = np.round(A, 2)
        numCycles = len(A)
        ie, cf, dr = grid_search(numCycles, A)
        results = CallBases(ie, cf, dr, numCycles, A)
#        print(results)
#        print(results['basecalls'])
#        print('cumulative error: %f' % results['err'])

        expected_basecalls = oligo_sequences.get(spot_name)[:numCycles] if spot_name in oligo_sequences.keys() else ""
        dict_entry = {
            'spot_index': spot_index,
            'spot_name': spot_name,
            'cycles': numCycles,
            'expected_basecalls': expected_basecalls,
            'max_int_basecalls': max_call,
            'dephased_basecalls': results['basecalls'],
            'maxint_edit_distance': edit_distance(max_call, expected_basecalls),
            'dephased_edit_distance': edit_distance(results['basecalls'], expected_basecalls),
            'ie': ie,
            'cf': cf,
            'dr': dr,
        }
        results_list.append(dict_entry)

    df = pd.DataFrame(results_list)
    df.sort_values(by=['spot_index'], inplace=True)
    print(df)

    print(f"Writing {output_csv}")
    df.to_csv(output_csv, index=False)

    return df

# TODO, test
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]
