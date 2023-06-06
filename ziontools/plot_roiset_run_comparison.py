import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import pathlib
from itertools import product

'''
variations:
with/without 365
fixed y axis

'''
test = 1

def plot_roiset_run_comparison(runs_metrics_csv: list[str]):
    # inclusive offset correction
    dfs = [pd.read_csv(run) for run in runs_metrics_csv]

    for df in dfs:
        df['Ravg'] -= 4096
        df['Gavg'] -= 4096
        df['Bavg'] -= 4096

    spots = set(dfs[0]['spot'].unique())
    for df in dfs[1:]:
        spots = spots.intersection(set(df['spot'].unique()))
    spots = sorted(list(spots))
    print(f"spots: {len(spots)}, {spots}")

    if test:
        excitations = [590, 645]
        image_channels = ['R', 'G']
    else:
        excitations = [445, 525, 590, 645]  # 365
        image_channels = ['R', 'G', 'B']

    channels = list(product(image_channels, excitations))
    print(len(channels), "channels: ", channels)

    fig = make_subplots(
        #        shared_yaxes=True,
        #        shared_yaxes='all',
        rows=len(spots) * 3, cols=len(channels),  # e.g. 9x15
    )

    line_colors = ['red', 'green', 'blue', 'black']

    for r, spot in enumerate(spots):
        for c, channel in enumerate(channels):
            print("subplot: ", r * c + c, spot, channel)
            for i, df in enumerate(dfs):
                dft = df.loc[(df['spot'] == spot) & (df['WL'] == channel[1])]

                nb_cycles = max(df['cycle'])  # expensive TODO

                # xaxis
                # X = dft['cycle']
                X = dft['TS']
                channelname = channel[0] + 'avg'

                for h in range(3):

                    if h == 0:
                        Y = dft[channelname]
                    elif h == 1:  # if normalized 0-100:
                        sig_max = max(dft[channelname])  # max
                        sig_min = min(dft[channelname])  # min
                        Y = (dft[channelname] - sig_min) / (sig_max - sig_min) * 100
                    else:
                        Y = dft[[channelname, 'cycle']]
                        print(nb_cycles)
                        for cy in range(1, nb_cycles + 1):
                            dfa = Y.loc[df['cycle'] == cy, channelname]
                            #                            print(dfa[channelname])
                            sig_max = max(dfa)
                            sig_min = min(dfa)
                            Y.loc[df['cycle'] == cy, channelname] = (Y.loc[
                                                                         df['cycle'] == cy, channelname] - sig_min) / (
                                                                                sig_max - sig_min) * 100
                        Y = Y[channelname]

                    fig.add_trace(
                        go.Scatter(
                            x=X, y=Y,
                            legendgroup=runs_metrics_csv[i], showlegend=(r == 0 and c == 0),
                            marker=dict(
                                size=1,
                                symbol=34,
                                color=line_colors[i],
                                line=dict(width=1, color="DarkSlateGrey")
                            ),
                            mode="lines+markers",
                            name=runs_metrics_csv[i],
                            text=r,  # spot TODOdf
                        ),
                        row=2 * r + h + 1, col=c + 1
                    )

            if 1:
                #            if c == 0:
                fig.update_yaxes(title_text=spot, row=r + 1, col=c + 1)
                fig.update_yaxes(title_text="orig", secondary_y=True)
                #            if r == len(spots) - 1:
                fig.update_xaxes(title_text=(channel[0] + str(channel[1])), row=r + 1, col=c + 1)

    #    fig.update_layout(yaxis = dict(range=[0, 2**15]))

    #    fig.update_yaxes(range=[0, 36000])
    #    fig.update_yaxes(range=[0, 1], dtick=0.2)
    fig.update_layout(height=2000, width=3000,
                      title_text="")

    return fig


