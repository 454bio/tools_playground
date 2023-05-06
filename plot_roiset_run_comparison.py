#!/usr/bin/env python

import plotly.express as px
import argparse
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

test = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='plots run comparison',
        epilog='help'
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        action='store',
        type=argparse.FileType('r'),
        nargs='+',
        dest='input',
        default='roiset.csv',
        help="Input csv file"
    )

    parser.add_argument(
        '-n', '--normalize',
        action='store_true',
        dest='normalized',
        default=False,
        help="normalize data"
    )

    parser.add_argument(
        '-g', '--graph',
        action='store_true',
        dest='show_graph',
        default=False,
        help="plot into browser"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        type=argparse.FileType('w'),
        action='store',
        dest='output',
        default='.',
        help="output filename, e.g. plot.jpg"
    )

    args = parser.parse_args()

    normalized = args.normalized
    show_graph = args.show_graph

    runs = []
    for f in args.input:
        runs.append(f.name)
        print(f.name)

    outputfilename = args.output.name
    print(f"outputfilename: {outputfilename}")

    '''
    # check if inputfilename exists
    if not pathlib.Path(inputfilename).is_file():
        print(f"Cannot open {inputfilename}")
        exit(-1)
    df = pd.read_csv(inputfilename)
    print(df)
    '''

    dfs = [pd.read_csv(run) for run in runs]

    if test:
        excitations = [590, 645]
        image_channels = ['R', 'G']
        fig = make_subplots(
            rows=4, cols=4,
        )
    else:
        excitations = [445, 525, 590, 645] # 365
        image_channels = ['R', 'G', 'B']
        fig = make_subplots(
#            shared_yaxes=True,
#            shared_yaxes='all',
            rows=9, cols=15,
        )
    channels = list(product(image_channels, excitations))
    print(channels)

    spots = set(dfs[0]['spot'].unique())
    for df in dfs[1:]:
        spots = spots.intersection(set(df['spot'].unique()))
    spots = sorted(list(spots))

    line_colors = ['red', 'green', 'blue', 'black']

    for r, spot in enumerate(spots):
        for c, channel in enumerate(channels):
            print (r*c+c, spot, channel)
            for i, df in enumerate(dfs):
                dft = df.loc[(df['spot'] == spot) & (df['WL'] == channel[1])]

                # xaxis
                # X = dft['cycle']
                X = dft['TS']

                if normalized:
                    sig0 = dft[channel[0]+'avg'].iloc[0]
                    Y = (dft[channel[0]+'avg']-4096)/(sig0-4096)*100
                else:
                    Y = dft[channel[0]+'avg']

                fig.add_trace(
                    go.Scatter(
                        x=X, y=Y,
                        legendgroup=runs[i], showlegend=(r==0 and c==0),
                        marker_color=line_colors[i],
                        line_width=1,
                        marker=dict(size=2),
#                        mode='markers',
                        name=runs[i],
                        text=spot,
                    ),
                row=r + 1, col=c + 1)

            if 1:
#            if c == 0:
                fig.update_yaxes(title_text=spot, row=r + 1, col=c + 1)
#            if r == len(spots) - 1:
                fig.update_xaxes(title_text=(channel[0] + str(channel[1])), row=r + 1, col=c + 1)

#    fig.update_layout(yaxis = dict(range=[0, 2**15]))

#    fig.update_yaxes(range=[0, 36000])
#    fig.update_yaxes(range=[0, 1], dtick=0.2)
    fig.update_layout(height=2000, width=3000,
                      title_text="")

    plot1 = plot(fig, output_type='div')

    print(f"Writing {outputfilename}")

#    fig.write_image(outputfilename.replace(".csv", ".svg"))
    fig.write_image(outputfilename, scale=1.5)

    if show_graph:
        fig.show()

    print("exit")
