#!/usr/bin/env python

import argparse
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import pathlib
from itertools import product

test = 0

if __name__ == '__main__':
    """    requires multiple pixel for each spot    """

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='plots channel x channel triangle graph',
        epilog='help'
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        action='store',
        type=argparse.FileType('r'),
        dest='input',
        default='spot_pixel_roiset.csv',
        help="Input csv file"
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

    show_graph = args.show_graph

    inputfilename = args.input.name
    print(f"inputfilename: {inputfilename}")

    outputfilename = args.output.name
    print(f"outputfilename: {outputfilename}")

    # check if inputfilename exists
    if not pathlib.Path(inputfilename).is_file():
        print(f"Cannot open {inputfilename}")
        exit(-1)

    df = pd.read_csv(inputfilename)
    print(df)


    if test:
        excitations = [590, 645]
        image_channels = ['R', 'G']
    else:
        excitations = [365, 445, 525, 590, 645]
        image_channels = ['R', 'G', 'B']

    channels = list(product(image_channels, excitations))
    print(len(channels), "channels: ", channels)

    fig = make_subplots(
        rows=len(channels), cols=len(channels),  # e.g. 15x15
    )

    spot_colors = ['red', 'orange', 'green', 'black', 'blue', 'brown', 'white', 'black', 'magenta']
    #    colormap = {41:0, 21:1, 31:2, 51:3, 63:4}
    colormap = {
        'S0': 'black', # background
        'S1': 'green', # 488
        'S2': 'orange', # 532
        'S3': '#4000ff', # 594 color blueish, better than  pink or magenta?
        'S4': 'red', # 647
        'S5': 'pink', # scatter
        '11': 'black', 21: 'green', 31: 'yellow', 41: 'red', 51: 'blue', 63: 'pink'
    }

    unique_spots = df['spot'].unique()  # [:5] # TODO
    print(unique_spots)

    for r, tupy in enumerate(channels):
        print()
        for c, tupx in enumerate(channels):
            if c > r:
                continue
            print(tupy, tupx, end=" ", flush=True)

            for sidx, s in enumerate(unique_spots):
                # one spot
                dft = df.loc[(df['spot'] == s)]
                dfx = dft.loc[(dft['WL'] == tupx[1])]
                dfy = dft.loc[(dft['WL'] == tupy[1])]

                fig.add_trace(
                    go.Scatter(
                        x=dfx[tupx[0]],
                        y=dfy[tupy[0]],
                        text=dft['i'],
                        marker_color=spot_colors[sidx],
                        #marker_color=colormap[s],
                        marker=dict(size=2), mode='markers',
                        name='S' + str(s) + '_' + tupy[0] + str(tupy[1]) + '_' + tupx[0] + str(tupx[1])
                    ),
                    row=r + 1, col=c + 1
                )

            if c == 0:
                fig.update_yaxes(title_text=tupy[0] + str(tupy[1]), row=r + 1, col=c + 1)
            if r == len(channels) - 1:
                fig.update_xaxes(title_text=tupx[0] + str(tupx[1]), row=r + 1, col=c + 1)

    fig.update_layout(height=2000, width=3000,
                      title_text="")

    plot1 = plot(fig, output_type='div')

    print(f"Writing {outputfilename}")

#    fig.write_image(outputfilename.replace(".png", ".svg"))
    fig.write_image(outputfilename, scale=1.5)

    if show_graph:
        fig.show()

    print("exit")
