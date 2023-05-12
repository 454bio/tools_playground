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

    parser.add_argument(
        "-m", "--mix",
        action='store',
        type=float,
        nargs='+',
        dest='rgb_mix',
        help="RGB percentages as float e.g. 33.3%"
    )


    args = parser.parse_args()

    show_graph = args.show_graph

    if args.rgb_mix:
        assert len(args.rgb_mix) == 3, "Please provide 3 floats"
        mix = True
        p = {
            'R': args.rgb_mix[0]/100,
            'G': args.rgb_mix[0]/100,
            'B': args.rgb_mix[2]/100
        }
        print(f'using rgb mix: {p}')
    else:
        mix = False

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

        if mix:
            image_channels = ['M']
            # adding mixed columns
            for exc in excitations:
                # e.g.  df['M365'] = p['R']*df['R365'] + p['G']*df['G365'] + p['B']*df['B365']
                df['M'+str(exc)] = p['R']*df['R'+str(exc)] + p['G']*df['G'+str(exc)] + p['B']*df['B'+str(exc)]

    channels = list(product(image_channels, excitations))
#    channels = [('G', 525), ('G', 590), ('G', 645), ('B', 365), ('B', 445), ('B', 525), ('B', 590), ('B', 645)]
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
        'S3': '#4000ff', # 594 color blueish
        'S4': 'red', # 647
        'S5': 'pink', # scatter
        '11': 'black', 21: 'green', 31: 'yellow', 41: 'red', 51: 'blue', 63: 'pink',
        'D488': 'green',
        'D532': 'orange',
        'D594': '#4000ff',
        'D647': 'red',
        'J000': 'black'
    }

    unique_spots = df['spot'].unique()  # [:5] # TODO
    print(unique_spots)

    for r, tupy in enumerate(channels):
        print()
        for c, tupx in enumerate(channels):
            if c > r:
                continue
            print(tupy, tupx, end=" ", flush=True)

            x_channel = tupx[0]+str(tupx[1])
            y_channel = tupy[0]+str(tupy[1])

            for sidx, s in enumerate(unique_spots):
                # one spot
                df_spot = df.loc[(df['spot'] == s)]

                fig.add_trace(
                    go.Scatter(
                        x=df_spot[x_channel],
                        y=df_spot[y_channel],
                        text=df_spot['pixel_i'],
                        marker_color=spot_colors[sidx],
                        #marker_color=colormap[s],
                        marker=dict(size=2), mode='markers',
                        name='S' + str(s) + '_' + y_channel + '_' + x_channel
                    ),
                    row=r + 1, col=c + 1
                )

            if c == 0:
                fig.update_yaxes(title_text=y_channel, row=r + 1, col=c + 1)
            if r == len(channels) - 1:
                fig.update_xaxes(title_text=x_channel, row=r + 1, col=c + 1)
            fig.update_xaxes(range=[4000, max(df[x_channel])], row=r + 1, col=c + 1)
            fig.update_yaxes(range=[4000, max(df[y_channel])], row=r + 1, col=c + 1)
#            print(f"max x {x_channel} : {max(df_spot[x_channel])}, max y {y_channel} : {max(df_spot[y_channel])}")

#    fig.update_xaxes(range=[4000, max(df_spot[x_channel])])
#    fig.update_yaxes(range=[4000, max(df_spot[y_channel])])
    fig.update_layout(height=3000, width=3000,
                      title_text="")

    plot1 = plot(fig, output_type='div')

    print(f"Writing {outputfilename}")

#    fig.write_image(outputfilename.replace(".png", ".svg"))
    fig.write_image(outputfilename, scale=1.5)

    if show_graph:
        fig.show()

    print("exit")
