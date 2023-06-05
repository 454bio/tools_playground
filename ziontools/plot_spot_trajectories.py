#!/usr/bin/env python

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

import common


def plot_spot_trajectories(input_metrics_filename: str, channel_names_subset: list[str], spot_names_subset: list[str], output_image_filename: str):

    df = pd.read_csv(input_metrics_filename)
    print(df)

    default_channel_names = [
        'R365', 'G365', 'B365',
        'R445', 'G445', 'B445',
        'R525', 'G525', 'B525',
        'R590', 'G590', 'B590',
        'R645', 'G645', 'B645'
    ]

    if channel_names_subset:
        channels = channel_names_subset
    else:
        channels = default_channel_names

    if spot_names_subset:
        unique_spots = spot_names_subset
    else:
        unique_spots = set(df['spot'].unique())

    print(len(channels), "channels: ", channels)
    print(len(unique_spots), "unique_spots: ", unique_spots)

    spot_color_map = common.default_base_color_map
    for i, s in enumerate(unique_spots):
        if s not in spot_color_map:
            spot_color_map[s] = common.default_spot_colors[i % 5]

    print(f"Generate triangle subplots:")
    fig = make_subplots(
        rows=len(channels) - 1, cols=len(channels) - 1,  # e.g. 14x14
    )
    for r, y_channel in enumerate(channels):
        for c, x_channel in enumerate(channels):
            if c >= r:
                continue
            print(f"{y_channel}x{x_channel} ", end=" ", flush=True)
            fig.update_yaxes(title_text=y_channel, row=r, col=c + 1)
            fig.update_xaxes(title_text=x_channel, row=r, col=c + 1)

            for s in unique_spots:
                df_spot = df.loc[(df['spot'] == s)]
                df_x = df.loc[(df['spot'] == s) & (df['WL'] == int(x_channel[1:]))]
                df_y = df.loc[(df['spot'] == s) & (df['WL'] == int(y_channel[1:]))]

                fig.add_trace(
                    go.Scatter(
                        x=df_x[x_channel[0] + 'avg'],
                        y=df_y[y_channel[0] + 'avg'],
                        text=df_x['cycle'],
                        marker_color=spot_color_map[s],
                        marker=dict(size=9, symbol="arrow-bar-up", angleref="previous"),
                        mode='markers+lines',
                        name=str(s),
                        legendgroup=s, showlegend=(r == 1 and c == 0)
                    ),
                    row=r, col=c + 1
                )
        print("-")  # end of row reached

    fig.update_layout(
        height=3000,
        width=3000,
        title_text=""
    )
    fig.update_layout(
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(size=40),
            itemsizing='constant'
        )
    )

    # plot1 = plot(fig, output_type='div')

    if output_image_filename:
        fig.write_image(output_image_filename, scale=1.5)

    fig.show()


if __name__ == '__main__':

    test = 0

    if not test:
        parser = argparse.ArgumentParser(
            description='Color Transformation',
            epilog=''
        )

        parser.add_argument(
            "-i", "--spot-metrics",
            required=True,
            action='store',
            dest='spot_metrics_filename',
            help="Spot metrics e.g.: /tmp/metrics.csv"
        )

        parser.add_argument(
            "-c", "--channel_subset",
            action='store',
            type=str,
            nargs='+',
            dest='channel_subset',
            help="Channel subset e.g. -c G445 G525 R590 B445, Default: all 15 channels"
        )

        parser.add_argument(
            "-s", "--spot_subset",
            action='store',
            type=str,
            nargs='+',
            dest='spot_subset',
            help="Spot subset e.g. -s C G A T BG, Default: all spots in the RoiSet"
        )

        # Output png
        parser.add_argument(
            "-o", "--output",
            action='store',
            dest='output_image_filename',
            help="Output directory for .png , e.g.: /tmp/S001/trajectory.png"
        )

        args = parser.parse_args()

        spot_metrics_filename = args.spot_metrics_filename
        print(f"spot_metrics_filename: {spot_metrics_filename}")

        channel_names_subset = None
        if args.channel_subset:
            assert len(args.channel_subset) >= 2, "Please provide at least 2 channels"
            for ch in args.channel_subset:
                pattern = "^[R|G|B](\d{3})$"
                match = re.search(pattern, ch)
                if not match:
                    print(f"{ch} doesn't match format, e.g. R365")
                    exit(-1)

            channel_names_subset = args.channel_subset

        spot_names_subset = None
        if args.spot_subset:
            spot_names_subset = list(set(args.spot_subset))

        output_image_filename = args.output_image_filename
        print(f"output_image_filename: {output_image_filename}")
        if not output_image_filename:
            print(f"generate html graph only")

    else:
        spot_metrics_filename = ''
        output_image_filename = ''
        channel_names_subset = ['R365', 'G365', 'B365', 'R445']
        spot_names_subset = ['G', 'S1', 'S2']

    print(channel_names_subset)
    plot_spot_trajectories(spot_metrics_filename, channel_names_subset, spot_names_subset, output_image_filename)
