from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import pathlib
from itertools import product
import re

from .common import default_base_color_map, default_spot_colors

def plot_triangle(
        spot_data_filename: str,
        channel_subset: list[str],
        spot_names_subset: list[str],
        rgb_mix
) -> go.Figure:

    print(f"input_csv filename: {spot_data_filename}")

    # check if spot_data_filename exists
    if not pathlib.Path(spot_data_filename).is_file():
        print(f"Cannot open {spot_data_filename}")
        exit(-1)

    df = pd.read_csv(spot_data_filename)
    print(df)

    excitations = [365, 445, 525, 590, 645]
    image_channels = ['R', 'G', 'B']
    channels = [a[0] + str(a[1]) for a in product(image_channels, excitations)]
    if channel_subset:
        channels = channel_subset

    if rgb_mix:
        p = rgb_mix
        image_channels = ['M']
        # adding mixed columns
        for exc in excitations:
            # e.g.  df['M365'] = p['R']*df['R365'] + p['G']*df['G365'] + p['B']*df['B365']
            df['M' + str(exc)] = p['R'] * df['R' + str(exc)] + p['G'] * df['G' + str(exc)] + p['B'] * df[
                'B' + str(exc)]

    print(len(channels), "channels: ", channels)

    unique_spots = df['spot_name'].unique()
    if spot_names_subset:
        spots = [s for s in spot_names_subset if s in unique_spots]
        unique_spots = set(spots)

    # add random colors
    spot_color_map = default_base_color_map
    for i, s in enumerate(unique_spots):
        if s not in spot_color_map:
            spot_color_map[s] = default_spot_colors[i % 5]

    if spot_names_subset:
        df = df[df.spot_name.isin(spot_names_subset)]

    print(len(unique_spots), "unique_spots: ", unique_spots)

    # custom
    use_custom_max_intensity_map = False
    custom_max_intensity = {
        'R365': 10000, 'G365': 10000, 'B365': 10000,
        'R445': 10000, 'G445': 12000, 'B445': 10000,
        'R525': 25000, 'G525': 60000, 'B525': 15000,
        'R590': 30000, 'G590': 25000, 'B590': 10000,
        'R645': 35000, 'G645': 15000, 'B645': 10000,
    }
    # TODO, HACK
    if 'M445' in channels:
        min_intensity = 0
    else:
        min_intensity = 4000

    spot_indizes = df.spot_index.unique()

    print(f"Generate triangle subplots:")
    fig = make_subplots(
        rows=len(channels) - 1, cols=len(channels) - 1,  # e.g. 14x14
    )

    for r, y_channel in enumerate(channels):
        for c, x_channel in enumerate(channels):
            if c >= r:
                continue
            print(f"{y_channel}x{x_channel} ", end=" ", flush=True)

            for spot_index in spot_indizes:
                # one spot
                df_spot = df.loc[(df['spot_index'] == spot_index)]
                spot_name = df_spot.spot_name.unique()[0]

                fig.add_trace(
                    go.Scatter(
                        x=df_spot[x_channel],
                        y=df_spot[y_channel],
                        text='#' + df_spot['pixel_i'].astype(str) + '_y' + df_spot['r'].astype(str) + '_x' + df_spot[
                            'c'].astype(str),
                        marker_color=spot_color_map[spot_name],
                        marker=dict(size=2),
                        mode='markers',
                        name=str(spot_name),
                        legendgroup=s, showlegend=(r == 1 and c == 0)
                    ),
                    row=r, col=c + 1
                )

            #            if c == 0:
            fig.update_yaxes(title_text=y_channel, row=r, col=c + 1)
            #            if r == len(channels) - 1:
            fig.update_xaxes(title_text=x_channel, row=r, col=c + 1)
            if use_custom_max_intensity_map:
                fig.update_xaxes(range=[min_intensity, custom_max_intensity[x_channel]], row=r, col=c + 1)
                fig.update_yaxes(range=[min_intensity, custom_max_intensity[y_channel]], row=r, col=c + 1)
            else:
                fig.update_xaxes(range=[min_intensity, max(df[x_channel])], row=r, col=c + 1)
                fig.update_yaxes(range=[min_intensity, max(df[y_channel])], row=r, col=c + 1)
        #            print(f"max x {x_channel} : {max(df_spot[x_channel])}, max y {y_channel} : {max(df_spot[y_channel])}")
        print("-")  # end of row reached

    #    fig.update_xaxes(range=[4000, max(df_spot[x_channel])])
    #    fig.update_yaxes(range=[4000, max(df_spot[y_channel])])
    fig.update_layout(height=3000, width=3000,
                      title_text="")
    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                                  font=dict(size=40),
                                  itemsizing='constant'
                                  ))

    return fig
