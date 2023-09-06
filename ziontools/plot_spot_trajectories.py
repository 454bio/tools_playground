import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import default_base_color_map, default_spot_colors


def plot_spot_trajectories(
        input_metrics_filename: str,
        channel_names_subset: list[str],
        spot_names_subset: list[str]
) -> go.Figure:

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
        unique_spots = set(df['spot_name'].unique())

    print(len(channels), "channels: ", channels)
    print(len(unique_spots), "unique_spots: ", unique_spots)

    spot_color_map = default_base_color_map
    for i, s in enumerate(unique_spots):
        if s not in spot_color_map:
            spot_color_map[s] = default_spot_colors[i % 5]

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
                df_spot = df.loc[(df['spot_name'] == s)]
                df_x = df.loc[(df['spot_name'] == s) & (df['WL'] == int(x_channel[1:]))]
                df_y = df.loc[(df['spot_name'] == s) & (df['WL'] == int(y_channel[1:]))]

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
        height=2560,
        width=2560,
        title_text=""
    )
    fig.update_layout(
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(size=40),
            itemsizing='constant'
        )
    )

    return fig
