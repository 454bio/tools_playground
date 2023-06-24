import math
import os
import re
import pandas as pd
import numpy as np
from sklearn import linear_model
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import ndimage
import roifile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import get_cycle_files, default_base_color_map, default_spot_colors, oligo_sequences

dye_bases = ["G", "C", "A", "T"]


def get_roi_mask(shape, rois):

    mask = np.zeros(shape, dtype=np.uint8)
    print(f"mask: {type(mask)}, {mask.dtype}, {mask.shape}")

    for idx, roi in enumerate(rois):
        if __debug__:
            print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)
#            print(roi)

        yc = np.uint16(roi.top + (roi.bottom - roi.top) / 2)
        xc = np.uint16(roi.left + (roi.right - roi.left) / 2)
        x_axis_length = int((roi.right - roi.left) / 2.0)
        y_axis_length = int((roi.bottom - roi.top) / 2.0)

        label_id = idx + 1
        mask = cv.ellipse(mask, (xc, yc), [x_axis_length, y_axis_length], angle=0, startAngle=0, endAngle=360,
                          color=label_id, thickness=-1)

    # how many regions?
    nb_labels = len(rois)
    label_ids = np.arange(1, nb_labels + 1)  # range(1, nb_labels + 1)
    print(f"labels: {nb_labels}")

    sizes = ndimage.sum_labels(np.ones(mask.shape), mask, range(nb_labels + 1)).astype(int)
    print(f"number of pixels per label: {sizes}")

    return [mask, nb_labels]



def plot_bars(
        df: pd.DataFrame,
        title: str,
        spot_names_subset: list[str] = None
) -> go.Figure:
    if spot_names_subset:
        df = df[df.spot_name.isin(spot_names_subset)]

    spot_indizes = df.spot_index.unique()

    '''
    # Create figure with secondary x-axis
#    fig = go.Figure(layout=layout)
    layout = go.Layout(
        title="Basecalls, Spot " + spot_name,
        xaxis=XAxis(
            title="Cycles"
        ),
        xaxis2=XAxis(
            title="ACGT",
            overlaying='x',
            side='top',
        ),
        yaxis=dict(
            title="Y values"
        ),
    )
    '''

    cols = 4
    fig = make_subplots(
        rows=math.ceil(len(spot_indizes)/cols), cols=cols
    )

    for i, spot_index in enumerate(spot_indizes):

        r = (i // cols)+1
        c = (i % cols)+1

        df_spot = df.loc[(df['spot_index'] == spot_index)]
        spot_name = df_spot.spot_name.unique()[0]
        print(f"spot: {i}, idx: {spot_index}, name: {spot_name}, row={r}, col={c}")

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
            title_text=str(spot_index) + '  ' + spot_name + "  (" + oligo_sequences.get(spot_name, "")[:16] + ")",
            title_font={"size": 24},
            row=r, col=c
        )

        fig.update_yaxes(
            range=[-0.2, 1.2],
            row=r, col=c
        )

    fig.update_layout(
        height=3000,
        width=3000,
        title_text=title
    )

    fig.update_layout(
        legend=dict(title_font_family="Times New Roman",
                    font=dict(size=40)
                    )
    )

    return fig


def calculate_and_apply_transformation(
        spot_data_filename: str,
        roizipfilepath: str,
        input_directory_path: str,
        output_directory_path: str,
        channel_names_subset: list[str],
        spot_names_subset: list[str]
) -> pd.DataFrame:
    if channel_names_subset:
        assert len(channel_names_subset) >= 2, "Please provide at least 2 channels"
        for ch in channel_names_subset:
            pattern = "^[R|G|B](\d{3})$"
            match = re.search(pattern, ch)
            if not match:
                print(f"{ch} doesn't match format, e.g. R365")
                exit(-1)

        channel_names = channel_names_subset
    else:
        channel_names = ['R365', 'G365', 'B365', 'R445', 'G445', 'B445', 'R525', 'G525', 'B525', 'R590', 'G590', 'B590', 'R645', 'G645', 'B645']

    df = pd.read_csv(spot_data_filename)
    print(df)

    '''
        spot  pixel_i  timestamp_ms  cycle  R365  ...   G590   B590   R645   G645   B645
    0   BG          0         14241      1  4592  ...   5688   4736   5712   4984   4080
    1   BG          1         14241      1  4496  ...   5344   4784   6880   5136   4272
    70  SC         10         14241      1  5824  ...  65520  52528  65520  65520  35360
    71  SC         11         14241      1  4720  ...  65520  40736  65520  65520  29776
    '''

#    df = df[(df.spot != "B0")] # TODO
    # only look at these spots
#    df = df[df.spot.isin(["D1", "D2", "D3", "D4", "L0"])]

    df = df[df.spot_name.isin(dye_bases+['BG'])]

    n_features = len(channel_names)
    n_targets = len(dye_bases)

    offset = 4096
    BG_threshold = 25500
    SC_threshold = 64000*4

    unique_spot_names = df['spot_name'].unique()
    print("spots:", unique_spot_names)
    # make sure all dye_bases are in the spots_ist

    print("dye_bases:", dye_bases)

    dye_spot_to_index_map = {dye_spot: idx for idx, dye_spot in enumerate(dye_bases)}
    print(dye_spot_to_index_map)

    # intensity data
    X = df[channel_names].to_numpy()

    # camera offset correction
    X[X < offset] = offset
    X -= offset

    print("Datamatrix dimension:", X.shape)

    '''
    Generate Y
    1 0 0 0
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 1 0
    0 0 0 1
    0 0 0 1
    '''
    # (n_targets, n_features)
    # Y is a 2D matrix
    Y = np.zeros((len(df), n_targets))
    for i, base_spotname in enumerate(df['spot_name']):
        if base_spotname != "BG":
            Y[i, dye_spot_to_index_map[base_spotname]] = 1
#        else:
            #map to zero

    print(Y)

    model_ols = linear_model.LinearRegression()
    reg = model_ols.fit(X, Y)
    print(type(reg))
    print(reg.coef_)
    coef = model_ols.coef_
    intercept = model_ols.intercept_
    print('coef= ', coef)
    print('intercept= ', intercept)
#    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    print('coef= \n', coef*10000)
    print('intercept= ', intercept)


    # apply matrix to each cycle

    df_files = get_cycle_files(input_directory_path)
    print(df_files)

    rois = roifile.ImagejRoi.fromfile(roizipfilepath)
    print(rois)

    rows_list = []

    nb_cycles = max(df_files['cycle'])
    print("cycles:", nb_cycles)
#    nb_cycles = 8

    for cycle in range(1, nb_cycles+1):
        image_map = {}

        print("Apply transformation matrix on:")
        cyclefilenames = (df_files[df_files['cycle'] == cycle]).tail(5)
        print("cycle:", cycle, cyclefilenames.to_string())
        cycle_timestamp = cyclefilenames.iloc[0]['timestamp']

        for i, cyclefilename in cyclefilenames.iterrows():
            filenamepath = cyclefilename['filenamepath']
            wavelength = cyclefilename['wavelength']
            print("WL:", filenamepath, wavelength)
            image = cv.imread(filenamepath, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data

            # safe offset correction
            image[image < offset] = offset
            image -= offset

            print(f"RGB image shape: {image.shape}")
            image_map['R'+str(wavelength)] = image[:, :, 0]
            image_map['G'+str(wavelength)] = image[:, :, 1]
            image_map['B'+str(wavelength)] = image[:, :, 2]

        channels = [image_map[channel_name] for channel_name in channel_names]

        A = np.stack(channels, axis=2)
        dim = A.shape
        print(f"Matrix A shape: {dim}")
        assert (n_features == dim[2])

        # apply transformation to each pixel, reshape temporarily
        a = reg.predict(A.reshape(dim[0]*dim[1], n_features))
        # reshape back
        a = a.reshape(dim[0], dim[1], n_targets)
        print(f"Matrix a shape: {a.shape}")

        print("Transformation applied, shape", a.shape, type(a))

        oligo_mask, nb_labels = get_roi_mask((1520, 2028), rois)

#        nb_labels = len(rois)
        label_ids = np.arange(1, nb_labels + 1)  # range(1, nb_labels + 1)
        mean_list = []
        for i in range(n_targets):
            mean_list.append(ndimage.labeled_comprehension(a[:, :, i], oligo_mask, label_ids, np.mean, float, 0))

            img = a[:, :, i]
            cv.imwrite(os.path.join(output_directory_path, f"C{cycle:03d}_{dye_bases[i]}_{cycle_timestamp:09d}_gray.png"), (img+1)*100)
#            cv.imwrite(os.path.join(output_directory_path, f"C{cycle:03d}_{dye_bases[i]}_{cycle_timestamp:09d}_gray.tif"), img)


        for j, roi in enumerate(rois):
            if __debug__:
                print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)

            dict_entry = {
                'spot_index': j+1,
                'spot_name': roi.name,
                'cycle': cycle,
            }
            # base vector coefficients
            for i, base_spot_name in enumerate(dye_bases):
                dict_entry[base_spot_name] = mean_list[i][j]
            rows_list.append(dict_entry)


        # debug subplots
        if cycle == 1:

            fig, axs = plt.subplots(1, 5)

            for i in range(n_targets):
                img = a[:, :, i]
                print(f"min:  {img.min()}  , max: {img.max()}")

                cax_01 = axs[i].imshow(img, cmap='gray')
                fig.colorbar(cax_01, ax=axs[i])
                #        axs[i].xaxis.set_major_formatter(plt.NullFormatter())
                #        axs[i].yaxis.set_major_formatter(plt.NullFormatter())

                #    plt.show()

            mask = np.zeros(A.shape[:2], dtype=np.uint8)
            print(f"mask: {type(mask)}, {mask.dtype}, {mask.shape}")

            counts = [0] * n_targets
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    pixel = a[r, c]
                    label = np.array(pixel).argmax()
                    #            print(sum(A[r,c]), label)
                    #            if BG_threshold < sum(A[r,c]) and sum(A[r,c]) < SC_threshold:
                    mask[r, c] = label
                    #            else:
                    #                mask[r, c] = 5 # TODO

                    counts[label] += 1
            print("counts per base:", counts)

            colors = [
                default_base_color_map['BG'],
                default_base_color_map['G'],
                default_base_color_map['C'],
                default_base_color_map['A'],
                default_base_color_map['T'],
                'yellow', 'orange', 'magenta'
            ]
            scale = [0, 1, 2, 3, 4, 5, 6, 250]
            cmap = matplotlib.colors.ListedColormap(colors)
            norm = matplotlib.colors.BoundaryNorm(scale, len(colors))
            axs[4].imshow(mask, aspect="auto", cmap=cmap, norm=norm)

            #    axs[5].imshow(mask, aspect="auto", cmap=cmap, norm=norm, extent=[0, 400, 0, 300])

            #    x = np.random.normal(170, 10, 250)
            #    axs[5].hist(x)

            #    x = range(300)
            #    axs[5].plot(x, x, '--', linewidth=5, color='firebrick')

            #    plt.imshow(mask, aspect="auto", cmap=cmap, norm=norm)
            plt.show()

    # create final dataframe
    df_out = pd.DataFrame(rows_list)
    df_out.sort_values(by=['spot_index', 'spot_name', 'cycle'], inplace=True)
#           print(f"Writing {outputfilename}")
    df_out.to_csv(os.path.join(output_directory_path, "color_transformed_spots.csv"), index=False)
    print(df_out.to_string(index=False))

    return df_out

