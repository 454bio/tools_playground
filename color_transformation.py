#!/usr/bin/env python

import argparse
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
from plotly.graph_objs.layout import YAxis, XAxis, Margin

from common import *

'''

'''


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


def calculate_and_apply_transformation(df: pd.DataFrame, roizipfilepath: str, input_directory_path : str, output_directory_path : str):

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

    dye_bases = ["G", "C", "A", "T"]
    df = df[df.spot.isin(dye_bases+['BG'])]

    n_features = 15
    n_targets = len(dye_bases)

    offset = 4096
    BG_threshold = 25500
    SC_threshold = 64000*4

    unique_df_spots = df['spot'].unique()
    print("spots:", unique_df_spots)
    # make sure all dye_bases are in the spots_ist

    print("dye_bases:", dye_bases)

    dye_spot_to_index_map = {dye_spot: idx for idx, dye_spot in enumerate(dye_bases)}
    print(dye_spot_to_index_map)

    # intensity data
    X = df.iloc[:, -n_features:].to_numpy()

    # camera offset correction
    X[X<offset]=offset
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
    for i, base_spotname in enumerate(df['spot']):
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

    for cycle in range(1, nb_cycles):
        lst = []

        print("Apply transformation matrix on:")
        cyclefilenames = (df_files[df_files['cycle'] == cycle]).tail(5)
        print("c:", cycle, cyclefilenames.to_string())

        for filenamepath in cyclefilenames['filenamepath']:
            print(filenamepath)
            image = cv.imread(filenamepath, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data

            # save offset correction
            image[image < offset] = offset
            image -= offset

            print(f"imageshape {image.shape}")
            lst.append(image)


        A = np.stack(
            (
            lst[4][:, :, 0],  # 365
            lst[4][:, :, 1],
            lst[4][:, :, 2],
            lst[3][:, :, 0],  # 445
            lst[3][:, :, 1],
            lst[3][:, :, 2],
            lst[2][:, :, 0],  # 525
            lst[2][:, :, 1],
            lst[2][:, :, 2],
            lst[1][:, :, 0],  # 590
            lst[1][:, :, 1],
            lst[1][:, :, 2],
            lst[0][:, :, 0],  # 645
            lst[0][:, :, 1],
            lst[0][:, :, 2],
            ), axis=2
        )
        dim = A.shape
        print(dim)
        assert (n_features == dim[2])

        # apply transformation to each pixel, reshape temporarily
        a = reg.predict(A.reshape(dim[0]*dim[1], n_features))
        print(a.shape)
        # reshape back
        a = a.reshape(dim[0], dim[1], n_targets)

        print("Transformation applied, shape", a.shape, type(a))

        oligo_mask, nb_labels = get_roi_mask((1520, 2028), rois)

#        nb_labels = len(rois)
        label_ids = np.arange(1, nb_labels + 1)  # range(1, nb_labels + 1)
        mean_list = []
        for i in range(n_targets):
            mean_list.append(ndimage.labeled_comprehension(a[:, :, i], oligo_mask, label_ids, np.mean, float, 0))


        for j, roi in enumerate(rois):
            if __debug__:
                print(roi.name, roi.top, roi.bottom, roi.left, roi.right, roi.roitype, roi.subtype, roi.options, roi.version, roi.props, roi.position)

            dict_entry = {
                'spot': roi.name.lstrip('spot'),
                'cycle': cycle,
            }
            # base vector coefficients
            for i, base_spot_name in enumerate(dye_bases):
                dict_entry[base_spot_name] = mean_list[i][j]
            rows_list.append(dict_entry)

    # create final dataframe
    df = pd.DataFrame(rows_list)
    df.sort_values(by=['spot', 'cycle'], inplace=True)
#           print(f"Writing {outputfilename}")
#           df.to_csv(outputfilename, index=False)
    print(df.to_string(index=False))

    '''
    spot1:
    A  G  C  T
                  c1
                  c2
                  c3
                  ..
                  c16    
    '''

    # debug
#    print(a)

    # Plot
    fig, axs = plt.subplots(1, 6)

    for i in range(n_targets):
        img = a[:, :, i]
        print(f"min:  {img.min()}  , max: {img.max()}")

        cax_01 = axs[i].imshow(img, cmap='gray')
        fig.colorbar(cax_01, ax=axs[i])
#        axs[i].xaxis.set_major_formatter(plt.NullFormatter())
#        axs[i].yaxis.set_major_formatter(plt.NullFormatter())

#    plt.show()

        cv.imwrite(os.path.join(output_directory_path, str(i)+'_gray.png'), (img+1)*100)
        cv.imwrite(os.path.join(output_directory_path, str(i)+'_gray.tif'), img)

        mask = np.zeros(A.shape[:2], dtype=np.uint8)
        print(f"mask: {type(mask)}, {mask.dtype}, {mask.shape}")


    counts = [0]*n_targets
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
    print(counts)


    # {'B000': 0, 'D488': 1, 'D532': 2, 'D594': 3, 'D647': 4, 'S000': 5}
    colors = ['green', 'yellow', 'orange', 'red', '#000000', 'blue', 'magenta']
    scale = [0, 1, 2, 3, 4, 5, 6, 250]
    cmap=matplotlib.colors.ListedColormap(colors)
    norm=matplotlib.colors.BoundaryNorm(scale, len(colors))
    axs[5].imshow(mask, aspect="auto", cmap=cmap, norm=norm)


#    axs[5].imshow(mask, aspect="auto", cmap=cmap, norm=norm, extent=[0, 400, 0, 300])

#    x = np.random.normal(170, 10, 250)
#    axs[5].hist(x)

#    x = range(300)
#    axs[5].plot(x, x, '--', linewidth=5, color='firebrick')

#    plt.imshow(mask, aspect="auto", cmap=cmap, norm=norm)
    plt.show()

    colormap = {
        'BG': 'black',  # background
        'SC': 'pink',   # scatter
        'G': 'green',   # 488
        'C': 'yellow',  # 532
        'A': 'orange',  # 594
        'T': 'red',     # 647
    }

    fig = make_subplots(
        rows=7, cols=4
    )

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

    spot_names = list(df['spot'].unique())
    # fixed order
    spot_names = [
        'G', 'C', 'A', 'T',
        'S1', 'S2', 'S3', 'S4',
        'S5', 'S6', 'S7', 'S8',
        'S9', 'S10', 'S11', 'S12',
        'S13', 'S14', 'S15', 'S16',
        'S17', 'S18', 'S19', 'S20',
        'X1', 'X2', 'X3', 'BG'
    ]

    print(spot_names)

    for i, spot_name in enumerate(spot_names):

        r = (i // 4)+1
        c = (i % 4)+1

        df_spot = df.loc[(df['spot'] == spot_name)]
        print(f"spot: {i} , {spot_name}  row={r}, col={c}")

        # Add traces
        for base_spot_name in dye_bases:
            fig.add_trace(
                # Scatter, Bar
                go.Bar(
                    x=df_spot['cycle'],
                    y=df_spot[base_spot_name],
                    name=base_spot_name,
                    marker_color=colormap[base_spot_name],
                    legendgroup=base_spot_name, showlegend=(i == 0)
                ),
                row=r, col=c
            )

        fig.add_trace(
            # Scatter, Bar
            go.Scatter(
                x=df_spot['cycle'],
                y=df_spot['G']/1000000+1,
                text=df_spot[dye_bases].idxmax(axis=1),  # column with highest value
                marker_color="black",
                mode="text",
                textposition="top center",
                textfont_size=30,
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
                      title_text=input_directory_path)

    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                                  font=dict(size=40)
                                  ))

    fig.write_image(os.path.join(output_directory_path, "bar.png"), scale=1.5)

    fig.show()


if __name__ == '__main__':

    test = 0

    if not test:
        parser = argparse.ArgumentParser(
            description='What the program does',
            epilog='Text at the bottom of help'
        )

        parser.add_argument(
            "-i", "--input",
            required=True,
            action='store',
            dest='input_directory_path',
            help="Input folder with .tif files"
        )

        parser.add_argument(
            "-s", "--spot_data",
            required=True,
            action='store',
            dest='spot_data_filename',
            help="Path to spot_data.csv"
        )

        parser.add_argument(
            "-r", "--roi",
            required=True,
            action='store',
            dest='roiset_file_path',
            help="roiset zipfile"
        )

        parser.add_argument(
            "-o", "--output",
            required=True,
            action='store',
    #        type=argparse.FileType('w'),
            dest='output_directory_path',
            help="output directory for .png and .csv files"
        )

        args = parser.parse_args()
        input_directory_path = args.input_directory_path
        print(f"input_directory_path: {input_directory_path}")

        output_directory_path = args.output_directory_path
        print(f"output_directory_path: {output_directory_path}")
        if not os.path.exists(output_directory_path):
            print(f"ERROR: output path {output_directory_path} doesn't exist")
            exit(-1)

        roiset_file_path = args.roiset_file_path
        print(f"roiset_file_path: {roiset_file_path}")

        spot_data_filename = args.spot_data_filename
        print(f"spot_data_filename: {spot_data_filename}")

    else:

        input_directory_path = ''
        spot_data_filename = ''
        roiset_file_path = ''
        output_directory_path = ''

    df = pd.read_csv(spot_data_filename)
    print(df)

    calculate_and_apply_transformation(df, roiset_file_path, input_directory_path, output_directory_path)
