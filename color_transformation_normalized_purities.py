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
import math
import os
import re
from pathlib import Path  


import common

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


def calculate_and_apply_transformation(df: pd.DataFrame, roizipfilepath: str, input_directory_path : str, output_directory_path : str, channel_names: list[str]):

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

    n_features = len(channel_names)
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

    df_files = common.get_cycle_files(input_directory_path)
    print(df_files)

    rois = roifile.ImagejRoi.fromfile(roizipfilepath)
    print(rois)

    rows_list = []

    nb_cycles = max(df_files['cycle'])
    print("cycles:", nb_cycles)
#    nb_cycles = 8

    for cycle in range(1, nb_cycles):
        image_map = {}

        print("Apply transformation matrix on:")
        cyclefilenames = (df_files[df_files['cycle'] == cycle]).tail(5)
        print("c:", cycle, cyclefilenames.to_string())

        for i, cyclefilename in cyclefilenames.iterrows():
            filenamepath = cyclefilename['filenamepath']
            wavelength = cyclefilename['wavelength']
            print("WL:", filenamepath, wavelength)
            image = cv.imread(filenamepath, cv.IMREAD_UNCHANGED)[:, :, ::-1]  # BGR to RGB, 16bit data

            # safe offset correction
            image[image < offset] = offset
            image -= offset

            print(f"RGB image shap: {image.shape}")
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
        
    #Normalization s.t. total base sum = 1    
    for i in range (len(rows_list)):       
        curr_sum = df['G'][i] + df['C'][i] + df['A'][i] + df['T'][i]
        df['G'][i] = df['G'][i] / curr_sum
        df['C'][i] = df['C'][i] / curr_sum
        df['A'][i] = df['A'][i] / curr_sum
        df['T'][i] = df['T'][i] / curr_sum

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


    colors = [
        common.default_base_color_map['BG'],
        common.default_base_color_map['G'],
        common.default_base_color_map['C'],
        common.default_base_color_map['A'],
        common.default_base_color_map['T'],
        'yellow', 'orange', 'magenta'
    ]
    scale = [0, 1, 2, 3, 4, 5, 6, 250]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(scale, len(colors))
    axs[5].imshow(mask, aspect="auto", cmap=cmap, norm=norm)


#    axs[5].imshow(mask, aspect="auto", cmap=cmap, norm=norm, extent=[0, 400, 0, 300])

#    x = np.random.normal(170, 10, 250)
#    axs[5].hist(x)

#    x = range(300)
#    axs[5].plot(x, x, '--', linewidth=5, color='firebrick')

#    plt.imshow(mask, aspect="auto", cmap=cmap, norm=norm)
    plt.show()

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


    unique_spot_names = list(df['spot'].unique())

    spot_names = []
    spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('T')))
    spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('A')))
    spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('C')))
    spot_names.insert(0, unique_spot_names.pop(unique_spot_names.index('G')))
    s_list = [a for a in unique_spot_names if a.startswith('S')]
    s_list.sort(key=lambda v: int(v.strip('S')))
    x_list = [a for a in unique_spot_names if a.startswith('X')]
    x_list.sort(key=lambda v: int(v.strip('X')))
    spot_names.extend(s_list)
    spot_names.extend(x_list)
    spot_names.append(unique_spot_names.pop(unique_spot_names.index('BG')))

    # fixed order
    '''
    spot_names = [
        'G', 'C', 'A', 'T',
        'S1', 'S2', 'S3', 'S4',
        'S5', 'S6', 'S7', 'S8',
        'S9', 'S10', 'S11', 'S12',
        'S13', 'S14', 'S15', 'S16',
        'S17', 'S18', 'S19', 'S20',
        'X1', 'X2', 'X3', 'BG'
    ]
    '''

    print(spot_names)

    cols = 4
    fig = make_subplots(
        rows=math.ceil(len(spot_names)/cols), cols=cols
    )
    
    
    purities = []
    
    for i, spot_name in enumerate(spot_names):

        r = (i // cols)+1
        c = (i % cols)+1

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
                    marker_color=common.default_base_color_map[base_spot_name],
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
                textfont_size=26,
                showlegend=False
            ),
            row=r, col=c
        )
        
        maxx =df_spot[dye_bases].max(axis=1)
        maximumum = round(maxx, 2)
        
        purities.append(maxx)
        
        fig.add_trace(
            # Scatter, Bar
            go.Scatter(
                x=df_spot['cycle'],
                y=df_spot['G']/1000000+1,
                text=maximumum,  # column with highest value
                marker_color="black",
                mode="text",
                textposition="middle center",
                textfont_size=13,
                showlegend=False
            ),
            row=r, col=c
        )
        
        
        max_value = df_spot[dye_bases].max().max()  # Calculate the minimum value for the current spot
        min_value = df_spot[dye_bases].min().min()  # Calculate the maximum value for the current spot
        
        upper_lim = 1.2
        lower_lim = -0.2
        
        if (max_value > upper_lim):
            upper_lim = max_value + 0.1
        
        if (min_value < lower_lim):
            lower_lim = min_value - 0.1
        
        max_value = str(max_value)[0:5]

        fig.update_xaxes(
            title_text=spot_name,
            title_font={"size": 24},
            row=r, col=c)

        fig.update_yaxes(range=[lower_lim, upper_lim], row=r, col=c)
        
        x_range = max(df_spot['cycle']) - min(df_spot['cycle'])  # Calculate the range of x-axis values
        x_center = x_range / 2 + min(df_spot['cycle'])  # Calculate the x-coordinate for the center
            

    fig.update_layout(height=3000, width=3000,
                      title_text=input_directory_path)

    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                                  font=dict(size=40)
                                  ))

    fig.write_image(os.path.join(output_directory_path, "bar.png"), scale=1.5)
    
    fig.show()
    
    new_purities = np.reshape(purities, (np.shape(purities)[0] * np.shape(purities)[1]))
    df['purity'] = new_purities
    
    df.to_csv(Path(output_directory_path) / f"output.csv", index=False)
    
    return df


if __name__ == '__main__':

    test = 0

    if not test:
        parser = argparse.ArgumentParser(
            description='Color Transformation',
            epilog=''
        )

        parser.add_argument(
            "-i", "--input",
            required=True,
            action='store',
            dest='input_directory_path',
            help="Directory with .tif files, e.g.: /tmp/S001/raws/"
        )

        parser.add_argument(
            "-p", "--spot-pixel-data",
            required=True,
            action='store',
            dest='spot_data_filename',
            help="Spot pixel data file (with at least: C G A T BG), e.g.: /tmp/spot_pixel_data.csv"
        )

        parser.add_argument(
            "-r", "--roiset",
            required=True,
            action='store',
            dest='roiset_file_path',
            help="ImageJ RoiSet file, e.g.: /tmp/RoiSet.zip"
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

        # Output
        parser.add_argument(
            "-o", "--output",
            required=True,
            action='store',
            dest='output_directory_path',
            help="Output directory for .png and .csv files, e.g.: /tmp/S001/analysis/"
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


        if args.channel_subset:
            assert len(args.channel_subset) >= 2, "Please provide at least 2 channels"
            for ch in args.channel_subset:
                pattern = "^[R|G|B](\d{3})$"
                match = re.search(pattern, ch)
                if not match:
                    print(f"{ch} doesn't match format, e.g. R365")
                    exit(-1)

            channel_names = args.channel_subset
        else:
            channel_names = ['R365', 'G365', 'B365', 'R445', 'G445', 'B445', 'R525', 'G525', 'B525', 'R590', 'G590', 'B590', 'R645', 'G645', 'B645']

        spot_names_subset = None
        if args.spot_subset:
            spot_names_subset = list(set(args.spot_subset))

    else:

        input_directory_path = ''
        spot_data_filename = ''
        roiset_file_path = ''
        output_directory_path = ''
        channel_names = ['R365', 'G365', 'B365', 'R445']

    df = pd.read_csv(spot_data_filename)
    print(df)

    calculate_and_apply_transformation(df, roiset_file_path, input_directory_path, output_directory_path, channel_names)