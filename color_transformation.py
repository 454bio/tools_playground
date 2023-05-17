import pandas as pd
import numpy as np
from sklearn import linear_model
import cv2 as cv
from common import *
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import ndimage
import roifile

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs.layout import YAxis, XAxis, Margin
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


def calculate_and_apply_transformation(df: pd.DataFrame, roizipfilepath: str):

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

    dye_bases = ["S1", "S2", "S3", "S4"]
    df = df[df.spot.isin(dye_bases)]

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

    df_files = get_cycle_files(inputpath)
    print(df_files)
#    first_files = df_files.loc[(df_files['file_info_nb'] >= image_number_645) & (df_files['file_info_nb'] < image_number_645+5)]

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

        cv.imwrite(str(i)+'_gray.png', (img+1)*100)

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


    # A488  Green   G
    # A532  Orange  C
    # CF594 Yellow  A
    # A647N Red     T
    #       Black   -

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


    # A488  Green   G
    # A532  Yellow  C
    # CF594 Orange  A
    # A647N Red     T

    colormap = {
        'B0': 'black', # background
        'BG': 'black', # background
        'SC': 'pink', # scatter
        'S1': 'green',   # 488  G
        'S2': 'yellow',  # 532  C
        'S3': 'orange',  # 594  A
        'S4': 'red',     # 647  T
    }

    fig = make_subplots(
        rows=4, cols=4
    )

    '''
    # Create figure with secondary x-axis
#    fig = go.Figure(layout=layout)
    layout = go.Layout(
        title="Basecalls, Spot " + oligo_spotname,
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

    for i, oligo_spotname in enumerate(df['spot'].unique()):

        r = (i // 4)+1
        c = (i % 4)+1

        df_spot = df.loc[(df['spot'] == oligo_spotname)]
        print(f"oligo spot: {i} , {oligo_spotname}  row={r}, col={c}")

        # Add traces
        for base_spot_name in dye_bases:
            fig.add_trace(
                # Scatter, Bar
                go.Bar(
                    x=df_spot['cycle'],
                    y=df_spot[base_spot_name],
                    name=base_spot_name,
                    marker_color=colormap[base_spot_name],
                ),
                row=r, col=c
            )
            fig.update_xaxes(title_text=oligo_spotname, row=r, col=c)

        fig.update_yaxes(range=[-0.1, 1.0], row=r, col=c)

    fig.update_layout(height=3000, width=3000,
                      title_text="Basecalls")

    fig.write_image("test_bar.png", scale=1.5)

    '''
    print(np.ones(df_spot.shape[0]) * 3)
    fig.add_trace(
        go.Scatter(x=('A','G','C','T','A','A','A','A','A','A'),
                   y=np.ones(df.shape[0]) * 0,

                   name="AGCT", xaxis='x2'),
    )
    '''

    fig.show()



def example_one_dye():
    unique_spots = df['spot'].unique()  # [:5] # TODO
    print(unique_spots)

    offset = 4096
    #offset = 0

    X = df.iloc[:,-15:].to_numpy()-offset
    print(type(X))
    print(X)


    # set vector y to 1 or 0
    y = np.where(df['spot'] == 'D488', 1, 0)
#        y = np.random.randint(10, size=(2505, 4))
    print(y)
    print(y.shape)
    print(sum(y))

    model_ols = linear_model.LinearRegression()
    reg = model_ols.fit(X, y)
    print(reg.coef_)
    coef = model_ols.coef_
    intercept = model_ols.intercept_
    print('coef= ', coef)
    print('intercept= ', intercept)



    # manual regression
    print("manual-----------------------------")
    A = np.ones((2505, 1))
    print(A.shape, X.shape)
    X = np.concatenate((A, X), axis=1)

    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(beta_hat)



    newdf = df[ (df['spot']=='D488') & (df['pixel_i']==35) ]
    print(newdf)
    v = newdf.iloc[:,-15:].to_numpy()-offset
    # v = np.array([[   5888,11728,6128,8432,17248,7664,12816,22528,10224,7600,6472,5056,5456,4616,4368  ]])
    print("automatic prediction-----------------------------")
    # disable scientif print mode , e.g. -e10
    np.set_printoptions(suppress=True)
    a = reg.predict(v)
    print("predict:", a)

    print("manual prediction-----------------------------")
    print("predict:", beta_hat[0] + np.dot(v,beta_hat[1:]))




'''
/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/00000002_001A_00002_645_C001_000004671.tif
/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/00000003_001A_00003_590_C001_000005838.tif
/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/00000004_001A_00004_525_C001_000010385.tif
/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/00000005_001A_00005_445_C001_000011162.tif
/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/00000006_001A_00006_365_C001_000014241.tif

~/454_Bio/tools_playground/triangle_extract.py -i . -r ~/454_Bio/runs/S108/RoiSet.zip -o ~/454_Bio/runs/S108/RoiSet.csv -s 27 -p 500
~/454_Bio/tools_playground/triangle_graph.py -i   ~/454_Bio/runs/S108/RoiSet.csv -o ~/454_Bio/runs/S108/RoiSet.png -g


'''
run = 2
if run == 1:
    inputpath = '/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/'
    spot_data_filename = "/home/domibel/454_Bio/runs/20230506_0911_S0102_0001/raws/out_limit.csv"

if run == 2:
    inputpath = '/home/domibel/454_Bio/runs/20230510_1517_dye1_test2_0001/raws'
    #spot_data_filename = '/home/domibel/454_Bio/runs/20230510_1517_dye1_test2_0001/raws/out.csv'
    spot_data_filename = '/home/domibel/454_Bio/runs/20230510_1517_dye1_test2_0001/raws/out_withBG2.csv'
    roizipfilepath = "/home/domibel/454_Bio/runs/20230510_1517_dye1_test2_0001/raws/RoiSetOligos.zip"

if run == 3:
    inputpath = '/home/domibel/454_Bio/runs/20230505_2111_S0101_0001/raws/'

if run == 4:
    inputpath = '/home/domibel/454_Bio/runs/20230501_1344_S0091_0001/raws/'

if run == 5:
    inputpath = '/home/domibel/454_Bio/runs/20230419_2037_S0079_0001/raws/'

if run == 6:
    inputpath = '/mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230510_1731_S0108R_0001/raws/'
    spot_data_filename = '/home/domibel/454_Bio/runs/S108/RoiSet.csv'


df = pd.read_csv(spot_data_filename)
print(df)


#example_one_dye()
calculate_and_apply_transformation(df, roizipfilepath)


'''
    # random image test
    R = np.random.rand(3, 2, n_features)*30000
    print(R)
    a = reg.predict(R.reshape(6, n_features))
    print(a)
    a = a.reshape(3, 2, n_targets)
    print(a.shape)
    print(a)
'''

