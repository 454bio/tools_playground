#!/usr/bin/env python

import argparse
import ziontools

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
        dest='outputfile',
        help="output filename, e.g. plot.jpg"
    )

    args = parser.parse_args()

    normalized = args.normalized
    show_graph = args.show_graph

    runs = []
    for f in args.input:
        runs.append(f.name)
        print(f.name)

    outputfilename = args.outputfile.name
    print(f"outputfilename: {outputfilename}")

    '''
    # check if inputfilename exists
    if not pathlib.Path(inputfilename).is_file():
        print(f"Cannot open {inputfilename}")
        exit(-1)
    df = pd.read_csv(inputfilename)
    print(df)
    '''

    fig = ziontools.plot_roiset_run_comparison(
        runs
    )

#    plot1 = plot(fig, output_type='div')

    print(f"Writing {outputfilename}")

#    fig.write_image(outputfilename.replace(".png", ".svg"))
    fig.write_image(outputfilename, scale=1.5)

    if show_graph:
        fig.show()

    print("exit")
