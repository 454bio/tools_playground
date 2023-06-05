#!/usr/bin/env python

import argparse
import sys
sys.path.insert(0, "/home/domibel/454_Bio/tools_playground/")
import ziontools


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
        dest='input_csv',
        help = "Spot pixel data file, e.g.: /tmp/spot_pixel_data.csv"
    )

    parser.add_argument(
        "-m", "--mix",
        action='store',
        type=float,
        nargs='+',
        dest='rgb_mix',
        help="RGB percentages as float e.g. -m 33.3 25 30.0"
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

    parser.add_argument(
        "-o", "--output",
        required=True,
        type=argparse.FileType('w'),
        action='store',
        dest='output_image',
        help="output image filename, e.g.: /tmp/plot.jpg"
    )

    parser.add_argument(
        '-g', '--graph',
        action='store_true',
        dest='show_graph',
        default=False,
        help="plot into browser"
    )

    args = parser.parse_args()

    outputfilename = args.output_image.name
    print(f"output_image filename: {outputfilename}")

    rgb_mix = None
    if args.rgb_mix:
        assert len(args.rgb_mix) == 3, "Please provide 3 floats"
        rgb_mix = {
            'R': args.rgb_mix[0] / 100,
            'G': args.rgb_mix[1] / 100,
            'B': args.rgb_mix[2] / 100
        }
        print(f'using rgb mix: {rgb_mix}')

    if args.channel_subset:
        assert len(args.channel_subset) >= 2, "Please provide at least 2 channels"
        for ch in args.channel_subset:
            pattern = "^[R|G|B](\d{3})$"
            match = re.search(pattern, ch)
            if not match:
                print(f"{ch} doesn't match format, e.g. R365")
                exit(-1)

    fig = ziontools.plot_triangle(
        args.input_csv.name,
        args.channel_subset,
        args.spot_subset,
        rgb_mix
    )

    print(f"Generate figure ...")
#    plot1 = plot(fig, output_type='div')

    print(f"Write {outputfilename} ...")
#    fig.write_image(outputfilename.replace(".png", ".svg"))
    fig.write_image(outputfilename, scale=1.5)

    if args.show_graph:
        fig.show()

    print("done")
