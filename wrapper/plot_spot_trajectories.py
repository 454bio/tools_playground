#!/usr/bin/env python

import argparse
import re
import ziontools

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
        print(channel_names_subset)

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

    fig = ziontools.plot_spot_trajectories(
        spot_metrics_filename,
        channel_names_subset,
        spot_names_subset
    )


    # plot1 = plot(fig, output_type='div')

    if output_image_filename:
        fig.write_image(output_image_filename, scale=1.5)

    fig.show()
