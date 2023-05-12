# tools_playground

extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/Analysis/S0096_RoiSet.zip \
    -o S0096.csv

extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/Analysis/S0097_RoiSet.zip \
    -o S0097.csv

extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/Analysis/S0098_RoiSet.zip \
    -o S0098.csv

extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/Analysis/S0099_RoiSet5.zip \
    -o S0099.csv

plot_roiset_run_comparison.py -h

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o orig.jpg

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o normalized.jpg -n





triangle_extract.py -i . -r RoiSet5.zip -o RoiSet5.csv -s 40 -p 4
triangle_graph.py -i RoiSet5.csv -o triangle.png -g -m 20 33.3 12
