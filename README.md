# tools_playground

1. Extract basic metrics from ROIs

```
extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/Analysis/S0096_RoiSet.zip \
    -o S0096.csv
```
```
extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/Analysis/S0097_RoiSet.zip \
    -o S0097.csv
```
```
extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/Analysis/S0098_RoiSet.zip \
    -o S0098.csv
```
```
extract_roiset_metrics_to_csv.py \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/raws/ \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/Analysis/S0099_RoiSet5.zip \
    -o S0099.csv
```

```
plot_roiset_run_comparison.py -h

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o orig.jpg

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o normalized.jpg -n
```


2. Create triangle graphs
```
extract_roiset_pixel_data.py -i . -r RoiSet5.zip -o RoiSet5.csv -s 7 -p 500

triangle_graph.py -i RoiSet5.csv -o triangle.png -g -m 20 33.3 12

triangle_graph.py -i RoiSet5.csv -o triangle.png -g -c G445 G525 R590 B445

```


3. Create basic basecaller graph
```
extract_roiset_pixel_data.py -s 7 -p 500 \
    -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/raws \
    -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/RoiSetJRC4_28spotsACGT.zip \
    -o /tmp/roi_pixel_data.csv
    
color_transformation.py \
     -i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/raws \
     -r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/RoiSetJRC4_28spotsACGT.zip \
     -s /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/roi_pixel_data.csv \
     -o /tmp

cd /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001
color_transformation.py \
     -i raws \
     -r analysis/RoiSetJRC4_28spotsACGT.zip \
     -s analysis/roi_pixel_data.csv \
     -o /tmp
     -c G445 G525 R590 B445
```
