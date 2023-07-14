import pandas as pd
import glob
import re
import os

default_spot_colors = [
    'orange',
    'brown',
    'magenta',
    'yellow',
    'lightblue',
    'sandybrown'
]

default_base_color_map = {
    'A': 'green',
    'C': 'blue',
    'T': 'red',
    'G': 'black',
    'BG': 'lightgrey'
}

oligo_sequences = {
    '357': 'ATGCAGTCGACGTACTATGCAGTCGACGTACT',
    '358': 'CGTATCGACTATGCAGCGTATCGACTATGCAG',
    '360': 'GACTCGATGCTCAGTAGACTCGATGCTCAGTA',
    '364': 'TCAGTACGATGACTGCTCAGTACGATGACTGC',
    '370': 'ACGTGACTAGTGCATCACGTGACTAGTGCATC',
    '372': 'ATGCAGTCGACGTACTATGCAGTCGACGTACT',
    '373': 'CGTATCGACTATGCAGCGTATCGACTATGCAG',
    '375': 'GACTCGATGCTCAGTAGACTCGATGCTCAGTA',
    '377': 'GTCAGCTACGACTGATGTCAGCTACGACTGAT',
    '379': 'TCAGTACGATGACTGCTCAGTACGATGACTGC',
    '574': 'GGGGGGGGGGTAAGAA',
    '575': 'AAAAAAAAAATAAGAA',
    '576': 'CCCCCCCCCCTAAGAA',
    '577': 'TTTTTTTTTTTAAGAA',
    '632': 'AAATGCAGTCGACGTACTATGCAGTC',
    '633': 'CCCGTATCGACTATGCAGCGTATCGA',
    '634': 'GGGACTCGATGCTCAGTAGACTCGAT',
    '635': 'TTTCAGTACGATGACTGCTCAGTACG',
    '648': 'TTTGCATTAAGAAATTAAAAAAGCTAAAAAAAAAA',
    '649': 'AAAGCATTAAGAAATTAAAAAAGCTAAAAAAAAAA',
    '650': 'GGGGCATTAAGAAATTAAAAAAGCTAAAAAAAAAA',
    '651': 'CCCGCATTAAGAAATTAAAAAAGCTAAAAAAAAAA',
    '662': 'TGACTAGAGCATACGG',
    '663': 'CGACTAGAGCATACGG',
    '664': 'GTCGTAGAGCATACGG',
    '665': 'AGACTAGAGCATACGG',
}

def get_cycle_files(inputpath: str) -> pd.DataFrame:

    file_names = sorted(glob.glob(inputpath + "/*tif", recursive=False))
    print(f"Found {len(file_names)} tif files.")
    assert len(file_names) >= 5, "not enough raw files"

    files_list = []
    for idx, filenamepath in enumerate(file_names):

        filename = os.path.basename(filenamepath)

        # extract file info
        filenameRegex = re.compile(r'(\d+)_(\d+A)_(\d+)_(\d+)_(C\d+)_(\d+).tif')
        match = filenameRegex.search(filename)
        if match:
#            print(match.groups(), type(match))
            file_info_nb = int(match.group(1))
            file_info_wl = int(match.group(4))
            file_info_cy = int(match.group(5).lstrip("C"))
            file_info_ts = int(match.group(6))

            # skip files with bad cycle infos
            if files_list and file_info_cy < files_list[-1]['cycle']:
                print(f"ERROR: unexpected cycle number {file_info_cy} for file: {filename}")
                continue

            dict_entry = {
                'file_info_nb': file_info_nb,'cycle': file_info_cy, 'wavelength': file_info_wl, 'timestamp': file_info_ts, 'filenamepath': filenamepath
            }
            files_list.append(dict_entry)
        else:
            print(f"ERROR Bad filename: {filename}")
            continue

        print(f"{idx}  {filename}  WL:{file_info_wl:03d}  CY:{file_info_cy}  TS:{file_info_ts }")

    df_files = pd.DataFrame(files_list)

#    print(df_files)
#    print(df_files.to_string())
    #    df_files.sort_values(by=['spot','cycle', 'TS'], inplace=True)
    assert df_files["cycle"].is_monotonic_increasing , "check the last few files"

    '''
    # extract end of run data
    # create new df TODO
    lst = []
    for cycle in df_files['cycle'].unique():
        df = df_files[df_files['cycle'] == cycle].tail(5)
        for index, row in df.iterrows():
            lst.append(list(row))

    df_ret = pd.DataFrame(lst, columns=df_files.columns)
    print(df_ret)
    return df_ret
    '''
    return df_files


