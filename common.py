import pandas as pd
import glob
import re
import os

def get_cycle_files(inputpath: str) -> pd.DataFrame:

    file_names = sorted(glob.glob(inputpath + "/*tif", recursive=False))
    print(f"Found {len(file_names)} tif files.")

    files_list = []
    for idx, filenamepath in enumerate(file_names):

        filename = os.path.basename(filenamepath)

        print(f"{idx}  {filename}")

        # extract file info
        filenameRegex = re.compile(r'(\d+)_(\d+A)_(\d+)_(\d+)_(C\d+)_(\d+).tif')
        match = filenameRegex.search(filename)
        if match:
            print(match.groups(), type(match))
            file_info_nb = int(match.group(3))
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

        print(f"WL:{file_info_wl}  CY:{file_info_cy}  TS:{file_info_ts }")

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

