"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import glob
import pandas as pd
import shutil
import os

from pathlib import Path

if __name__ == "__main__":
    path = '/media/hdd/Documents/TP4CFT_Datasets/2023_preprocessed/FINAL/training/'
    fplan_path = '/media/hdd/Documents/TP4CFT_Datasets/2023_preprocessed/FINAL/output/flights'

    out_path = '../../fplans'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    for dir in glob.glob(path+'/trainingData_*'):
        month = dir.split('/')[-1].split('_')[-1][:6]
        for f in glob.glob(dir+'/*'):
            df = pd.read_csv(f, sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                          'Lon': 'longitude', 'Alt': 'altitude',
                                                          'Unixtime': 'timestamp'})
            uRTk = df['flightKey'].unique()
            for RTk in uRTk:
                RTk_fplan_paths = glob.glob(fplan_path+'/*/'+str(RTk)+'/*')
                dest_path = out_path+'/'+month+'/'+str(RTk)
                Path(dest_path).mkdir(
                    parents=True,
                    exist_ok=True)

                for fplan in RTk_fplan_paths:
                    if fplan[-3:] == 'rad':
                        continue
                    source_month = fplan.split('/')[-3]
                    name = source_month+'_'+fplan.split('/')[-1]
                    shutil.copy2(fplan, dest_path+'/'+name)
