"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pandas as pd

path = '../../training_data/'
scenario = '.............'

own_df = pd.read_csv(path+scenario+'.rdr', sep='\t')
df = pd.read_csv(path+scenario, sep='\t')

timestamp_start = own_df['Unixtime'].iloc[0]
timestamp_end = own_df['Unixtime'].iloc[-1]

df = df[(df['Unixtime'] >= timestamp_start) & (df['Unixtime'] <= timestamp_end)]

df = pd.concat([own_df, df], axis=0)
df = df.astype({'RTKey': 'int64', 'Unixtime': 'int64'})
df.to_csv('...................', index=False)
