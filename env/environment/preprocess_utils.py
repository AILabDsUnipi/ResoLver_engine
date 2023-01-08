"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import glob
from pathlib import Path
import pandas as pd
import numpy as np
from numba import njit
import os
from math import sqrt, degrees
import utils
import json
import pickle
import env_config

final_td = 30

#Also compute speeds!!
def interpolate_flights(path, outpath):
    Path(outpath+'/interpolated').mkdir(parents=True, exist_ok=True)

    for file in glob.glob(path + '/*.rdr'):
        dfs = []
        neighbour_flights_file = file.split('/')[-1].split('.')[0]

        own_df = pd.read_csv(file, sep='\t')
        start_time = own_df['Unixtime'].iloc[0]
        end_time = own_df['Unixtime'].iloc[-1]
        dfs.append([own_df, outpath + '/interpolated/' + neighbour_flights_file+'_interpolated.rdr', '.rdr'])
        neighbour_flights = pd.read_csv(path + '/' + neighbour_flights_file, sep='\t')

        dfs.append([neighbour_flights, outpath + '/interpolated/' + neighbour_flights_file+'_interpolated', ''])

        for df_init, filename, type in dfs:
            df = df_init[(df_init['Unixtime'] >= start_time) & (df_init['Unixtime'] <= end_time)]
            interpolated_flights_np_lst = []
            last_points = []
            for name, group in df.groupby(['RTKey']):
                group_dups = group[group.duplicated(keep=False)]

                group_np = group.drop_duplicates().sort_values('Unixtime').values

                if group_np.shape[0] < 2:
                    continue

                points_lst, sectors_lst = interpolate_flight(group_np[:, :-1].astype(np.float64),
                                                             group_np[:, -1].astype(str))
                last_points.append(group_np[-1])

                if len(points_lst) == 0:
                    continue

                interpolated_flight_np = np.concatenate([points_lst, np.array(sectors_lst)[:, np.newaxis]], axis=1)

                # sectors_lst
                interpolated_flights_np_lst.append(interpolated_flight_np)

            interpolated_df = pd.DataFrame(np.concatenate(interpolated_flights_np_lst, axis=0),
                                           columns=df.columns)
            interpolated_df.to_csv(filename, index=False, sep='\t')
            final_points_df = pd.DataFrame(last_points, columns=own_df.columns)
            final_points_df.to_csv(outpath + '/interpolated/' + neighbour_flights_file+'_final_points'+type, index=False)

@njit
def interpolate_flight(flight_np, sectors_np):
    """
    :param flight_np: Nx[0: RTKey, 1: Lat, 2: Lon, 3: Alt, 4: Unixtime, 5: Sector]
    :return:
    """

    interpolated_point = np.zeros((flight_np.shape[1]), dtype=np.float64)
    interpolated_5_list_sectors = []
    interpolated_5_list = []
    constant_td = 1

    for idx, t_point in enumerate(flight_np):
        if idx < flight_np.shape[0] - 1:
            point = (t_point[2], t_point[1])
            alt = t_point[3]
            alt_next = flight_np[idx + 1][3]
            td = flight_np[idx + 1][4] - flight_np[idx][4]
            if td == 0:
                print(flight_np[idx + 1][4])
                print(flight_np[idx][4])
                print(list(flight_np[idx]))
                print(list(flight_np[idx+1]))

            v_speed = (alt_next - alt) / td
            alt_new = alt
            p_next = point
            interpolation_lon_speed = (flight_np[idx + 1][2] - flight_np[idx][2]) / td
            interpolation_lat_speed = (flight_np[idx + 1][1] - flight_np[idx][1]) / td

            for t in range(int(td)):
                interpolated_point[0] = flight_np[idx][0]
                interpolated_point[2] = p_next[0]
                interpolated_point[1] = p_next[1]
                interpolated_point[3] = alt_new
                interpolated_point[4] = flight_np[idx][4] + t

                p_next = (
                p_next[0] + interpolation_lon_speed * constant_td, p_next[1] + interpolation_lat_speed * constant_td)
                alt_new = alt_new + v_speed * constant_td

                if interpolated_point[4] % final_td == 0:
                    interpolated_5_list_sectors.append(str(sectors_np[idx]))
                    interpolated_5_list.append(np.copy(interpolated_point))
        elif idx == flight_np.shape[0] - 1:

            interpolated_point[0] = flight_np[idx][0]
            interpolated_point[2] = flight_np[idx][2]
            interpolated_point[1] = flight_np[idx][1]
            interpolated_point[3] = flight_np[idx][3]
            interpolated_point[4] = flight_np[idx][4]

            if interpolated_point[4] % final_td == 0:
                interpolated_5_list.append(np.copy(interpolated_point))
                interpolated_5_list_sectors.append(str(sectors_np[idx]))

    return interpolated_5_list, interpolated_5_list_sectors

def compute_speed(point0, point1):
    td = point1['timestamp'] - point0['timestamp']
    if td != final_td:
        print(td)
        print(point0)
        print(point1)
        exit(0)
    lon_speed = (point1['longitude']-point0['longitude'])/td
    lat_speed = (point1['latitude'] - point0['latitude'])/td

    speed_magnitude = sqrt(lon_speed**2 + lat_speed**2)

    return speed_magnitude

def mybearing(v0, v1):

    angle = degrees(np.arctan2(v1[0] - v0[0], v1[1] - v0[1]))
    whole_angle = (360 + angle)%360

    return whole_angle, angle


def equibins(df):

    bins_median_dict = {}
    for col in ['d_angles_transformed', 'd_h_speeds_transformed']:
        out, bins = pd.qcut(df[col], q=20, retbins=True)
        df['bin_'+col] = out

        bins_median_dict[col] = df.groupby(['bin_'+col]).median()[col].values

    return bins_median_dict


def compute_dangles_dspeeds():

    max_abs_dbearings = []
    min_abs_dbearings = []
    max_dbearings = []
    min_dbearings = []

    max_abs_dangles = []
    min_abs_dangles = []
    max_dangles = []
    min_dangles = []

    max_abs_transformed_dangles = []
    min_abs_transformed_dangles = []
    max_transformed_dangles = []
    min_transformed_dangles = []

    transformer = utils.utils_global['transformer']

    dangles_transformed_total = []
    d_h_speeds_transformed_total = []
    d_v_speeds_total = []

    path = '/media/hdd/Documents/TP4CFT_Datasets/2023_preprocessed/FINAL/training/interpolated'
    for file in glob.glob(path+'/*'):

        df = pd.read_csv(file, sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                         'Lon': 'longitude', 'Alt': 'altitude',
                                                         'Unixtime': 'timestamp'})

        df = df[(df['Sector'] != 'N/A') & (~df['Sector'].isna())]

        for name, group in df.groupby('flightKey', sort=False):

            group_sorted = group.sort_values(by='timestamp').drop_duplicates().reset_index(drop=True)

            transformed_angles = []
            v_speeds = []
            speeds_transformed = []
            if group_sorted.shape[0] == 1:
                continue
            for idx, t_point in group_sorted.iterrows():

                if idx < group_sorted.shape[0] - 1:

                    transformed_point = transformer.transform(t_point['longitude'], t_point['latitude'])
                    transformed_point_next = transformer.transform(group_sorted.iloc[idx + 1]['longitude'],
                                                                   group_sorted.iloc[idx + 1]['latitude'])

                    _, ta = mybearing(transformed_point, transformed_point_next)
                    transformed_angles.append(ta)

                    s_transformed = compute_speed({'longitude': transformed_point[0],
                                                  'latitude': transformed_point[1],
                                                  'timestamp': t_point['timestamp']},
                                                  {'longitude': transformed_point_next[0],
                                                   'latitude': transformed_point_next[1],
                                                   'timestamp': group_sorted.iloc[idx + 1]['timestamp']})

                    speeds_transformed.append(s_transformed)

                    v_s = (group_sorted.iloc[idx + 1]['altitude'] -
                           t_point['altitude'])/(group_sorted.iloc[idx + 1]['timestamp'] - t_point['timestamp'])
                    v_speeds.append(v_s)
                else:

                    transformed_point = transformer.transform(t_point['longitude'], t_point['latitude'])
                    transformed_point_prev = transformer.transform(group_sorted.iloc[idx - 1]['longitude'],
                                                                   group_sorted.iloc[idx - 1]['latitude'])

                    _, ta = mybearing(transformed_point_prev, transformed_point)
                    transformed_angles.append(ta)

                    s_transformed = compute_speed({'longitude': transformed_point_prev[0],
                                                   'latitude': transformed_point_prev[1],
                                                   'timestamp': group_sorted.iloc[idx - 1]['timestamp']},
                                                  {'longitude': transformed_point[0],
                                                   'latitude': transformed_point[1],
                                                   'timestamp': t_point['timestamp']})
                    speeds_transformed.append(s_transformed)

                    v_s = (t_point['altitude'] - group_sorted.iloc[idx - 1]['altitude'])/\
                          (t_point['timestamp'] - group_sorted.iloc[idx - 1]['timestamp'])
                    v_speeds.append(v_s)

            transformed_angles_np = np.array(transformed_angles)
            v_speeds_np = np.array(v_speeds)
            s_transormed_np = np.array(speeds_transformed)

            dangles_transformed_np_tmp = (transformed_angles_np[1:] - transformed_angles_np[:-1])
            dangles_complementary_np = np.where(dangles_transformed_np_tmp > 0,
                                                (360 - dangles_transformed_np_tmp) % 360,
                                                (360 + dangles_transformed_np_tmp) % 360)
            sign = np.sign(dangles_transformed_np_tmp)
            dangles_transformed_np = np.where(np.abs(dangles_transformed_np_tmp) > 180,
                                              -sign * dangles_complementary_np, dangles_transformed_np_tmp)
            gt90 = np.where(np.abs(dangles_transformed_np) > 90)[0]

            if gt90.shape[0] > 0:
                print('gt90')
                print(name)
                print(gt90)
                print(dangles_transformed_np[gt90])
                print(file)
            dangles_transformed_np = dangles_transformed_np[:, np.newaxis]

            d_h_speeds_transformed_np = (s_transormed_np[1:] - s_transormed_np[:-1])[:, np.newaxis]
            d_v_speeds_np = (v_speeds_np[1:] - v_speeds_np[:-1])[:, np.newaxis]

            dangles_transformed_total.append(dangles_transformed_np)
            d_h_speeds_transformed_total.append(d_h_speeds_transformed_np)
            d_v_speeds_total.append(d_v_speeds_np)

    dangles_transformed_np = np.concatenate(dangles_transformed_total, axis=0)
    d_h_speeds_transformed_np = np.concatenate(d_h_speeds_transformed_total, axis=0)
    d_v_speeds_np = np.concatenate(d_v_speeds_total, axis=0)
    ddf = pd.DataFrame(np.concatenate([dangles_transformed_np, d_h_speeds_transformed_np, d_v_speeds_np],
                                      axis=1),
                       columns=['d_angles_transformed', 'd_h_speeds_transformed', 'd_v_speeds'])

    bins_median_dict = equibins(ddf)
    Path(env_config.env_config['dependency_files_directory']).mkdir(parents=True, exist_ok=True)
    with open(env_config.env_config['dependency_files_directory']+'/bins_median_dict.pickle', 'wb') as handle:
        pickle.dump(bins_median_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    thedir = '/media/hdd/Documents/TP4CFT_Datasets/2023_preprocessed/FINAL/training'

    for fpath in glob.glob('/media/hdd/Documents/TP4CFT_Datasets/2023_preprocessed/FINAL/training/*', recursive=False):
        if os.path.isdir(fpath) and fpath.split('/')[-1] != 'interpolated':
            interpolate_flights(fpath, thedir)
