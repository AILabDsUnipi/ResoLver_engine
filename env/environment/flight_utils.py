"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd
import copy
from numba import njit
from math import sqrt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from utils import mybearing, utils_global

@njit
def compute_speed(point0, point1):
    """Computes speed between two 4D spatiotemporal points.
    :param point0: First point (x,y,z,t).
    :param point1: Second point (x,y,z,t)
    :return: speed components for each axis, speed magnitude
    """
    td = point1[3] - point0[3]
    x_speed_component = (point1[0]-point0[0])/td
    y_speed_component = (point1[1] - point0[1])/td
    alt_speed = (point1[2]-point0[2])/td

    speed_magnitude = sqrt(x_speed_component**2 + y_speed_component**2)

    return x_speed_component, y_speed_component, alt_speed, speed_magnitude

def transform_compute_speed_bearing(df):
    df = transform_coordinates(df)

    df = compute_speed_bearing(df)

    return df

def transform_coordinates(df):

    df[['x', 'y']] = df[['longitude', 'latitude']].apply(lambda x: utils_global['transformer']
                                                         .transform(x['longitude'], x['latitude']),
                                                         axis=1, result_type='expand')
    return df

def transform_wrapper(elem):
    return utils_global['transformer'].transform(elem[0], elem[1])

def transform_wrapper_inverse(elem):
    return utils_global['inverse_transformer'].transform(elem[0], elem[1])

def transform_coordinates_np(arr_np, inverse=False):
    """
    :param arr_np: M x [lon,lat] array
    :return: M x [x,y] array
    """
    if not inverse:
        transformed_np = np.apply_along_axis(transform_wrapper, 1, arr_np)
    else:
        transformed_np = np.apply_along_axis(transform_wrapper_inverse, 1, arr_np)
    return transformed_np

def compute_speed_bearing(df):

    groups = []

    for name, group_df in df.groupby('flightKey', sort=False):
        if group_df.shape[0] <= 1:
            continue
        group = group_df.copy()
        states_np = group[['x', 'y']].values[:-1]
        next_states_np = group[['x', 'y']].values[1:]
        course_np = [mybearing(s, s_prime)[0] for s, s_prime in zip(states_np, next_states_np)]

        course_np = np.append(course_np[0], course_np)
        group['course'] = course_np

        dxy_np = next_states_np - states_np
        dxy_np = np.append([dxy_np[0]], dxy_np, axis=0)

        dt_np = group['timestamp'].values[1:] - group['timestamp'].values[:-1]
        dalt_np = group['altitude'].values[1:] - group['altitude'].values[:-1]
        alt_speed_np = dalt_np/dt_np
        alt_speed_state_np = np.append(alt_speed_np[0], alt_speed_np)
        group['alt_speed'] = alt_speed_state_np

        dt_np = np.append([dt_np[0]], dt_np)

        group['speed_x_component'] = dxy_np[:, 0] / dt_np
        group['speed_y_component'] = dxy_np[:, 1] / dt_np
        group['speed'] = np.sqrt(np.sum(group[['speed_x_component', 'speed_y_component']].values ** 2, axis=1).astype(np.float32))

        groups.append(group)

    if len(groups) > 0:
        df = pd.concat(groups, axis=0)
        return df

    return None
