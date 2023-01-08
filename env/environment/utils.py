"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

from numba import njit, prange, jit
from pyproj import Transformer
from math import cos, sin, radians, degrees, sqrt, floor, ceil
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
import geometry_utils
from env_config import env_config
import flight_plan_utils

utils_global = {'transformer': Transformer.from_crs("EPSG:4326", "EPSG:2062", always_xy=True),
                'inverse_transformer': Transformer.from_crs("EPSG:2062", "EPSG:4326", always_xy=True),
                'debug_folder': 'A2_1547541184_LECBP2R_IBE31KF_debug_files'}


@njit
def mybearing(v0, v1):
    """
    Given two 2D points (x,y) computes the clockwise angle (whole angle)
    and the 0-180 degrees angle (angle)
    with the minus sign denoting counter-clockwise and the plus sign denoting clockwise
    between the line formed by these two points and the y axis.
    :param v0: starting point (x,y)
    :param v1: end point (x,y)
    :return: clockwise angle (whole angle), angle (angle) with the minus sign denoting counter-clockwise
             and the plus sign denoting clockwise
    """
    angle = degrees(np.arctan2(float(v1[0] - v0[0]), float(v1[1] - v0[1])))
    whole_angle = (360 + angle) % 360

    return np.float64(whole_angle), np.float64(angle)


@njit
def mybearing_speed_components(x_speed, y_speed):
    """
    Given x and y speed components computes the clockwise angle (whole angle)
    and the 0-180 degrees angle (angle)
    with the minus sign denoting counter-clockwise and the plus sign denoting clockwise
    between the line formed by these two points and the y axis.
    :param x_speed: speed horizontal component
    :param y_speed: speed vertical component
    :return: clockwise angle (whole angle), angle (angle) with the minus sign denoting counter-clockwise
             and the plus sign denoting clockwise
    """
    angle = degrees(np.arctan2(x_speed, y_speed))
    whole_angle = (360 + angle) % 360

    return np.float64(whole_angle), np.float64(angle)


def equibins():

    origin = 'LEMG'
    destination = 'EGKK'
    mypath = '/media/hdd/Documents/TP4CFT_Datasets/dbearing_dspeeds_'+origin+'_'+destination+'/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    dfs = []
    for file in files:
        dfs.append(pd.read_csv(mypath+file))

    df = pd.concat(dfs, axis=0)
    bins_median_dict = {}
    for col in ['d_angles_transformed', 'd_h_speeds_transformed']:
        out, bins = pd.qcut(df[col], q=20, retbins=True)
        df['bin_'+col] = out

        bins_median_dict[col] = df.groupby(['bin_'+col]).median()[col].values

    return bins_median_dict

@njit
def flights_cross_in_t_w(intersection_p, ownship_p, own_velocity, t_w):

    intersection_p_own_p_vector = intersection_p-ownship_p
    t = np.sqrt(np.sum(own_velocity**2))/np.sqrt(np.sum(intersection_p_own_p_vector**2))

    if t <= t_w:
        return True

    return False

@njit
def compute_alternative_speed_components(flight_speed_components, d_angles, d_speeds):
    """

    :param flight_speed_components: array of [x_speed,y_speed] components
    :param d_angles:
    :param d_speeds:
    :return:
    """
    flights_h_speed_components = np.zeros((flight_speed_components.shape[0],len(d_angles)*len(d_speeds), 2))

    for i, x_y_speed_components in enumerate(flight_speed_components):
        x_speed = x_y_speed_components[0]
        y_speed = x_y_speed_components[1]
        speed_magnitude = sqrt(x_speed**2 + y_speed**2)

        b, _ = mybearing_speed_components(x_speed, y_speed)
        h_speed_components = np.zeros((len(d_angles)*len(d_speeds), 2))
        for a in range(len(d_angles)):
            new_b = b + d_angles[a]

            for s in range(len(d_speeds)):
                h_speed = speed_magnitude + d_speeds[s]
                x_speed_component = (sin(radians(new_b)))*h_speed
                y_speed_component = (cos(radians(new_b)))*h_speed
                h_speed_components[a*len(d_speeds)+s] = [x_speed_component, y_speed_component]

        flights_h_speed_components[i] = h_speed_components
    return flights_h_speed_components

@njit
def compute_cpa(A0, A1, u, v):
    """
    A0: position of ownship
    A1: position of neighbour
    u: speed vector of ownship
    v: speed vector of neighbour
    tcpa = -W0(v-u)/norm(v-u)**2
    where W0 = vec(A0A1)
    """

    W0 = A1-A0
    flag = True
    if np.all(v-u) == 0 or np.all(A0-A1) == 0:
        tcpa = 0
        flag = False
    else:
        tcpa = -np.dot(W0, (v-u))/np.sum((v-u)**2)
    dcpa = W0+(v-u)*tcpa

    ownship_location = A0+u*tcpa
    neighbour_location = A1+v*tcpa

    d = np.sqrt(np.sum((dcpa)**2))

    return tcpa, dcpa, d, ownship_location, neighbour_location, flag

@njit
def compute_tcpa(A0, A1, u, v):
    """
    A0: position of ownship
    A1: position of neighbour
    u: speed vector of ownship
    v: speed vector of neighbour
    tcpa = -W0(v-u)/norm(v-u)**2
    where W0 = vec(A0A1)
    """
    if np.all(v - u) == 0 or np.all(A0 - A1) == 0:
        tcpa = 0
    else:
        W0 = A1-A0
        tcpa = -np.dot(W0, (v-u))/np.sum((v-u)**2)

    return tcpa

@njit
def compute_dcpa(A0, A1, u, v, tcpa):
    W0 = A1 - A0
    dcpa = W0+(v-u)*tcpa

    ownship_location = A0+u*tcpa
    neighbour_location = A1+v*tcpa

    d = np.sqrt(np.sum((dcpa)**2))

    return dcpa, d, ownship_location, neighbour_location

@njit
def myfactorial(x):
    result = 1
    if x <= 1:
        return 1

    for i in range(x, 1, -1):
        result *= i

    return result

def to_WKT_file(flight_np, fname, mode):
    t_w = env_config['tw']
    point0 = flight_np[1:3]
    point1 = point0+flight_np[3:5]*t_w
    ls1 = LineString([point0, point1])
    ls1_transformed = transform(utils_global['inverse_transformer'].transform, ls1)
    p0 = Point(point0)
    p0_transformed = transform(utils_global['inverse_transformer'].transform, p0)
    header = 'flight1;flight2;geometry;altitude;alt_speed\n'
    with open(fname+'_line.csv', mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(str(flight_np[0])+';'+str(flight_np[0])+';'+ls1_transformed.wkt +
                ';'+str(flight_np[5])+';'+str(flight_np[6])+'\n')
    with open(fname + '_point.csv', mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(str(flight_np[0]) + ';' + str(flight_np[0]) + ';' + p0_transformed.wkt + ';' + str(flight_np[5]) +
                ';'+str(flight_np[6]) + '\n')

def to_WKT_file_cpa(flightid1,flightid2, cpa, tcpa, dcpa ,
                    ownship_alt_cpa, neighbour_alt_cpa, v_d_cpa,
                    bearing_cpa_ownship_neighbour_relative, fname, mode):
    header = 'flight1;flight2;tcpa;dcpa;geometry;flight1_alt_cpa;' \
             'flight2_alt_cpa;v_d_cpa;bearing_cpa_ownship_neighbour_relative\n'
    p0 = Point(cpa)
    p0_transformed = transform(utils_global['inverse_transformer'].transform, p0)
    with open(fname + '_point.csv', mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(str(flightid1) + ';' + str(flightid2) + ';' + str(tcpa) +
                ';' + str(dcpa) + ';' + p0_transformed.wkt + ';' + str(ownship_alt_cpa) +
                ';' + str(neighbour_alt_cpa)+';' + str(v_d_cpa)+';' + str(bearing_cpa_ownship_neighbour_relative)+'\n')

def to_WKT_file_intersection_p(flightid1, flightid2, intersection_p, passed, intersection_angle,
                               crossing_distance, time_to_crossing_point, fname, mode):
    header = 'flight1;flight2;passed;geometry;intersection angle;crossing_distance;time_to_crossing_point\n'
    p0 = Point(intersection_p)
    p0_transformed = transform(utils_global['inverse_transformer'].transform, p0)

    with open(fname + '_point.csv', mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(str(flightid1) + ';' + str(flightid2) + ';' + str(passed) + ';'
                + p0_transformed.wkt + ';' + str(intersection_angle)+';' +
                str(crossing_distance) + ';' + str(time_to_crossing_point) + '\n')

def to_WKT_polygon_area(flightid, polygon, intersect, fname, mode):
    header = 'flight1;flight2;intersect;geometry\n'
    poly = Polygon(polygon)
    poly_transformed = transform(utils_global['inverse_transformer'].transform, poly)

    with open(fname + '_polygons.csv', mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(str(flightid) + ';' + str(flightid) + ';'+str(intersect) + ';' + poly_transformed.wkt + '\n')

@njit
def flight_distance(f1_position, f2_position):
    """
    Computes the horizontal and vertical distance at the current time point between two flights
    :param f1_position: [x,y,alt] of flight 1
    :param f2_position: [x,y,alt] of flight 2
    :return: horizontal distance, vertical distance
    """

    horizontal_distance = np.sqrt(np.sum((f1_position[:2]-f2_position[:2])**2))
    vertical_distance = np.abs(f1_position[2]-f2_position[2])

    return horizontal_distance, vertical_distance

def flight_distance_unit_testing():
    h_distance, v_distance = flight_distance(np.array([1., 3., 10.]), np.array([5., 0., 15.]))

    print(h_distance, v_distance)

    assert h_distance == 5 and v_distance == 5

def time_to_nextfl(flight_alt, alt_speed):
    final_alt = flight_alt
    if alt_speed < 0:
        final_alt = ceil(flight_alt/1000-1)*1000
        time_to_final_alt = (final_alt - flight_alt) / alt_speed

    elif alt_speed > 0:
        final_alt = floor(flight_alt/1000+1)*1000
        time_to_final_alt = (final_alt - flight_alt) / alt_speed

    return time_to_final_alt, final_alt

def detect_conflicts(p0, p1, u, v, h_tcpa, d_h_cpa, v_tcpa, d_v_cpa):

    ownship_alt = p0[2]
    neighbour_alt = p1[2]
    ownship_final_alt = ownship_alt
    neighbour_final_alt = neighbour_alt
    time_start = max(v_tcpa, 0)
    conflict_time = -1
    if u[2] < 0:
        ownship_final_alt = floor(p0[2]/1000)*1000
        ownship_time_to_final_alt = (ownship_final_alt - ownship_alt) / u[2]
    elif u[2] > 0:
        ownship_final_alt = ceil(p0[2]/1000)*1000
        ownship_time_to_final_alt = (ownship_final_alt - ownship_alt) / u[2]
    else:
        ownship_time_to_final_alt = max(h_tcpa, 0)

    if v[2] < 0:
        neighbour_final_alt = floor(p1[2]/1000)*1000
        neighbour_time_to_final_alt = (neighbour_final_alt - neighbour_alt) / v[2]
    elif v[2] > 0:
        neighbour_final_alt = ceil(p1[2]/1000)*1000
        neighbour_time_to_final_alt = (neighbour_final_alt - neighbour_alt) / v[2]
    else:
        neighbour_time_to_final_alt = max(h_tcpa, 0)

    time_end = min(neighbour_time_to_final_alt, ownship_time_to_final_alt)
    conflict = False
    time_list = []

    if v[2] == 0 and u[2] == 0:
        v_sep_minimum = 1000
        if ownship_alt >= 41000 or neighbour_alt >= 41000:
            v_sep_minimum = 2000

        if h_tcpa >= 0 and d_h_cpa < env_config['horizontal_sep_minimum'] and np.abs(ownship_alt-neighbour_alt) < v_sep_minimum:
            conflict = True

        return conflict
    elif time_start < time_end:
        time_list = [time_start] + [t for t in range(int(5 * round(time_start / 5)), int(5 * round(time_end / 5)), 5)
                                    if t > time_start] +\
                    [t for t in [int(5 * round(time_end / 5))] if int(5 * round(time_end / 5)) < time_end] + \
                    [time_end]

    elif time_start > time_end:

        time_list = [time_start] + [t for t in range(int(5 * round(time_start / 5)), int(5 * round(time_end / 5)), -5)
                                    if t < time_start] +\
                    [t for t in [int(5 * round(time_end / 5))] if int(5 * round(time_end / 5)) > time_end] + \
                    [time_end]
    elif time_start == time_end:
        time_list = [time_start]

    for time in time_list:

        v_sep_minimum = 1000
        vdcpa, vd, ownship_cpa_alt, neighbour_cpa_alt = \
            compute_dcpa(p0[2][np.newaxis], p1[2][np.newaxis], u[2][np.newaxis], v[2][np.newaxis], time)

        if ownship_cpa_alt >= 41000 or neighbour_cpa_alt >= 41000:
            v_sep_minimum = 2000

        if vd >= v_sep_minimum:
            break
        hdcpa, hd, ownship_cpa_location, neighbour_cpa_location = compute_dcpa(p0[:2], p1[:2], u[:2], v[:2], time)

        if hd < env_config['horizontal_sep_minimum']*1852:
            conflict = True
            conflict_time = time
            if time_start <= time_end:
                break
    if conflict:
        print('utils.detect_conflicts conflicts')

    return conflict

def next_wps_per_flight(exit_wps, flight_plans, flight_index, flight_arr, flight_idx, point, course):
    ## ATTENTION add degrees constraint
    flightKey = flight_arr[flight_idx][0]

    if len(flight_plans[flightKey]['fplans']) == 0:
        next_wps = exit_wps[flight_index[flightKey]['idx'], np.array([5, 6])][np.newaxis]
        p_start = flight_index[flightKey]['df'][['x', 'y']].values[0]
        p_end = flight_index[flightKey]['last_point'][['x', 'y']].values[0]
        p_end_rank = flight_plan_utils.compute_rank(p_end - p_start, p_end - p_start)
        point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

        bearing_to_next, _ = mybearing(point, next_wps[0].astype(np.float64))
        dcourse = bearing_to_next - course
        abs_dcourse = abs(dcourse)
        if abs_dcourse > 180:
            abs_dcourse = 360 - abs(dcourse)
        if abs_dcourse < 90:
            d_to_idx = -1
        else:
            d_to_idx = -2
        if point_rank < p_end_rank:
            return next_wps, -1, np.nan, d_to_idx
        else:
            return None, -1, np.nan, d_to_idx

    fplan = flight_plans[flightKey]['fplans'][flight_plans[flightKey]['current']][1]
    p_start = flight_plans[flightKey]['fplans'][flight_plans[flightKey]['current']][3]
    p_end = flight_plans[flightKey]['fplans'][flight_plans[flightKey]['current']][4]
    point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

    temp_idx = np.searchsorted(fplan['rank'].values, point_rank)
    idx = fplan.shape[0]

    for i in range(temp_idx, fplan.shape[0]):
        wp = fplan[['x', 'y']].values[i]
        bearing_to_next, _ = mybearing(point, wp.astype(np.float64))
        dcourse = bearing_to_next - course
        abs_dcourse = abs(dcourse)
        if abs_dcourse > 180:
            abs_dcourse = 360 - abs(dcourse)
        if abs_dcourse >= 90 and i < fplan[['x', 'y']].shape[0]:
            continue

        idx = i

        break

    if idx >= fplan.shape[0]:

        next_wps = exit_wps[flight_index[flightKey]['idx'], np.array([5, 6])][np.newaxis]

        bearing_to_next, _ = mybearing(point, next_wps[0].astype(np.float64))

        dcourse = bearing_to_next - course
        abs_dcourse = abs(dcourse)
        if abs_dcourse > 180:
            abs_dcourse = 360 - abs(dcourse)

        if abs_dcourse < 90:
            d_to_idx = -1
        else:
            d_to_idx = -2

        return None, idx, flight_plans[flightKey]['fplans'][flight_plans[flightKey]['current']][2], d_to_idx
    else:
        next_wps = fplan[['x', 'y']].iloc[idx:].values
        bearing_to_next, _ = mybearing(point, next_wps[0].astype(np.float64))
        dcourse = bearing_to_next - course
        d_to_idx = idx

        return next_wps, idx, flight_plans[flightKey]['fplans'][flight_plans[flightKey]['current']][2], d_to_idx

def direct_to_projection(flight_np, flight_plans, flight_index, flight_arr, flight_idx,
                         timestamp, res_act_id, exit_wps, tw, towards_wp_idx, executing_direct_to_flag,
                         res_type, towards_wp_fp_idx, actions_impact_flag):

    alt_speed = flight_np[6]
    course, _ = mybearing_speed_components(flight_np[3], flight_np[4])
    position = flight_np[1:3]
    alt = flight_np[5]
    speed = np.sqrt(np.sum(flight_np[3:5]**2))
    x_speed_component = flight_np[3]
    y_speed_component = flight_np[4]

    if res_type == 'A3' and actions_impact_flag:
        wps_r, wp_idx_r, fplan_used1, next_wp_idx = next_wps_per_flight(exit_wps, flight_plans, flight_index,
                                                                        flight_arr, flight_idx, position, course)
        towards_wp_fp_idx = flight_plans[flight_np[0]]['current']
        if next_wp_idx >= 0:
            towards_wp_idx = next_wp_idx+int(res_act_id.split('_')[3])-1
        else:
            towards_wp_idx = next_wp_idx

    fplan_used_flag1 = False
    fplan_used1 = None
    intersection_point_1 = None
    int_p_alt_aircraft_1 = None
    flight_plan_course_1 = None
    flight_plan_alt_1 = None
    flight_plan_h_speed_1 = None
    flight_plan_v_speed_1 = None

    if towards_wp_idx == -2:
        wps_r = np.array([position, position+np.array([x_speed_component, y_speed_component])*tw])
    else:
        if towards_wp_idx == -1:
            wps_r = exit_wps[flight_index[flight_np[0]]['idx'], np.array([5, 6])][np.newaxis]
        else:

            fplan = flight_plans[flight_np[0]]['fplans'][towards_wp_fp_idx][1]
            wps_r = fplan[['x', 'y']].values
            wps_r = wps_r[towards_wp_idx:]

        wps_r = np.concatenate([position[np.newaxis], wps_r], axis=0).astype(np.float64)
        fplan_used_flag1 = True
        intersection_point_1 = wps_r[0]
        r_distance = np.sqrt(np.sum((wps_r[0] - position) ** 2, axis=-1))
        r_t = r_distance / speed
        int_p_alt_aircraft_1 = alt + alt_speed * r_t
        if len(flight_plans[flight_np[0]]['fplans']) > 0:
            fplan = flight_plans[flight_np[0]]['fplans'][towards_wp_fp_idx][1]

            flight_plan_alt_1 = fplan['altitude'].values[towards_wp_idx]

            if towards_wp_idx < fplan.shape[0] - 1:
                wp_r = fplan[['x', 'y', 'altitude', 'timestamp']].values[towards_wp_idx]
                wp_r_next = fplan[['x', 'y', 'altitude', 'timestamp']].values[towards_wp_idx + 1]
                flight_plan_course_1, _ = mybearing(wp_r[:2], wp_r_next[:2])
                dist = np.sqrt(np.sum((wp_r_next[:2] - wp_r[:2]) ** 2, axis=-1))
                time_diff = (wp_r_next[3] - wp_r[3])
                if time_diff == 0:
                    time_diff = 30
                flight_plan_h_speed_1 = dist / time_diff
                flight_plan_v_speed_1 = (wp_r_next[2] - wp_r[2]) / time_diff
            elif towards_wp_idx > 0:
                wp_r = fplan[['x', 'y', 'altitude', 'timestamp']].values[towards_wp_idx]
                wp_prev = fplan[['x', 'y', 'altitude', 'timestamp']].values[towards_wp_idx - 1]
                flight_plan_course_1, _ = mybearing(wp_prev[:2], wp_r[:2])
                dist = np.sqrt(np.sum((wp_r[:2] - wp_prev[:2]) ** 2, axis=-1))
                time_diff = (wp_r[3] - wp_prev[3])
                if time_diff == 0:
                    time_diff = 30
                flight_plan_h_speed_1 = dist / time_diff
                flight_plan_v_speed_1 = (wp_r[2] - wp_prev[2]) / time_diff

    projection_info = [fplan_used_flag1, fplan_used1,
                       intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,
                       flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1]

    line_segments_np = geometry_utils.poly_line_segments(wps_r, dim2=2, convex=False)

    tw = env_config['tw']
    if alt_speed != 0:
        time_to_final_alt1, final_alt1 = time_to_nextfl(alt, alt_speed)
        tw = time_to_final_alt1

    enriched_segments_w_alt_np = flight_plan_utils.enrich_segments_w_time(line_segments_np, timestamp, speed, alt, alt_speed, tw)
    res_id = 'no_resolution'
    if res_act_id is not None:
        res_id = res_act_id.split('_')[2:]
        res_id = '_'.join(res_id)
    projection_ID = str(int(timestamp)) + '_' + str(int(flight_np[0])) + '_' + res_id
    if enriched_segments_w_alt_np.shape[0] > 0:
        projection_points_1 = geometry_utils.line_segments_to_points(enriched_segments_w_alt_np, False)

        (projection_points_1[:, 0], projection_points_1[:, 1]) = flight_plan_utils.update_lon_lat_from_x_y_vectorized(
            projection_points_1[:, 0],
            projection_points_1[:, 1])
        projection_points_1 = np.concatenate([[[projection_ID, int(flight_np[0]), timestamp, res_id, r_num]
                                               for r_num in range(projection_points_1.shape[0])],
                                              projection_points_1,],
                                             axis=1)
    else:
        projection_points_1 = np.empty((0, 0, 5))

    return projection_points_1, projection_ID, enriched_segments_w_alt_np, projection_info

def course_change_projection(flight_np, flight_plans, flight_index, flight_arr, flight_idx,
                             timestamp, res_act_id, exit_wps, duration, action_starting_time_point):

    alt_speed = flight_np[6]
    course, _ = mybearing_speed_components(flight_np[3], flight_np[4])
    position = flight_np[1:3]
    alt = flight_np[5]
    speed = np.sqrt(np.sum(flight_np[3:5]**2))
    x_speed_component = flight_np[3]
    y_speed_component = flight_np[4]
    duration_i = duration[flight_index[flight_np[0]]['idx']]
    action_starting_time_point_i = action_starting_time_point[flight_index[flight_np[0]]['idx']]
    r_position = position + \
                 np.array([x_speed_component, y_speed_component]) * \
                 (duration_i-(timestamp-action_starting_time_point_i))
    wps_r, wp_idx_r, fplan_used1, angle_constraint = \
        next_wps_per_flight(exit_wps, flight_plans, flight_index, flight_arr, flight_idx, r_position, course)

    fplan_used_flag1 = False
    fplan_used1 = None
    intersection_point_1 = None
    int_p_alt_aircraft_1 = None
    flight_plan_course_1 = None
    flight_plan_alt_1 = None
    flight_plan_h_speed_1 = None
    flight_plan_v_speed_1 = None

    if wps_r is None:
        wps_r = np.array([position, r_position])

    else:
        if len(flight_plans[flight_np[0]]['fplans']) > 0:
            fplan_used_flag1 = True
            intersection_point_1 = wps_r[0]
            r_distance = np.sqrt(np.sum((wps_r[0]-r_position)**2, axis=-1))
            r_t = r_distance/speed
            int_p_alt_aircraft_1 = alt+alt_speed*r_t
            fplan = flight_plans[flight_np[0]]['fplans'][flight_plans[flight_np[0]]['current']][1]
            flight_plan_alt_1 = fplan['altitude'].values[wp_idx_r]
            if wp_idx_r < wps_r.shape[0] - 1:
                wp_r = fplan[['x', 'y', 'altitude', 'timestamp']].values[wp_idx_r]
                wp_r_next = fplan[['x', 'y', 'altitude', 'timestamp']].values[wp_idx_r+1]
                flight_plan_course_1, _ = mybearing(wp_r[:2], wp_r_next[:2])
                dist = np.sqrt(np.sum((wp_r_next[:2] - wp_r[:2])**2, axis=-1))
                time_diff = (wp_r_next[3] - wp_r[3])
                if time_diff == 0:
                    time_diff = 30
                flight_plan_h_speed_1 = dist/time_diff
                flight_plan_v_speed_1 = (wp_r_next[2] - wp_r[2])/time_diff
            elif wp_idx_r > 0:
                wp_r = fplan[['x', 'y', 'altitude', 'timestamp']].values[wp_idx_r]
                wp_prev = fplan[['x', 'y', 'altitude', 'timestamp']].values[wp_idx_r-1]
                flight_plan_course_1, _ = mybearing(wp_prev[:2], wp_r[:2])
                dist = np.sqrt(np.sum((wp_r[:2] - wp_prev[:2])**2, axis=-1))
                time_diff = (wp_r[3] - wp_prev[3])
                if time_diff == 0:
                    time_diff = 30
                flight_plan_h_speed_1 = dist/time_diff
                flight_plan_v_speed_1 = (wp_r[2] - wp_prev[2])/time_diff

        wps_r = np.concatenate([np.array([position, r_position]), wps_r], axis=0).astype(np.float64)

    projection_info = [fplan_used_flag1, fplan_used1,
                       intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,
                       flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1]

    line_segments_np = geometry_utils.poly_line_segments(wps_r, dim2=2, convex=False)

    tw = env_config['tw']
    if alt_speed != 0:
        time_to_final_alt1, final_alt1 = time_to_nextfl(alt, alt_speed)
        tw = time_to_final_alt1

    enriched_segments_w_alt_np = \
        flight_plan_utils.enrich_segments_w_time(line_segments_np, timestamp, speed, alt, alt_speed, tw)
    res_id = 'no_resolution'
    if res_act_id is not None:
        res_id = res_act_id.split('_')[2:]
        res_id = '_'.join(res_id)
    projection_ID = str(int(timestamp)) + '_' + str(int(flight_np[0])) + '_' + res_id

    if enriched_segments_w_alt_np.shape[0] > 0:
        projection_points_1 = geometry_utils.line_segments_to_points(enriched_segments_w_alt_np, False)

        (projection_points_1[:, 0], projection_points_1[:, 1]) = \
            flight_plan_utils.update_lon_lat_from_x_y_vectorized(projection_points_1[:, 0], projection_points_1[:, 1])
        projection_points_1 = np.concatenate([[[projection_ID, int(flight_np[0]), timestamp, res_id, r_num]
                                               for r_num in range(projection_points_1.shape[0])], projection_points_1,],
                                             axis=1)
    else:
        projection_points_1 = np.empty((0, 0, 5))

    return projection_points_1, projection_ID, enriched_segments_w_alt_np, projection_info

def flight_projection(flight_np, flight_plans, timestamp, res_act_id):

    current_fplan_id1 = np.nan
    current_fplan_x_y1 = np.zeros([0, 2])

    res_id = 'no_resolution'
    if res_act_id is not None:
        res_id = res_act_id.split('_')[2:]
        res_id = '_'.join(res_id)
    projection_ID = str(int(timestamp)) + '_' + str(int(flight_np[0])) + '_' + res_id

    if len(flight_plans[flight_np[0]]['fplans']) > 0:
        current_fplan1 = flight_plans[flight_np[0]]['fplans'][flight_plans[flight_np[0]]['current']]

        current_fplan_x_y1 = current_fplan1[1][['x', 'y', 'altitude', 'timestamp']].values

        current_fplan_id1 = current_fplan1[2]

    enriched_segments_w_alt_1, fplan_used_flag1, fplan_used1, \
    intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1, \
    flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1 =\
    flight_plan_utils.project_flight(flight_np[0], current_fplan_id1, current_fplan_x_y1, flight_np[np.array([1, 2, 5])],
                                     flight_np[np.array([3, 4, 6])], timestamp, env_config['tw'], projection_ID)

    projection_info = [fplan_used_flag1, fplan_used1,
    intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,
    flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1]

    if enriched_segments_w_alt_1.shape[0] == 0:
        projection_points_1 = np.empty((0, 9))
        return projection_points_1, projection_ID, enriched_segments_w_alt_1, projection_info

    projection_points_1 = geometry_utils.line_segments_to_points(enriched_segments_w_alt_1, False)

    (projection_points_1[:, 0], projection_points_1[:, 1]) = \
        flight_plan_utils.update_lon_lat_from_x_y_vectorized(projection_points_1[:, 0], projection_points_1[:, 1])

    projection_points_1 = \
        np.concatenate([[[projection_ID, int(flight_np[0]), timestamp, res_id, r_num]
                         for r_num in range(projection_points_1.shape[0])], projection_points_1,],
                       axis=1)

    return projection_points_1, projection_ID, enriched_segments_w_alt_1, projection_info

def search_projection_ID(projection_ID, projection_dict, flight_i_features, flight_plans, timestamp, res_act_id_own,
                         res_type, flight_arr, flight_idx, flight_index, exit_wps, duration, action_starting_time_point,
                         executing_course_change_mask, just_finished_course_change_mask, tw, executing_direct_to_mask,
                         towards_wp_idxs, towards_wp_fp_idx, actions_impact_flag):

    if not projection_ID in projection_dict:

        if res_type == 'S2' or executing_course_change_mask[flight_index[flight_i_features[0]]['idx']] or \
                (just_finished_course_change_mask[flight_index[flight_i_features[0]]['idx']] and
                 res_type not in ['A1', 'A2', 'A3', 'A4']):
            projection_points_1, _, enriched_segments, projection_info = \
                course_change_projection(flight_i_features, flight_plans, flight_index, flight_arr, flight_idx,
                                         timestamp, res_act_id_own, exit_wps, duration, action_starting_time_point)
        elif res_type == 'A3' or executing_direct_to_mask[flight_index[flight_i_features[0]]['idx']]:
            projection_points_1, _, enriched_segments, projection_info =\
                direct_to_projection(flight_i_features, flight_plans, flight_index, flight_arr, flight_idx,
                                     timestamp, res_act_id_own, exit_wps, tw,
                                     int(towards_wp_idxs[flight_index[flight_i_features[0]]['idx']]),
                                     executing_direct_to_mask[flight_index[flight_i_features[0]]['idx']], res_type,
                                     int(towards_wp_fp_idx[flight_index[flight_i_features[0]]['idx']]), actions_impact_flag)
        else:
            projection_points_1, _, enriched_segments, projection_info = \
                flight_projection(flight_i_features, flight_plans, timestamp, res_act_id_own)
        projection_dict[projection_ID] = [projection_points_1, True, enriched_segments, projection_info]

    return projection_dict[projection_ID]

def find_flight_phase(RTkey, flight_plans, timestamp):

    if len(flight_plans[RTkey]['fplans']) == 0:
        return 'no fplan'
    idx = flight_plans[RTkey]['current']
    phases_df = flight_plans[RTkey]['fplans'][idx][5]
    phase = phases_df[(phases_df['TimeStart'] <= timestamp) &
                      (phases_df['TimeEnd'] >= timestamp)]['Phase'].values

    return_phase = 'unknown'
    if len(phase) > 0:
        return_phase = phase[0]

    return return_phase

def compute_conflicts(flights_np, flight_plans, timestamp, sectors, sectorData, res_act_ID, flight_index,
                      spec_flight_id, res_act_id, projection_dict, exit_wps, duration, action_starting_time_point,
                      executing_course_change_mask, just_finished_course_change_mask, executing_direct_to_mask,
                      towards_wp_idxs, towards_wp_fp_idx, actions_impact_flag):
    """
    Returns the following features of the edges between agents as np array:
      [flights_id_i, flights_id_j, tcpa, dcpa, intersection_angle,
       bearing_cpa_ownship_neighbour_relative, v_d_cpa,
       distance at crossing point, time_to_crossing_point, horizontal_distance, vertical_distance, considered_flag]
    :param flights_np: [0:RTkey,1:x,2:y,3:x_speed,4:y_speed,5:altitudeInFeets,6:alt_speed] #,7:max_altitude]
    :param point: [x,y,altitude]
    :return:
    """
    # note: max alt should be per flight
    max_alt = env_config['max_alt']
    min_alt = 1000
    possible_paths_intersection_filter_count = 0
    crossing_flights_filter_count = 0
    v_sep_minimum_filter_count = 0
    main_info_list = []
    conflict_info_list = []
    conflict_params_list = []
    projection_list = []
    edges = np.zeros((flights_np.shape[0]*(flights_np.shape[0]), 32))
    first_conf = True
    first_conflict_point_info = None
    cpa_info = None
    for i in range(flights_np.shape[0]):
        first_conf = True
        if (spec_flight_id is not None) and (flights_np[i][0] != spec_flight_id):
            continue
        flight_i_features = flights_np[i]
        flight_i_point = flight_i_features[np.array([1, 2, 5])]

        current_fplan_id1 = np.nan
        current_fplan_x_y1 = np.zeros([0, 2])

        if len(flight_plans[flight_i_features[0]]['fplans']) > 0:
            current_fplan1 = flight_plans[flight_i_features[0]]['fplans'][flight_plans[flight_i_features[0]]['current']]

            current_fplan_x_y1 = current_fplan1[1][['x', 'y', 'altitude', 'timestamp']].values

            current_fplan_id1 = current_fplan1[2]

        if spec_flight_id is None:
            j_start = i
        else:
            j_start = 0

        res_act_id_own = res_act_id

        res_id = 'no_resolution'
        res_type = 'no_resolution'
        if res_act_id is None:
            res_act_id_own = res_act_ID[flight_index[int(flight_i_features[0])]['idx']]
            if res_act_id_own == 0 or res_act_id_own == '0':
                res_act_id_own = None
            else:
                res_id = res_act_id_own.split('_')[2:]
                res_type = res_id[0]
                res_id = '_'.join(res_id)
        else:
            res_id = res_act_id.split('_')[2:]
            res_type = res_id[0]
            res_id = '_'.join(res_id)

        projection_ID_own = str(int(timestamp)) + '_' + str(int(flight_i_features[0])) + '_' + res_id
        if res_id == '':
            print(res_act_id)
            print(res_act_id_own)
            print(res_id)
            assert False

        projection_points_1_entry = search_projection_ID(projection_ID_own, projection_dict, flight_i_features,
                                                         flight_plans, timestamp, res_act_id_own, res_type,
                                                         flights_np, i, flight_index, exit_wps, duration,
                                                         action_starting_time_point, executing_course_change_mask,
                                                         just_finished_course_change_mask, env_config['tw'],
                                                         executing_direct_to_mask,
                                                         towards_wp_idxs, towards_wp_fp_idx, actions_impact_flag)

        if projection_points_1_entry[1]:
            projection_list.extend(projection_points_1_entry[0].tolist())
            projection_points_1_entry[1] = False

        for j in range(j_start, flights_np.shape[0]):

            edges_idx = i*(flights_np.shape[0])+j
            edges_ji_idx = j*(flights_np.shape[0])+i

            neighbour = flights_np[j]
            current_fplan_id2 = np.nan
            current_fplan_x_y2 = np.zeros([0, 2])
            if len(flight_plans[neighbour[0]]['fplans']) > 0:
                current_fplan2 = flight_plans[neighbour[0]]['fplans'][flight_plans[neighbour[0]]['current']]
                current_fplan_x_y2 = current_fplan2[1][['x', 'y', 'altitude', 'timestamp']].values
                current_fplan_id2 = current_fplan2[2]

            neighbour_point = neighbour[np.array([1, 2, 5])]
            edges[edges_idx, 0] = flight_i_features[0]
            edges[edges_idx, 1] = neighbour[0]
            v_sep_minimum = 1000
            if flight_i_point[2] >= 41000 or neighbour[5] >= 41000:
                v_sep_minimum = 2000
            if i == j:
                continue

            res_act_id_n = None
            res_id = 'no_resolution'
            res_type = 'no_resolution'
            if res_act_id is None:
                res_act_id_n = res_act_ID[flight_index[int(neighbour[0])]['idx']]
                if res_act_id_n == 0 or res_act_id_n == '0':
                    res_act_id_n = None
                else:
                    res_id = res_act_id_n.split('_')[2:]
                    res_type = res_id[0]
                    res_id = '_'.join(res_id)

            projection_ID_n = str(int(timestamp)) + '_' + str(int(neighbour[0])) + '_' + res_id

            if res_id == '':
                print(res_act_id)
                print(res_act_id_n)
                print(res_id)
                assert False

            projection_points_2_entry = search_projection_ID(projection_ID_n, projection_dict, neighbour,
                                                             flight_plans, timestamp, res_act_id_n, res_type,
                                                             flights_np, j, flight_index, exit_wps, duration,
                                                             action_starting_time_point, executing_course_change_mask,
                                                             just_finished_course_change_mask,
                                                             env_config['tw'], executing_direct_to_mask,
                                                             towards_wp_idxs, towards_wp_fp_idx, actions_impact_flag)
            if projection_points_2_entry[1]:
                projection_list.extend(projection_points_2_entry[0].tolist())
                projection_points_2_entry[1] = False

            horizontal_distance, vertical_distance = flight_distance(flight_i_point, neighbour[np.array([1, 2, 5])])
            edges[edges_idx, 9] = horizontal_distance
            edges[edges_idx, 10] = vertical_distance
            edges[edges_ji_idx, 0] = neighbour[0]
            edges[edges_ji_idx, 1] = flight_i_features[0]
            edges[edges_ji_idx, 9] = horizontal_distance
            edges[edges_ji_idx, 10] = vertical_distance
            conflict_flag = False
            crossing_point = None
            conflict_time = 0
            conflict_h_distance = horizontal_distance
            conflict_v_distance = vertical_distance
            bearing_cpa_ownship, _ = mybearing_speed_components(flight_i_features[3], flight_i_features[4])
            bearing_cpa_neighbour, _ = mybearing_speed_components(neighbour[3], neighbour[4])
            bearing_cpa_ownship_neighbour, _ = mybearing(flight_i_features[1:3], neighbour[1:3])
            bearing_cpa_j_i, _ = mybearing(neighbour[1:3], flight_i_features[1:3])
            intersection_angle_cpa = bearing_cpa_ownship - bearing_cpa_neighbour
            intersection_angle_ji_cpa = bearing_cpa_neighbour - bearing_cpa_ownship
            bearing_cpa_ownship_neighbour_relative = bearing_cpa_ownship - bearing_cpa_ownship_neighbour
            bearing_cpa_j_i_relative = bearing_cpa_neighbour - bearing_cpa_j_i
            neighbour_position_cpa = neighbour[np.array([1,2,5])]
            own_position_cpa = flight_i_features[np.array([1,2,5])]
            d_h_cp = np.nan
            t_to_crossing_point = np.nan
            considered = True
            if flight_i_features[5] >= 41000 or neighbour[5] >= 41000:
                v_sep_minimum_cpa = 2000
            else:
                v_sep_minimum_cpa = 1000
            if not (projection_points_1_entry[0].shape[0] == 0 or projection_points_2_entry[0].shape[0] == 0):

                conflict_flag, conflict_time, conflict_h_distance, conflict_v_distance,\
                own_position_cpa, neighbour_position_cpa,\
                crossing_point, t_to_crossing_point, d_h_cp, d_v_cp, conflict_velocity1, conflict_velocity2,\
                v_sep_minimum_cpa, considered, time_threshold,\
                enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
                first_conflict_point_info, last_conflict_point_info, cpa_info =\
                flight_plan_utils.detect_conflicts_w_fplan(flight_i_features[0], np.copy(projection_points_1_entry[2]),
                                                           current_fplan_x_y1, current_fplan_id1,
                                                           flight_i_features[np.array([1, 2, 5])],
                                                           flight_i_features[np.array([3, 4, 6])], neighbour[0],
                                                           np.copy(projection_points_2_entry[2]),
                                                           current_fplan_x_y2, current_fplan_id2,
                                                           neighbour[np.array([1, 2, 5])],
                                                           neighbour[np.array([3, 4, 6])],
                                                           timestamp, env_config['tw'],
                                                           env_config['horizontal_sep_minimum'],
                                                           env_config['first_conflict_point'])
                bearing_cpa_ownship, _ = mybearing_speed_components(conflict_velocity1[0], conflict_velocity1[1])
                bearing_cpa_neighbour, _ = mybearing_speed_components(conflict_velocity2[0], conflict_velocity2[1])

                bearing_cpa_ownship_neighbour, _ = mybearing(own_position_cpa, neighbour_position_cpa)
                bearing_cpa_j_i, _ = mybearing(neighbour_position_cpa, own_position_cpa)
                bearing_cpa_ownship_neighbour_relative = bearing_cpa_ownship - bearing_cpa_ownship_neighbour
                intersection_angle_cpa = bearing_cpa_ownship - bearing_cpa_neighbour
                intersection_angle_ji_cpa = bearing_cpa_neighbour - bearing_cpa_ownship
                bearing_cpa_j_i_relative = bearing_cpa_neighbour - bearing_cpa_j_i

            loss_flag = False
            if horizontal_distance < env_config['horizontal_sep_minimum']*1852 and vertical_distance < v_sep_minimum:
                loss_flag = True

            if conflict_flag or loss_flag:
                # preparing output start
                (fplan_used_flag1, fplan_used1,
                intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,
                flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1) = projection_points_1_entry[3]

                (fplan_used_flag2, fplan_used2,
                intersection_point_2, int_p_alt_aircraft_2, flight_plan_course_2,
                flight_plan_alt_2, flight_plan_h_speed_2, flight_plan_v_speed_2) = projection_points_2_entry[3]

                course1, _ = mybearing_speed_components(flight_i_features[3], flight_i_features[4])
                course2, _ = mybearing_speed_components(neighbour[3], neighbour[4])
                speed_h1 = np.sqrt(flight_i_features[3] ** 2 + flight_i_features[4] ** 2)
                speed_h2 = np.sqrt(neighbour[3] ** 2 + neighbour[4] ** 2)

                if crossing_point is None:
                    cross_point_x = None
                    cross_point_y = None
                else:
                    cross_point_x = crossing_point[0]
                    cross_point_y = crossing_point[1]

                if int(flight_i_features[0]) < int(neighbour[0]):
                    flight1 = flight_i_features
                    flight2 = neighbour
                    min_rtkey = int(flight_i_features[0])
                    max_rtkey = int(neighbour[0])
                else:
                    flight1 = neighbour
                    flight2 = flight_i_features
                    min_rtkey = int(neighbour[0])
                    max_rtkey = int(flight_i_features[0])
                    (fplan_used_temp, course_temp, speed_temp, fplan_used_flag_temp) = \
                        (fplan_used1, course1, speed_h1, fplan_used_flag1)
                    (fplan_used1, course1, speed_h1, fplan_used_flag1) = \
                        (fplan_used2, course2, speed_h2, fplan_used_flag2)
                    (fplan_used2, course2, speed_h2, fplan_used_flag2) = \
                        (fplan_used_temp, course_temp, speed_temp, fplan_used_flag_temp)

                res_act_id1 = None
                res_act_id2 = None
                res_act_t_type = None
                if res_act_id is not None:
                    if int(flight_i_features[0]) < int(neighbour[0]):
                        res_act_id1 = res_act_id
                    else:
                        res_act_id2 = res_act_id
                    flight_i_res_act = res_act_id
                    n_res_act = 0
                    res_act_t_type = 'foreseen'
                else:
                    flight_i_res_act = res_act_ID[flight_index[int(flight_i_features[0])]['idx']]
                    n_res_act = res_act_ID[flight_index[int(neighbour[0])]['idx']]
                    idx = flight_index[int(min_rtkey)]['idx']
                    res_act = res_act_ID[idx]

                    if res_act != 0:
                        res_act_id1 = res_act
                        res_act_t_type = 'issued'
                    idx = flight_index[int(max_rtkey)]['idx']
                    res_act = res_act_ID[idx]
                    if res_act != 0:
                        res_act_id2 = res_act
                        res_act_t_type = 'issued'
                # preparing output end
                if loss_flag:
                    phase1 = find_flight_phase(int(flight1[0]), flight_plans, timestamp)
                    phase2 = find_flight_phase(int(flight2[0]), flight_plans, timestamp)
                    conflict_ID = '_'.join([str(int(timestamp)), str(min_rtkey), str(max_rtkey)])

                    main_info = [int(timestamp), int(flight1[0]), int(flight2[0]), None,
                                 fplan_used1, None, fplan_used2, course1, course2, speed_h1, speed_h2,
                                 flight1[6], flight2[6], flight1[1], flight1[2],
                                 flight1[5], flight2[1], flight2[2],
                                 flight2[5], conflict_ID, phase1, phase2, 0, 'loss', res_act_id1,
                                 res_act_id2, res_act_t_type]
                    main_info_list.append(main_info)
                elif conflict_flag:
                    if first_conf and projection_points_1_entry[1]:
                        projection_list.extend(projection_points_1_entry[0].tolist())
                        projection_points_1_entry[1] = False

                    if conflict_time < 120:
                        c_type = 'alert'
                    else:
                        c_type = 'conflict'
                    conflict_ID = '_'.join([str(int(timestamp)), str(min_rtkey), str(max_rtkey)])
                    phase1 = find_flight_phase(int(flight1[0]), flight_plans, timestamp)
                    phase2 = find_flight_phase(int(flight2[0]), flight_plans, timestamp)
                    main_info = [int(timestamp), int(flight1[0]), int(flight2[0]), fplan_used_flag1, fplan_used1,
                                 fplan_used_flag2, fplan_used2, course1, course2, speed_h1, speed_h2,
                                 flight1[6], flight2[6], flight1[1], flight1[2],
                                 flight1[5], flight2[1], flight2[2],
                                 flight2[5], conflict_ID, phase1, phase2, time_threshold, c_type, res_act_id1,
                                 res_act_id2, res_act_t_type]
                    main_info_list.append(main_info)

                    sector1 = None
                    sector2 = None

                    for sectorID in sectors:
                        sector_flag2, _, _ = \
                            sectorData.point_in_sector(own_position_cpa[:3], timestamp, sectorID, True)
                        sector_flag1, _, _ = \
                            sectorData.point_in_sector(neighbour_position_cpa[:3], timestamp, sectorID, True)
                        if sector_flag1:
                            sector1 = sectorID
                        if sector_flag2:
                            sector2 = sectorID

                        if sector_flag1 and sector2:
                            break
                    if crossing_point is not None:
                        (crossing_point_lon, crossing_point_lat) = \
                            utils_global['inverse_transformer'].transform(cross_point_x, cross_point_y)
                    else:
                        (crossing_point_lon, crossing_point_lat) = (None, None)

                    if intersection_point_1 is not None:
                        (intersection_point_lon_1, intersection_point_lat_1) = \
                            utils_global['inverse_transformer'].transform(intersection_point_1[0],
                                                                          intersection_point_1[1])
                    else:
                        (intersection_point_lon_1, intersection_point_lat_1) = (None, None)

                    if intersection_point_2 is not None:
                        (intersection_point_lon_2, intersection_point_lat_2) = \
                            utils_global['inverse_transformer'].transform(intersection_point_2[0],
                                                                          intersection_point_2[1])
                    else:
                        (intersection_point_lon_2, intersection_point_lat_2) = (None, None)

                    if first_conflict_point_info is None:
                        first_conflict_point_info = [None]*15
                        last_conflict_point_info = [None]*9

                    if cpa_info is None:
                        cpa_info = [None]*9

                    if first_conflict_point_info is not None and last_conflict_point_info is None:
                        print(first_conflict_point_info)
                        print(last_conflict_point_info)
                        print(main_info)
                        assert False

                    if conflict_time == 0:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
                              'ALERT: CONFLICT TIME IS ZERO!!!\n'
                              '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

                    conflict_info = [conflict_ID, int(flight_i_features[0]), cpa_info[3], cpa_info[4],
                                     cpa_info[5], cpa_info[0], cpa_info[1], cpa_info[2],
                                     first_conflict_point_info[3], first_conflict_point_info[4],
                                     first_conflict_point_info[5],
                                     first_conflict_point_info[0], first_conflict_point_info[1],
                                     first_conflict_point_info[2],
                                     last_conflict_point_info[3], last_conflict_point_info[4],
                                     last_conflict_point_info[5],
                                     last_conflict_point_info[0], last_conflict_point_info[1],
                                     last_conflict_point_info[2],
                                     crossing_point_lon, crossing_point_lat,
                                     t_to_crossing_point, d_h_cp, d_v_cp, sector1, res_act_id1, res_act_id2,
                                     res_act_t_type, projection_ID_own]

                    conflict_info2 = [conflict_ID, int(neighbour[0]), cpa_info[6],
                                      cpa_info[7], cpa_info[8],
                                      cpa_info[0], cpa_info[1], cpa_info[2],
                                      first_conflict_point_info[6], first_conflict_point_info[7],
                                      first_conflict_point_info[8],
                                      first_conflict_point_info[0], first_conflict_point_info[1],
                                      first_conflict_point_info[2],
                                      last_conflict_point_info[6], last_conflict_point_info[7],
                                      last_conflict_point_info[8],
                                      last_conflict_point_info[0], last_conflict_point_info[1],
                                      last_conflict_point_info[2],
                                      crossing_point_lon, crossing_point_lat,
                                      t_to_crossing_point, d_h_cp, d_v_cp, sector2, res_act_id1, res_act_id2,
                                      res_act_t_type, projection_ID_n]

                    conflict_info_list.append(conflict_info)
                    conflict_info_list.append(conflict_info2)

                    conflict_params_1 = [conflict_ID, int(flight_i_features[0]), intersection_point_lon_1,
                                         intersection_point_lat_1,
                                         int_p_alt_aircraft_1, course1, flight_plan_course_1,
                                         flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1, res_act_id1, res_act_id2, res_act_t_type]

                    conflict_params_2 = [conflict_ID, int(neighbour[0]), intersection_point_lon_2,
                                         intersection_point_lat_2,
                                         int_p_alt_aircraft_2, course2, flight_plan_course_2,
                                         flight_plan_alt_2, flight_plan_h_speed_2, flight_plan_v_speed_2, res_act_id1,
                                         res_act_id2, res_act_t_type]
                    conflict_params_list.append(conflict_params_1)
                    conflict_params_list.append(conflict_params_2)

                    if projection_points_2_entry[1]:
                        projection_list.extend(projection_points_2_entry[0].tolist())
                        projection_points_2_entry[1] = False

            if first_conflict_point_info is None:
                first_conflict_point_info = [None] * 15
                last_conflict_point_info = [None] * 9

            if cpa_info is None:
                cpa_info = [None]*9

            features_np = np.array([flight_i_features[0], neighbour[0], conflict_time, conflict_h_distance,
                                    intersection_angle_cpa,
                                    bearing_cpa_ownship_neighbour_relative,
                                    neighbour_position_cpa[2]-own_position_cpa[2]])

            features_np = np.append(features_np, d_h_cp)
            features_np = np.append(features_np, t_to_crossing_point)
            features_np = np.append(features_np, horizontal_distance)
            features_np = np.append(features_np, vertical_distance)
            features_np = np.append(features_np, v_sep_minimum_cpa)
            features_np = np.append(features_np, float(considered))
            features_np = np.append(features_np, v_sep_minimum)
            # time, h_d, v_d
            features_np = np.append(features_np, first_conflict_point_info[:3])
            # position own
            features_np = np.append(features_np, first_conflict_point_info[3:6])
            # position neighbour
            features_np = np.append(features_np, first_conflict_point_info[6:9])
            #velocity_own
            features_np = np.append(features_np, first_conflict_point_info[9:12])
            # velocity_neighbour
            features_np = np.append(features_np, first_conflict_point_info[12:15])
            # time, h_d, v_d
            features_np = np.append(features_np, cpa_info[:3])

            edges[edges_idx] = features_np
            features_ji_np = np.copy(features_np)
            features_ji_np[0] = neighbour[0]
            features_ji_np[1] = flight_i_features[0]
            features_ji_np[4] = intersection_angle_ji_cpa
            features_ji_np[5] = bearing_cpa_j_i_relative
            features_ji_np[6] = own_position_cpa[2] - neighbour_position_cpa[2]

            #position j
            features_ji_np[17:20] = first_conflict_point_info[6:9]
            #position i
            features_ji_np[20:23] = first_conflict_point_info[3:6]
            #velocity j
            features_ji_np[23:26] = first_conflict_point_info[12:15]
            #velocity i
            features_ji_np[26:29] = first_conflict_point_info[9:12]

            edges[edges_ji_idx] = features_ji_np

        if res_act_id is not None:
            if first_conf and projection_points_1_entry[1]:
                projection_list.extend(projection_points_1_entry[0].tolist())
                projection_points_1_entry[1] = False
                first_conf = False

    return main_info_list, conflict_info_list, conflict_params_list, projection_list, edges


def compute_edges(flights_np, d_angles, d_speeds, h_separation_threshold, t_w):
    """
    Returns the following features of the edges between agents as np array:
      [flights_id_i, flights_id_j, tcpa, dcpa, intersection_angle,
       bearing_cpa_ownship_neighbour_relative, v_d_cpa,
       distance at crossing point, time_to_crossing_point, horizontal_distance, vertical_distance, considered_flag]
    :param flights_np: [0:RTkey,1:x,2:y,3:x_speed,4:y_speed,5:altitudeInFeets,6:alt_speed] #,7:max_altitude]
    :param point: [x,y,altitude]
    :return:
    """
    # note: max alt should be per flight
    max_alt = 42000
    min_alt = 1000
    possible_paths_intersection_filter_count = 0
    crossing_flights_filter_count = 0
    v_sep_minimum_filter_count = 0

    edges = np.zeros((flights_np.shape[0]*(flights_np.shape[0]), 14))

    alternative_h_speed_components = compute_alternative_speed_components(flights_np[:, 3:5], d_angles, d_speeds)

    for i in range(flights_np.shape[0]):

        m = 'w'
        m1 = 'w'

        flight_i_features = flights_np[i]

        flight_i_point = flight_i_features[np.array([1, 2, 5])]
        alt_speed = flight_i_features[6]
        h_speed_components = alternative_h_speed_components[i]
        ownship_max_alt = max_alt
        velocity = h_speed_components[0]

        for j in range(i, flights_np.shape[0]):

            edges_idx = i*(flights_np.shape[0])+j
            edges_ji_idx = j*(flights_np.shape[0])+i

            neighbour = flights_np[j]
            neighbour_point = neighbour[np.array([1, 2, 5])]
            edges[edges_idx, 0] = flight_i_features[0]
            edges[edges_idx, 1] = neighbour[0]
            if i == j:
                continue
            horizontal_distance, vertical_distance = flight_distance(flight_i_point, neighbour[np.array([1, 2, 5])])
            edges[edges_idx, 9] = horizontal_distance
            edges[edges_idx, 10] = vertical_distance
            edges[edges_ji_idx, 0] = neighbour[0]
            edges[edges_ji_idx, 1] = flight_i_features[0]
            edges[edges_ji_idx, 9] = horizontal_distance
            edges[edges_ji_idx, 10] = vertical_distance

            neighbour_max_alt = max_alt

            v_sep_minimum = 1000
            if flight_i_point[2] >= 41000 or neighbour[5] >= 41000:
                v_sep_minimum = 2000

            m1 = 'a'

            cpa = []
            d_cpa_components = []

            neighbour_alt_speed_components = alternative_h_speed_components[j]

            for n_id, neighbour_speed_components in enumerate(neighbour_alt_speed_components):

                for v_id, velocity in enumerate(h_speed_components):
                    tcpa, dcpa, d, ownship_location, neighbour_location, h_cpa_flag = \
                        compute_cpa(flight_i_point[0:2], neighbour[1:3], velocity, neighbour_speed_components)
                    cpa_lst = [tcpa, d]
                    cpa_lst.extend(ownship_location)
                    cpa_lst.extend(neighbour_location)
                    cpa_lst.extend([v_id, n_id])

                    cpa.append(cpa_lst)
                    d_cpa_components.append(dcpa)

            # find closest distance
            cpa_np = np.array(cpa)
            cpa_idx = np.argmin(cpa_np[:, 1])

            tcpa = cpa[cpa_idx][0]
            dcpa = cpa[cpa_idx][1]
            own_speed_idx = int(cpa[cpa_idx][6])
            neighbour_speed_idx = int(cpa[cpa_idx][7])

            ownship_cpa_location = cpa_np[cpa_idx][2:4]

            neighbour_cpa_location = cpa_np[cpa_idx][4:6]

            speed_components = h_speed_components[own_speed_idx]
            neighbour_speed_components = neighbour_alt_speed_components[neighbour_speed_idx]

            v_tcpa, v_dcpa, v_d, ownship_altitude, neighbour_altitude, v_cpa_flag = \
                compute_cpa(np.array([flight_i_point[2]]), np.array([neighbour[5]]),
                            np.array([alt_speed]), np.array([neighbour[6]]))

            detect_conflicts(flight_i_point, neighbour_point, np.append(speed_components, [alt_speed]),
                             np.append(neighbour_speed_components, [neighbour[6]]), tcpa, dcpa, v_tcpa, v_dcpa)

            crossing_distance, time_to_crossing_point, intersection_p_cpa_speed = \
                geometry_utils.distance_at_crossing_point(flight_i_point[:2],
                                                                  neighbour[1:3],
                                                                  speed_components,
                                                                  neighbour_speed_components)

            bearing_ownship, _ = mybearing_speed_components(speed_components[0], speed_components[1])

            bearing_neighbour, _ = mybearing_speed_components(neighbour_speed_components[0], neighbour_speed_components[1])

            intersection_angle = bearing_ownship - bearing_neighbour

            intersection_angle_ji = bearing_neighbour - bearing_ownship

            bearing_cpa_ownship_neighbour, _ = mybearing(ownship_cpa_location, neighbour_cpa_location)
            bearing_cpa_j_i, _ = mybearing(neighbour_cpa_location, ownship_cpa_location)

            bearing_cpa_ownship_neighbour_relative = bearing_ownship - bearing_cpa_ownship_neighbour

            bearing_cpa_j_i_relative = bearing_neighbour - bearing_cpa_j_i

            # warning might result to "bug" (negative altitude) if tcpa < 0
            ownship_alt_cpa = flight_i_point[2] + alt_speed * tcpa
            ownship_alt_cpa = min(ownship_max_alt, ownship_alt_cpa)
            ownship_alt_cpa = max(min_alt, ownship_alt_cpa)

            neighbour_alt_cpa = neighbour[5] + neighbour[6] * tcpa
            neighbour_alt_cpa = min(neighbour_max_alt, neighbour_alt_cpa)
            neighbour_alt_cpa = max(min_alt, neighbour_alt_cpa)
            v_sep_minimum_cpa = 1000.
            if neighbour_alt_cpa >= 41000 or ownship_alt_cpa >= 41000:
                v_sep_minimum_cpa = 2000.

            v_d_cpa = neighbour_alt_cpa - ownship_alt_cpa
            v_d_cpa_ji = -v_d_cpa

            features_np = np.array([flight_i_features[0], neighbour[0], tcpa, dcpa, intersection_angle,
                                    bearing_cpa_ownship_neighbour_relative, v_d_cpa])

            features_np = np.append(features_np, crossing_distance)
            features_np = np.append(features_np, time_to_crossing_point)
            features_np = np.append(features_np, horizontal_distance)
            features_np = np.append(features_np, vertical_distance)
            features_np = np.append(features_np, v_sep_minimum_cpa)
            features_np = np.append(features_np, 1.)
            features_np = np.append(features_np, v_sep_minimum)

            edges[edges_idx] = features_np
            features_ji_np = np.copy(features_np)
            features_ji_np[0] = neighbour[0]
            features_ji_np[1] = flight_i_features[0]
            features_ji_np[4] = intersection_angle_ji
            features_ji_np[5] = bearing_cpa_j_i_relative
            features_ji_np[6] = v_d_cpa_ji
            edges[edges_ji_idx] = features_ji_np

    return edges, possible_paths_intersection_filter_count,\
           crossing_flights_filter_count, v_sep_minimum_filter_count


@njit
def compute_features_w_neighbour_uncertainty_parallel(neighbours, point, h_speed_components, alt_speed,
                                                      ownship_max_alt, d_angles, d_speeds, h_separation_threshold, t_w):
    """
    :param neighbours: [0:RTkey,1:x,2:y,3:x_speed,4:y_speed,5:altitudeInFeets,6:alt_speed,7:max_altitude]
    :param point: [x,y,altitude]
    :return:
    """

    possible_paths_intersection_filter_count = 0
    crossing_flights_filter_count = 0
    v_sep_minimum_filter_count = 0
    neighbours_features = np.zeros((len(neighbours), 15))

    n_i_list = []

    for n_i in range(len(neighbours)):

        neighbour = neighbours[n_i]

        v_sep_minimum = 1000
        if point[2] >= 41000 or neighbour[5] >= 41000:
            v_sep_minimum = 2000

        if abs(point[2]-neighbour[5]) >= v_sep_minimum and (alt_speed == 0 and neighbour[6] == 0):
            v_sep_minimum_filter_count += 1
            continue

        velocity = h_speed_components[0]
        passed, intersection_p = \
            geometry_utils.flights_passed_intersection(point[0:2], neighbour[1:3], velocity, neighbour[3:4])

        if intersection_p[0] == np.inf and intersection_p[1] == np.inf:
            pass
        else:
            if passed or not flights_cross_in_t_w(intersection_p, point[0:2], velocity, t_w):
                crossing_flights_filter_count +=1
                continue

        if not geometry_utils.possible_paths_intersect(point[:2], neighbour[1:3],
                                                       velocity, neighbour[3:5],
                                                       d_angles, d_speeds,
                                                       h_separation_threshold=h_separation_threshold,
                                                       t_w=t_w):
            possible_paths_intersection_filter_count += 1
            continue

        cpa = []
        d_cpa_components = []

        neighbour_alt_speed_components = compute_alternative_neighbour_speed_components_parallel(neighbour[3],
                                                                                                 neighbour[4],
                                                                                                 d_angles, d_speeds)

        for n_id, neighbour_speed_components in enumerate(neighbour_alt_speed_components):

            for v_id, velocity in enumerate(h_speed_components):

                tcpa, dcpa, d, ownship_location, neighbour_location = \
                    compute_cpa(point[0:2], neighbour[1:3], velocity, neighbour_speed_components)

                cpa_lst = [tcpa, d]
                cpa_lst.extend(ownship_location)
                cpa_lst.extend(neighbour_location)
                cpa_lst.extend([v_id, n_id])

                cpa.append(cpa_lst)

                d_cpa_components.append(dcpa)


        # find closest distance
        cpa_np = np.array(cpa)
        cpa_idx = np.argmin(cpa_np[:, 1])

        tcpa = cpa[cpa_idx][0]
        dcpa = cpa[cpa_idx][1]
        own_speed_idx = int(cpa[cpa_idx][6])
        neighbour_speed_idx = int(cpa[cpa_idx][7])

        ownship_cpa_location = cpa_np[cpa_idx][2:4]
        neighbour_cpa_location = cpa_np[cpa_idx][4:6]

        speed_components = h_speed_components[own_speed_idx]
        neighbour_speed_components = neighbour_alt_speed_components[neighbour_speed_idx]

        crossing_distance, time_to_crossing_point = \
            geometry_utils.distance_at_crossing_point(point[:2], neighbour[3:5], speed_components,
                                                      neighbour_speed_components)

        bearing_ownship, _ = mybearing_speed_components(speed_components[0], speed_components[1])

        bearing_neighbour, _ = mybearing_speed_components(neighbour_speed_components[0], neighbour_speed_components[1])
        intersection_angle = (360 + (bearing_ownship - bearing_neighbour)) % 360

        bearing_cpa_ownship_neighbour, _ = mybearing(ownship_cpa_location, neighbour_cpa_location)

        bearing_cpa_ownship_neighbour_relative = (360 + (bearing_ownship - bearing_cpa_ownship_neighbour)) % 360

        # warning might result to "bug" (negative altitude) if tcpa < 0
        ownship_alt_cpa = point[2] + alt_speed * tcpa
        ownship_alt_cpa = min(ownship_max_alt, ownship_alt_cpa)
        neighbour_alt_cpa = neighbour[5] + neighbour[6] * tcpa
        neighbour_alt_cpa = min(neighbour[7], neighbour_alt_cpa)
        v_d_cpa = neighbour_alt_cpa - ownship_alt_cpa

        features_np = np.array([tcpa, dcpa, intersection_angle, bearing_cpa_ownship_neighbour_relative,
                                v_d_cpa, neighbour[0], neighbour_alt_cpa, neighbour[7]])

        features_np = np.append(features_np, ownship_cpa_location)
        features_np = np.append(features_np, neighbour_cpa_location)
        features_np = np.append(features_np, ownship_alt_cpa)
        features_np = np.append(features_np, crossing_distance)
        features_np = np.append(features_np, time_to_crossing_point)

        neighbours_features[n_i, :] = features_np
        n_i_list.append(n_i)

    return neighbours_features, n_i_list, possible_paths_intersection_filter_count,\
           crossing_flights_filter_count, v_sep_minimum_filter_count


if __name__ == "__main__":
    flight_distance_unit_testing()
    exit(0)
    for i in range(10):
        print(myfactorial(i))