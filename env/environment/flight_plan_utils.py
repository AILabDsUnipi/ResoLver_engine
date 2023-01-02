from datetime import datetime
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from numba import njit
import matplotlib.pyplot as plt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
import utils
import geometry_utils
import flight_utils
from env_config import env_config, numba_dict

fplan_utils_global = \
    {'fplan_path': '.',
     'debug_folder': '.'}

def lon_lat_from_x_y(x, y):
    lon, lat = utils.utils_global['inverse_transformer'].transform(x, y)

    return lon, lat


update_lon_lat_from_x_y_vectorized = np.vectorize(lon_lat_from_x_y)

def write_course_fplan_line_segments(course_fplan_line_segments, flightKey):
    update_lon_lat_from_x_y_vectorized = np.vectorize(lon_lat_from_x_y)

    lon, lat = update_lon_lat_from_x_y_vectorized(course_fplan_line_segments[:, :, 0, np.newaxis],
                                                  course_fplan_line_segments[:, :, 1, np.newaxis])

    lon_lat = np.concatenate([lon, lat], axis=-1)
    lon_lat = np.append(lon_lat[:, 0], [lon_lat[-1, 1]], axis=0)

    time = course_fplan_line_segments[:, :, 2, np.newaxis]
    time = np.append(time[:, 0], [time[-1, 1]], axis=0)
    lon_lat_time = np.concatenate([lon_lat, time], axis=1)

    np.savetxt(fname='testing_fplan_projection_course_fplan'+str(flightKey)+'.csv', X = lon_lat_time, delimiter=',',
               comments='', header='lon,lat,time')

def find_h_position_at_t_from_segments(enriched_segments, t):
    t_in_segments = False
    position = np.array([0., 0.])

    segment_velocity = np.array([0., 0.])
    for enriched_segment in enriched_segments:

        if t < enriched_segment[0, 2]:
            break
        if t > enriched_segment[1, 2]:
            continue

        segment_velocity = (enriched_segment[1, :2] - enriched_segment[0, :2]) / \
                           np.abs((enriched_segment[1, 2] - enriched_segment[0, 2]))

        dt = t - enriched_segment[0, 2]
        position = enriched_segment[0, :2] + segment_velocity * dt

        t_in_segments = True

    return position, t_in_segments, segment_velocity

def conflict_search_around_vcpa_bu(alt2_vcpa, alt1_vcpa, alt_speed1, alt_speed2, v_sep_minimum, time, tw_init,
                                   enriched_segments_1, enriched_segments_2, timestamp, h_sep_minimum, dt=5,
                                   fplan=None, fkey2=None, first_conflict_point=False):
    if time > tw_init:
        time = tw_init

    if enriched_segments_2[-1, -1, -1] < time+timestamp:
        time = enriched_segments_2[-1, -1, -1] - timestamp

    if enriched_segments_1[-1, -1, -1] < time+timestamp:
        time = enriched_segments_1[-1, -1, -1] - timestamp

    tw = min([enriched_segments_2[-1, -1, -1] - timestamp, enriched_segments_1[-1, -1, -1] - timestamp, tw_init])

    conflict_time = time
    conflict = False
    conflict_h_distance = 0
    conflict_v_distance = 0
    v_distance = np.abs(alt2_vcpa - alt1_vcpa)
    t_3dcpa = time

    p1_at_time, t_in_segments1, segment_velocity1 = \
        find_h_position_at_t_from_segments(enriched_segments_1, time+timestamp)
    p2_at_time, t_in_segments2, segment_velocity2 = \
        find_h_position_at_t_from_segments(enriched_segments_2, time+timestamp)
    dist_3d_min_velocity1 = segment_velocity1
    dist_3d_min_velocity2 = segment_velocity2

    if not (t_in_segments1 and t_in_segments2):
        print('Suspicious!')
        print(v_distance)
        print(timestamp)
        print(time)
        print(time+timestamp)
        print(enriched_segments_1)
        print(enriched_segments_2)
        print(fkey2)
        assert False
    h_distance = np.sqrt(np.sum((p1_at_time - p2_at_time)**2))
    dist_3d_min = np.sqrt(h_distance ** 2 + v_distance ** 2)
    t_3dcpa_h_dist = h_distance
    t_3dcpa_v_dist = v_distance
    own_position_cpa = np.append(p1_at_time, alt1_vcpa)
    neighbour_position_cpa = np.append(p2_at_time, alt2_vcpa)

    while v_distance < v_sep_minimum and time <= tw and time >= 0:
        if h_distance < h_sep_minimum * 1852:
            conflict = True
            conflict_time = time
            conflict_h_distance = h_distance
            conflict_v_distance = v_distance
            own_position_cpa = np.append(p1_at_time, alt1_vcpa)
            neighbour_position_cpa = np.append(p2_at_time, alt2_vcpa)
            if not first_conflict_point:
                break

        alt1_vcpa = alt1_vcpa + alt_speed1 * dt
        alt2_vcpa = alt2_vcpa + alt_speed2 * dt
        v_distance = np.abs(alt2_vcpa-alt1_vcpa)
        if alt1_vcpa >= 41000 or alt2_vcpa >= 41000:
            v_sep_minimum = 2000
        else:
            v_sep_minimum = 1000

        p1_at_time, t_in_segments1, segment_velocity1 = \
            find_h_position_at_t_from_segments(enriched_segments_1, time+timestamp)
        p2_at_time, t_in_segments2, segment_velocity2 = \
            find_h_position_at_t_from_segments(enriched_segments_2, time+timestamp)

        if not (t_in_segments1 and t_in_segments2):
            break

        h_distance = np.sqrt(np.sum((p1_at_time - p2_at_time)**2))
        dist_3d = np.sqrt(h_distance ** 2 + v_distance ** 2)
        if dist_3d < dist_3d_min:
            dist_3d_min = dist_3d_min
            t_3dcpa = time
            t_3dcpa_h_dist = h_distance
            t_3dcpa_v_dist = v_distance
            dist_3d_min_velocity1 = segment_velocity1
            dist_3d_min_velocity2 = segment_velocity2
            own_position_cpa = np.append(p1_at_time, alt1_vcpa)
            neighbour_position_cpa = np.append(p2_at_time, alt2_vcpa)

        time += dt

    if not conflict:
        return conflict, t_3dcpa, t_3dcpa_h_dist, t_3dcpa_v_dist, own_position_cpa, neighbour_position_cpa,\
               dist_3d_min_velocity1, dist_3d_min_velocity2

    return conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa,\
           segment_velocity1, segment_velocity2

def conflict_search_around_vcpa(alt2_vcpa, alt1_vcpa, alt_speed1,
                                alt_speed2, v_sep_minimum, time, tw_init,
                                enriched_segments_1, enriched_segments_2, timestamp,
                                h_sep_minimum, dt=5, fplan=None, fkey2=None, first_conflict_point=False):
    conflict_time = time
    conflict_h_distance = None
    conflict_v_distance = None
    own_position_cpa = None
    neighbour_position_cpa = None
    fd_segment_velocity1 = None
    fd_segment_velocity2 = None

    first_detected_conflict = True
    if time > tw_init:
        time = tw_init

    if enriched_segments_2[-1, -1, -1] < time+timestamp:
        time = enriched_segments_2[-1, -1, -1] - timestamp

    if enriched_segments_1[-1, -1, -1] < time+timestamp:
        time = enriched_segments_1[-1, -1, -1] - timestamp

    tw = min([enriched_segments_2[-1, -1, -1] - timestamp, enriched_segments_1[-1, -1, -1] - timestamp, tw_init])
    conflict_time = time
    conflict = False
    conflict_h_distance = 0
    conflict_v_distance = 0
    v_distance = np.abs(alt2_vcpa - alt1_vcpa)

    t_3dcpa_v_dist = v_distance

    dist_3d_min = 0
    i = 0

    while (v_distance < v_sep_minimum and time <= tw and time >= 0) or i == 0:

        v_distance = np.abs(alt2_vcpa-alt1_vcpa)
        if alt1_vcpa >= 41000 or alt2_vcpa >= 41000:
            v_sep_minimum = 2000
        else:
            v_sep_minimum = 1000

        p1_at_time, t_in_segments1, segment_velocity1 = find_h_position_at_t_from_segments(enriched_segments_1,
                                                                                           time+timestamp)
        p2_at_time, t_in_segments2, segment_velocity2 = find_h_position_at_t_from_segments(enriched_segments_2,
                                                                                           time+timestamp)

        if not (t_in_segments1 and t_in_segments2):
            if i == 0:
                print('Suspicious!')
                print(v_distance)
                print(timestamp)
                print(time)
                print(time + timestamp)
                print(enriched_segments_1)
                print(enriched_segments_2)
                print(fkey2)
                assert False

            break

        h_distance = np.sqrt(np.sum((p1_at_time - p2_at_time)**2))
        dist_3d = np.sqrt(h_distance ** 2 + (v_distance*0.3048) ** 2)

        if i == 0 or dist_3d < dist_3d_min:
            dist_3d_min = dist_3d
            t_3dcpa = time
            t_3dcpa_h_dist = h_distance
            t_3dcpa_v_dist = v_distance
            dist_3d_min_velocity1 = segment_velocity1
            dist_3d_min_velocity2 = segment_velocity2
            own_position_cpa_3d_min = np.append(p1_at_time, alt1_vcpa)
            neighbour_position_cpa_3d_min = np.append(p2_at_time, alt2_vcpa)

        if h_distance < h_sep_minimum * 1852 and v_distance < v_sep_minimum:
            conflict = True
            if first_detected_conflict:
                conflict_time = time
                conflict_h_distance = h_distance
                conflict_v_distance = v_distance
                own_position_cpa = np.append(p1_at_time, alt1_vcpa)
                neighbour_position_cpa = np.append(p2_at_time, alt2_vcpa)
                first_detected_conflict = False
                fd_segment_velocity1 = segment_velocity1
                fd_segment_velocity2 = segment_velocity2
                last_own_position_cpa = own_position_cpa
                last_neighbour_position_cpa = neighbour_position_cpa

            first_conflict_time = time
            first_conflict_h_distance = h_distance
            first_conflict_v_distance = v_distance
            first_own_position_cpa = np.append(p1_at_time, alt1_vcpa)
            first_neighbour_position_cpa = np.append(p2_at_time, alt2_vcpa)
            first_segment_velocity1 = segment_velocity1
            first_segment_velocity2 = segment_velocity2

            if not first_conflict_point:
                break

        elif conflict:
            break

        alt1_vcpa = alt1_vcpa + alt_speed1 * dt
        alt2_vcpa = alt2_vcpa + alt_speed2 * dt
        i += 1
        time += dt

    cpa_info = np.concatenate((np.array([t_3dcpa,
                                         t_3dcpa_h_dist,
                                         t_3dcpa_v_dist]),
                                         own_position_cpa_3d_min,
                                         neighbour_position_cpa_3d_min,),
                              axis=0)

    if not conflict:
        return conflict, t_3dcpa, t_3dcpa_h_dist, t_3dcpa_v_dist, own_position_cpa_3d_min, \
               neighbour_position_cpa_3d_min, dist_3d_min_velocity1, dist_3d_min_velocity2, \
               None, None, cpa_info


    first_conflict_point_info = \
        np.concatenate((np.array([first_conflict_time, first_conflict_h_distance, first_conflict_v_distance]),
                        first_own_position_cpa,
                        first_neighbour_position_cpa,
                        first_segment_velocity1,
                        np.array([alt_speed1]),
                        first_segment_velocity2,
                        np.array([alt_speed2])),
                       axis=0)

    last_conflict_point_info = \
        np.concatenate((np.array([conflict_time, conflict_h_distance, conflict_v_distance]),
                        last_own_position_cpa,
                        last_neighbour_position_cpa,
                        fd_segment_velocity1,
                        np.array([alt_speed1]),
                        fd_segment_velocity2,
                        np.array([alt_speed2])),
                       axis=0)

    return conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa,\
           fd_segment_velocity1, fd_segment_velocity2, first_conflict_point_info, last_conflict_point_info, cpa_info

def compute_cpa_w_fplans_w_v_speed(fKey1, enriched_segments_w_alt_1, current_fplan_x_y1, current_fplan_id1, p1, velocity1,
                                   fKey2, enriched_segments_w_alt_2, current_fplan_x_y2, current_fplan_id2, p2, velocity2,
                                   timestamp, tw, horizontal_sep_minimum, first_conflict_point):

    if enriched_segments_w_alt_1[-1, 1, 2] < enriched_segments_w_alt_2[-1, 1, 2]:
        enriched_segments_w_alt_2 = cut_segments(enriched_segments_w_alt_2, enriched_segments_w_alt_1[-1, 1, 2])
    elif enriched_segments_w_alt_2[-1, 1, 2] < enriched_segments_w_alt_1[-1, 1, 2]:
        enriched_segments_w_alt_1 = cut_segments(enriched_segments_w_alt_1, enriched_segments_w_alt_2[-1, 1, 2])

    enriched_segments_1 = enriched_segments_w_alt_1[:, :, :3]
    enriched_segments_2 = enriched_segments_w_alt_2[:, :, :3]

    proj_time_horizon = min(enriched_segments_1[-1, 1, 2], enriched_segments_2[-1, 1, 2]) - timestamp

    crosses, crossing_point, t_at_crossing_point, d_h_cp = crossing_point_w_fplans(enriched_segments_1,
                                                                                   enriched_segments_2)
    t_to_crossing_point = None
    d_v_cp = None
    if crosses:
        t_to_crossing_point = t_at_crossing_point-timestamp

        alt1_cp = p1[2] + velocity1[2] * t_to_crossing_point
        alt2_cp = p2[2] + velocity2[2] * t_to_crossing_point
        d_v_cp = np.abs(alt1_cp-alt2_cp)

    t_v_cpa = utils.compute_tcpa(np.array([p1[2]]), np.array([p2[2]]),
                                 np.array([velocity1[2]]), np.array([velocity2[2]]))

    if t_v_cpa > tw:
        t_v_cpa = tw

    if t_v_cpa < 0:
        t_v_cpa = 0

    alt1_vcpa = p1[2] + velocity1[2] * t_v_cpa
    alt2_vcpa = p2[2] + velocity2[2] * t_v_cpa

    v_sep_minimum = 1000
    if alt1_vcpa >= 41000 or alt2_vcpa >= 41000:
        v_sep_minimum = 2000

    conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa, \
    segment_velocity1, segment_velocity2, first_conflict_point_info1, last_conflict_point_info1, cpa_info1 = \
        conflict_search_around_vcpa(alt2_vcpa, alt1_vcpa, velocity1[2],
                                    velocity2[2], v_sep_minimum, t_v_cpa, tw,
                                    enriched_segments_1, enriched_segments_2, timestamp,
                                    horizontal_sep_minimum, -5, first_conflict_point=first_conflict_point)

    cpa_info = cpa_info1
    first_conflict_point_info = None
    last_conflict_point_info = None
    if not conflict or first_conflict_point:
        conflict2, conflict_time2, conflict_h_distance2, conflict_v_distance2, own_position_cpa2,\
        neighbour_position_cpa2, segment_velocity21, segment_velocity22, last_conflict_point_info2,\
        first_conflict_point_info2, cpa_info2 = \
            conflict_search_around_vcpa(alt2_vcpa, alt1_vcpa, velocity1[2],
                                        velocity2[2], v_sep_minimum, t_v_cpa, tw,
                                        enriched_segments_1, enriched_segments_2, timestamp,
                                        horizontal_sep_minimum, +5, first_conflict_point=first_conflict_point)

        if first_conflict_point_info1 is not None and first_conflict_point_info2 is not None:
            if first_conflict_point_info1[0] < first_conflict_point_info2[2]:
                first_conflict_point_info = first_conflict_point_info1
            else:
                first_conflict_point_info = first_conflict_point_info2
        elif first_conflict_point_info1 is not None:
            first_conflict_point_info = first_conflict_point_info1
        elif first_conflict_point_info2 is not None:
            first_conflict_point_info = first_conflict_point_info2

        if last_conflict_point_info1 is not None and last_conflict_point_info2 is not None:
            if last_conflict_point_info1[0] > last_conflict_point_info2[2]:
                last_conflict_point_info = last_conflict_point_info1
            else:
                last_conflict_point_info = last_conflict_point_info2
        elif last_conflict_point_info1 is not None:
            last_conflict_point_info = last_conflict_point_info1
        elif last_conflict_point_info2 is not None:
            last_conflict_point_info = last_conflict_point_info2

        if conflict2 == conflict:
            if np.sqrt(cpa_info1[1]**2+(cpa_info1[2]*0.3048)**2) > np.sqrt(cpa_info2[1]**2+(cpa_info2[2]*0.3048)**2):
                cpa_info = cpa_info2
        elif conflict:
            cpa_info = cpa_info1
        elif conflict2:
            cpa_info = cpa_info2

        if not conflict2 and not conflict:
            dist_3d_1 = np.sqrt(conflict_h_distance ** 2 + conflict_v_distance ** 2)
            dist_3d_2 = np.sqrt(conflict_h_distance2 ** 2 + conflict_v_distance2 ** 2)
            if dist_3d_1 > dist_3d_2:
                (conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa,
                 neighbour_position_cpa, segment_velocity1, segment_velocity2) = \
                    (conflict2, conflict_time2, conflict_h_distance2, conflict_v_distance2, own_position_cpa2,
                     neighbour_position_cpa2, segment_velocity21, segment_velocity22)

        elif not conflict:
            (conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa,
             neighbour_position_cpa, segment_velocity1, segment_velocity2) = \
                (conflict2, conflict_time2, conflict_h_distance2, conflict_v_distance2, own_position_cpa2,
                 neighbour_position_cpa2, segment_velocity21, segment_velocity22)

    return conflict, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa,\
           crossing_point, t_to_crossing_point, d_h_cp, \
           d_v_cp, segment_velocity1, segment_velocity2, v_sep_minimum, proj_time_horizon, \
           enriched_segments_w_alt_1, enriched_segments_w_alt_2,\
           first_conflict_point_info, last_conflict_point_info, cpa_info

def project_flight(fKey1, current_fplan_id1, current_fplan_x_y1, p1, velocity1, timestamp, tw, projectionID):

    if velocity1[2] != 0:
        time_to_final_alt1, final_alt1 = utils.time_to_nextfl(p1[2], velocity1[2])
        tw = time_to_final_alt1

    enriched_segments_w_alt_1, fplan_used_flag1, fplan_used1,\
    intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,\
    flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1 = \
        course_fplan_intersection(fKey1, current_fplan_id1,
                                  current_fplan_x_y1, p1[:2], p1[2],
                                  velocity1[:2], velocity1[2],
                                  timestamp, tw)

    if fplan_used_flag1 == False and current_fplan_x_y1.shape[0] > 0:
        enriched_segments_w_alt_1, fplan_used_flag1, fplan_used1, \
        intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1, \
        flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1 = \
        point_close_to_segment(fKey1, current_fplan_id1, current_fplan_x_y1, p1[:2], p1[2],
                               velocity1[:2], velocity1[2], timestamp, tw, projectionID)

    return enriched_segments_w_alt_1, fplan_used_flag1, fplan_used1,\
           intersection_point_1, int_p_alt_aircraft_1, flight_plan_course_1,\
           flight_plan_alt_1, flight_plan_h_speed_1, flight_plan_v_speed_1

def detect_conflicts_w_fplan(fKey1, projection_segments1, current_fplan_x_y1, current_fplan_id1, p1, velocity1,
                             fKey2, projection_segments2, current_fplan_x_y2, current_fplan_id2, p2, velocity2,
                             timestamp, tw, horizontal_sep_minimum, first_conflict_point):

    velocity_2d1 = velocity1[:2]
    velocity_2d2 = velocity2[:2]
    nan_arr = np.array([np.nan, np.nan, np.nan]).astype(np.float64)
    considered = True
    conflict_flag = False
    ## debug section ##
    # p1[2] = 42000
    # p2 = np.copy(p1)
    # p2[2] -= 2000
    # velocity1[2] = 0
    # velocity2 = np.copy(velocity1)
    # velocity2[2] = 5
    #debug section end
    current_threshold = 1000
    future_v_sep_minimum = 1000
    if p1[2] >= 41000 or p2[2] >= 41000:
        current_threshold = 2000
    v_sep_minimum = current_threshold

    final_alt1 = current_alt1 = p1[2]
    final_alt2 = current_alt2 = p2[2]
    time_to_final_alt1 = 0
    time_to_final_alt2 = 0

    if velocity1[2] == velocity2[2]:

        conflict = True
        time_threshold = tw

        if velocity2[2] == 0:
            if p1[2] >= 41000 or p2[2] >= 41000:
                v_sep_minimum = 2000
            if np.abs(p1[2]-p2[2]) >= v_sep_minimum:
                conflict = False
                considered = False
                return conflict, np.nan, np.nan, np.abs(p1[2] - p2[2]), nan_arr, nan_arr, None, \
                       np.nan, np.nan, np.nan, nan_arr, nan_arr, v_sep_minimum, considered, None, \
                       None, None, None, None, None
        else:
            time_to_final_alt1, final_alt1 = utils.time_to_nextfl(p1[2], velocity1[2])
            time_to_final_alt2, final_alt2 = utils.time_to_nextfl(p2[2], velocity2[2])
            if final_alt1 >= 41000 or final_alt2 >= 41000:
                future_v_sep_minimum = 2000

            time_threshold = min(time_to_final_alt1, time_to_final_alt2)
            if (abs(final_alt1-final_alt2) >= future_v_sep_minimum and
                abs(p1[2]-p2[2]) >= current_threshold) or time_threshold <= 0.1:
                conflict = False
                considered = False
                return conflict, np.nan, np.nan, np.abs(p1[2] - p2[2]), nan_arr, nan_arr, None, \
                       np.nan, np.nan, np.nan, nan_arr, nan_arr, v_sep_minimum, considered, None, \
                       None, None, None, None, None

            if (p1[2] < 41000 and final_alt1 == 41000) or (p1[2] < 41000 and final_alt1 == 41000) or \
                    p1[2] > 41000 or p2[2] > 41000:
                if abs(p1[2] - p2[2]) < current_threshold:
                    conflict = True
                else:
                    conflict = False
                v_sep_minimum = current_threshold
            else:
                if abs(p1[2] - p2[2]) < future_v_sep_minimum:
                    conflict = True
                else:
                    conflict = False
                v_sep_minimum = future_v_sep_minimum

        tcpa, dcpa, ownship_location_cpa, neighbour_location_cpa, segment_velocity1, segment_velocity2,\
        h_conflict, crossing_point, t_at_crossing_point, d_h_cp, crosses, proj_time_horizon, \
        enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
        first_point_info, last_point_info = \
            compute_cpa_w_fplans(fKey1, projection_segments1, current_fplan_x_y1, current_fplan_id1, p1[:3],
                                 velocity1[:3],
                                 fKey2, projection_segments2, current_fplan_x_y2, current_fplan_id2, p2[:3],
                                 velocity2[:3],
                                 timestamp, time_threshold, horizontal_sep_minimum, first_conflict_point)

        if h_conflict and not conflict:
            if ((final_alt1 == 41000 and time_to_final_alt1 <= time_to_final_alt2) or \
                    (final_alt2 == 41000 and time_to_final_alt2 <= time_to_final_alt1)) and \
                    (current_alt1 < 41000 and current_alt2 < 41000):
                t = min(time_to_final_alt1, time_to_final_alt2)
                if abs(current_alt1 - current_alt2) >= 1000:
                    conflict = False
                    v_dist = ((current_alt1 + velocity1[2] * t) - (current_alt2 + velocity2[2] * t))
                    if abs(v_dist) < 2000:
                        if t >= first_conflict_point:

                            ownship_location_t, _, seg_vel1 = find_h_position_at_t_from_segments(enriched_segments_w_alt_1, timestamp+t)
                            neighbour_location_t, _, seg_vel2 = find_h_position_at_t_from_segments(enriched_segments_w_alt_2, timestamp+t)
                            dist = np.sqrt(np.sum((ownship_location_t-neighbour_location_t)**2))
                            if dist < horizontal_sep_minimum*1852:
                                tcpa = t
                                dcpa = dist
                                segment_velocity1 = seg_vel1
                                segment_velocity2 = seg_vel2
                                first_point_info = np.concatenate([np.array([tcpa, dcpa]),
                                                                   ownship_location_t, neighbour_location_t,
                                                                   segment_velocity1, segment_velocity2], axis=0)
                                last_point_info = np.copy(first_point_info)
                                ownship_location_cpa = ownship_location_t
                                neighbour_location_cpa = neighbour_location_t

                                conflict = True
                                v_sep_minimum = 2000

        if crosses:
            d_v_cp = np.abs(p1[2] - p2[2])
        else:
            d_v_cp = None
        own_cpa_alt = p1[2]+tcpa*velocity1[2]
        n_cpa_alt = p2[2] + tcpa * velocity2[2]

        if first_point_info is not None:
            first_point_own_cpa_alt = p1[2]+first_point_info[0]*velocity1[2]
            first_point_n_cpa_alt = p2[2] + first_point_info[0] * velocity2[2]
            last_point_own_cpa_alt = p1[2] + last_point_info[0] * velocity1[2]
            last_point_n_cpa_alt = p2[2] + last_point_info[0] * velocity2[2]
            first_point_info_w_alt = np.zeros((15))
            first_point_info_w_alt[:2] = first_point_info[:2]
            first_point_info_w_alt[2] = np.abs(first_point_n_cpa_alt-first_point_own_cpa_alt)
            first_point_info_w_alt[3:5] = first_point_info[2:4]
            first_point_info_w_alt[5] = first_point_own_cpa_alt
            first_point_info_w_alt[6:8] = first_point_info[4:6]
            first_point_info_w_alt[8] = first_point_n_cpa_alt
            first_point_info_w_alt[9:11] = first_point_info[6:8]
            first_point_info_w_alt[11] = velocity1[2]
            first_point_info_w_alt[12:14] = first_point_info[8:10]
            first_point_info_w_alt[14] = velocity2[2]

            last_point_info_w_alt = np.zeros((9))
            last_point_info_w_alt[:2] = last_point_info[:2]
            last_point_info_w_alt[2] = np.abs(last_point_n_cpa_alt-last_point_own_cpa_alt)
            last_point_info_w_alt[3:5] = last_point_info[2:4]
            last_point_info_w_alt[5] = last_point_own_cpa_alt
            last_point_info_w_alt[6:8] = last_point_info[4:6]
            last_point_info_w_alt[8] = last_point_n_cpa_alt

            first_point_info = first_point_info_w_alt
            last_point_info = last_point_info_w_alt

        own_neighbour_c_location_np = np.array([[fKey1, ownship_location_cpa[0], ownship_location_cpa[1], own_cpa_alt],
                                                [fKey2, neighbour_location_cpa[0], neighbour_location_cpa[1],
                                                 n_cpa_alt]])

        cpa_info = np.concatenate((np.array([tcpa,
                                             dcpa,
                                             np.abs(p1[2]-p2[2])]),
                                   ownship_location_cpa, np.array([own_cpa_alt]),
                                   neighbour_location_cpa, np.array([n_cpa_alt])
                                   ), axis=0)

        return h_conflict and conflict, tcpa, dcpa, np.abs(p1[2]-p2[2]),\
               own_neighbour_c_location_np[0, 1:], own_neighbour_c_location_np[1, 1:], \
               crossing_point, t_at_crossing_point, d_h_cp, d_v_cp, segment_velocity1,\
               segment_velocity2, v_sep_minimum, considered, min(time_threshold, proj_time_horizon),\
               enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
               first_point_info, last_point_info, cpa_info

    elif velocity1[2] != 0 and velocity2[2] != 0:
        conflict = True
        time_to_final_alt1, final_alt1 = utils.time_to_nextfl(p1[2], velocity1[2])
        time_to_final_alt2, final_alt2 = utils.time_to_nextfl(p2[2], velocity2[2])

        if final_alt1 >= 41000 or final_alt2 >= 41000:
            future_v_sep_minimum = 2000

        time_threshold = min([tw, time_to_final_alt1, time_to_final_alt2])

        if (abs(final_alt1-p2[2]) >= max(future_v_sep_minimum, current_threshold) and
            abs(final_alt2-p1[2]) >= max(future_v_sep_minimum, current_threshold) and
            abs(final_alt1-final_alt2) >= max(future_v_sep_minimum, current_threshold) and
            abs(p1[2]-p2[2]) >= max(future_v_sep_minimum, current_threshold)) or time_threshold <= 0.1:
            conflict = False

            considered = False
            v_sep_minimum = current_threshold
            return conflict, np.nan, np.nan, np.abs(p1[2] - p2[2]), nan_arr, nan_arr, None,\
                   np.nan, np.nan, np.nan, nan_arr, nan_arr, v_sep_minimum, considered, None, \
                   None, None, None, None, None

        conflict_flag, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa,\
        crossing_point, t_to_crossing_point, d_h_cp, \
        d_v_cp, segment_velocity_1, segment_velocity_2, v_sep_minimum, proj_time_horizon, \
        enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
        first_conflict_point_info, last_conflict_point_info, cpa_info = \
        compute_cpa_w_fplans_w_v_speed(fKey1, projection_segments1, current_fplan_x_y1, current_fplan_id1, p1, velocity1,
                                       fKey2, projection_segments2, current_fplan_x_y2, current_fplan_id2, p2, velocity2,
                                       timestamp, time_threshold, horizontal_sep_minimum, first_conflict_point)

    elif (velocity1[2] != 0 and velocity2[2] == 0) or (velocity1[2] == 0 and velocity2[2] != 0):
        conflict = True
        if velocity1[2] != 0:
            time_threshold, final_alt1 = utils.time_to_nextfl(p1[2], velocity1[2])
            final_alt2 = p2[2]
        elif velocity2[2] != 0:
            final_alt1 = p1[2]
            time_threshold, final_alt2 = utils.time_to_nextfl(p2[2], velocity2[2])

        time_threshold = min(time_threshold, tw)
        v_sep_minimum = 1000
        if final_alt1 > 41000 or final_alt2 > 41000 or p1[2] > 41000 or p2[2] > 41000 or \
                (velocity2[2] == 0 and p2[2] >= 41000) or (velocity1[2] == 0 and p1[2] >= 41000):
            v_sep_minimum = 2000
        if (abs(final_alt1-final_alt2) >= v_sep_minimum and abs(p1[2]-p2[2]) >= v_sep_minimum) or time_threshold <= 0.1:
            conflict = False
            considered = False
            return conflict, np.nan, np.nan, np.abs(p1[2] - p2[2]), nan_arr, nan_arr, None,\
                   np.nan, np.nan, np.nan, nan_arr, nan_arr, v_sep_minimum, considered, None, \
                   None, None, None, None, None

        conflict_flag, conflict_time, conflict_h_distance, conflict_v_distance, own_position_cpa, neighbour_position_cpa, \
        crossing_point, t_to_crossing_point, d_h_cp, \
        d_v_cp, segment_velocity_1, segment_velocity_2, v_sep_minimum, proj_time_horizon,\
        enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
        first_conflict_point_info, last_conflict_point_info, cpa_info = \
            compute_cpa_w_fplans_w_v_speed(fKey1, projection_segments1, current_fplan_x_y1, current_fplan_id1, p1, velocity1,
                                           fKey2, projection_segments2, current_fplan_x_y2, current_fplan_id2, p2, velocity2,
                                           timestamp, time_threshold, horizontal_sep_minimum, first_conflict_point)

    if not conflict_flag:
        if (final_alt1 == 41000 and time_to_final_alt1 <= time_to_final_alt2) or \
                (final_alt2 == 41000 and time_to_final_alt2 <= time_to_final_alt1) and \
                (current_alt1 < 41000 and current_alt2 < 41000):
            t = min(time_to_final_alt1, time_to_final_alt2)
            if abs(current_alt1 - current_alt2) >= 1000:
                conflict = False
                alt1_410 = current_alt1 + velocity1[2] * t
                alt2_410 = current_alt2 + velocity2[2] * t
                v_dist = ((current_alt1 + velocity1[2] * t) - (current_alt2 + velocity2[2] * t))
                if abs(v_dist) < 2000:
                    ownship_location_t, t_in_seg_1_flag, seg_vel1 = \
                        find_h_position_at_t_from_segments(enriched_segments_w_alt_1, timestamp+t)
                    neighbour_location_t, t_in_seg_2_flag, seg_vel2 = \
                        find_h_position_at_t_from_segments(enriched_segments_w_alt_2, timestamp+t)
                    if t_in_seg_2_flag and t_in_seg_1_flag:
                        dist = np.sqrt(np.sum((ownship_location_t-neighbour_location_t)**2))
                        if dist < horizontal_sep_minimum*1852:
                            conflict_time = t
                            conflict_h_distance = dist
                            segment_velocity1 = seg_vel1
                            segment_velocity2 = seg_vel2
                            first_conflict_point_info = \
                                np.concatenate([np.array([conflict_time, conflict_h_distance, abs(v_dist)]),
                                                ownship_location_t, np.array([alt1_410]), neighbour_location_t,
                                                np.array([alt2_410]),
                                                segment_velocity1, np.array([velocity1[2]]),
                                                segment_velocity2,
                                                np.array([velocity2[2]])], axis=0)
                            last_conflict_point_info = np.copy(first_conflict_point_info)
                            own_position_cpa = np.concatenate([ownship_location_t, np.array([alt1_410])])
                            neighbour_position_cpa = np.concatenate([neighbour_location_t, np.array([alt2_410])])
                            conflict_flag = True
                            v_sep_minimum = 2000

    own_neighbour_c_location_np = np.array([[fKey1, own_position_cpa[0], own_position_cpa[1], own_position_cpa[2]],
                                            [fKey2, neighbour_position_cpa[0], neighbour_position_cpa[1],
                                             neighbour_position_cpa[2]]])

    return conflict_flag, conflict_time, conflict_h_distance, conflict_v_distance, own_neighbour_c_location_np[0, 1:],\
           own_neighbour_c_location_np[1, 1:],\
           crossing_point, t_to_crossing_point, d_h_cp, d_v_cp, segment_velocity_1, segment_velocity_2, v_sep_minimum,\
           considered, min(time_threshold, proj_time_horizon), \
           enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
           first_conflict_point_info, last_conflict_point_info, cpa_info

def crossing_point_w_fplans(enriched_segments1, enriched_segments2):

    crossing_point = None
    t_to_crossing_point = None
    d_h_cp = None
    crosses = False
    int_seg_i_j = [enriched_segments1.shape[0], enriched_segments2.shape[0]]
    int_segment_flag = False
    for i, enriched_segment1 in enumerate(enriched_segments1):
        for j, enriched_segment2 in enumerate(enriched_segments2):
            l_passing_s1_interesects_s2, intersect, intersection_p = \
                geometry_utils.line_segments_intersect2(enriched_segment1[:, :2], enriched_segment2[:, :2])
            if intersect:
                int_segment_flag = True
                if i < int_seg_i_j[0] and j < int_seg_i_j[1]:
                    int_seg_i_j[0] = i
                    int_seg_i_j[1] = j
                    int_segment_1 = enriched_segment1
                    int_segment_2 = enriched_segment2
                    int_p = intersection_p

    if int_segment_flag:
        crosses = True

        a1_speed = np.sqrt(np.sum((int_segment_1[1, :2]-int_segment_1[0, :2])**2, axis=-1)) / \
                   (int_segment_1[1, 2] - int_segment_1[0, 2])
        a2_speed = np.sqrt(np.sum((int_segment_2[1, :2] - int_segment_2[0, :2])**2, axis=-1)) / \
                   (int_segment_2[1, 2] - int_segment_2[0, 2])

        a1_to_crossing_point = np.sqrt(np.sum((int_p[:2]-int_segment_1[0, :2])**2, axis=-1))/a1_speed + int_segment_1[0, 2]
        a2_to_crossing_point = np.sqrt(np.sum((int_p[:2] - int_segment_2[0, :2]) ** 2, axis=-1)) / \
                               a2_speed + int_segment_2[0, 2]
        if a1_to_crossing_point < a2_to_crossing_point:
            p_at_time, t_in_segments, segment_velocity = \
                find_h_position_at_t_from_segments(enriched_segments2, a1_to_crossing_point)
            t_to_crossing_point = a1_to_crossing_point

            if not t_in_segments:
                print('suspicious 2')
                print(enriched_segments2)
                print(a1_to_crossing_point)
                assert False

            d_h_cp = np.sqrt(np.sum((int_p[:2]-p_at_time)**2, axis=-1))

        else:
            p_at_time, t_in_segments, segment_velocity = \
                find_h_position_at_t_from_segments(enriched_segments1, a2_to_crossing_point)
            t_to_crossing_point = a2_to_crossing_point
            if not t_in_segments:
                print('suspicious 2')
                print(enriched_segments1)
                print(a2_to_crossing_point)
                assert False
            d_h_cp = np.sqrt(np.sum((int_p[:2] - p_at_time) ** 2, axis=-1))

        crossing_point = int_p[:2]

    return crosses, crossing_point, t_to_crossing_point, d_h_cp

def crossing_point_w_fplans_no_jit(enriched_segments1, enriched_segments2):
    intersecting_segments = []
    crossing_point = None
    t_to_crossing_point = None
    d_h_cp = None
    crosses = False

    for i, enriched_segment1 in enumerate(enriched_segments1):
        for j, enriched_segment2 in enumerate(enriched_segments2):
            l_passing_s1_interesects_s2, intersect, intersection_p = \
                geometry_utils.line_segments_intersect2(enriched_segment1[:, :2], enriched_segment2[:, :2])
            if intersect:

                intersecting_segments.append([enriched_segment1, enriched_segment2, intersection_p, i, j])

    if len(intersecting_segments) != 0:
        crosses = True
        intersecting_segments_np = np.array(intersecting_segments)
        if len(intersecting_segments) > 1:
            min_i = intersecting_segments_np[intersecting_segments_np[:, 3] == np.min(intersecting_segments_np[:, 3])]
            int_p = min_i[min_i[:, 4] == np.min(min_i[:, 4])][0]

        else:
            int_p = intersecting_segments_np[0]

        a1_speed = np.sqrt(np.sum((int_p[0][1, :2]-int_p[0][0, :2])**2, axis=-1))/(int_p[0][1, 2]-int_p[0][0, 2])
        a2_speed = np.sqrt(np.sum((int_p[1][1, :2] - int_p[1][0, :2])**2, axis=-1)) / (int_p[1][1, 2] - int_p[1][0, 2])

        a1_to_crossing_point = np.sqrt(np.sum((int_p[2][:2]-int_p[0][0, :2])**2, axis=-1))/a1_speed + int_p[0][0, 2]
        a2_to_crossing_point = \
            np.sqrt(np.sum((int_p[2][:2] - int_p[1][0, :2]) ** 2, axis=-1)) / a2_speed + int_p[1][0, 2]
        if a1_to_crossing_point < a2_to_crossing_point:
            p_at_time, t_in_segments, segment_velocity = \
                find_h_position_at_t_from_segments(enriched_segments2, a1_to_crossing_point)
            t_to_crossing_point = a1_to_crossing_point

            if not t_in_segments:
                print('suspicious 2')
                print(enriched_segments2)
                print(a1_to_crossing_point)
                assert False

            d_h_cp = np.sqrt(np.sum((int_p[2][:2]-p_at_time)**2, axis=-1))

        else:
            p_at_time, t_in_segments, segment_velocity = \
                find_h_position_at_t_from_segments(enriched_segments1, a2_to_crossing_point)
            t_to_crossing_point = a2_to_crossing_point
            if not t_in_segments:
                print('suspicious 2')
                print(enriched_segments1)
                print(a2_to_crossing_point)
                assert False
            d_h_cp = np.sqrt(np.sum((int_p[2][:2] - p_at_time) ** 2, axis=-1))

        crossing_point = int_p[2][:2]

    return crosses, crossing_point, t_to_crossing_point, d_h_cp

def search_first_conflict_point(timestamp, time_window, enriched_segments_1, enriched_segments_2,
                                horizontal_sep_minimum, last_flag, tw):

    dt = -5
    if last_flag:
        dt = 5
    time = dt
    first_conflict_time = timestamp
    prev_p1, t_in_segments1, prev_segment_velocity1 = find_h_position_at_t_from_segments(enriched_segments_1,
                                                                                         timestamp)
    prev_p2, t_in_segments2, prev_segment_velocity2 = find_h_position_at_t_from_segments(enriched_segments_2,
                                                                                         timestamp)
    prev_d = np.sqrt(np.sum((prev_p2[:2] - prev_p1[:2]) ** 2, axis=-1))
    while True and time_window+time >= 0 and time_window+time <= tw:

        p1_at_time, t_in_segments1, segment_velocity1 = find_h_position_at_t_from_segments(enriched_segments_1,
                                                                                           time+timestamp)
        if not t_in_segments1:
            print('t not in segments 1')
            print('segments')
            print(enriched_segments_1)
            print('time+timestamp')
            print(time+timestamp)
            print('time: ', time)
            print('dt:', dt)
            print('assert False')
            assert False

        p2_at_time, t_in_segments2, segment_velocity2 = find_h_position_at_t_from_segments(enriched_segments_2,
                                                                                           time+timestamp)
        if not t_in_segments2:
            print('t not in segments 2')
            print('segments')
            print(enriched_segments_2)
            print('time')
            print(time+timestamp)
            print('assert False')
            assert False

        d = np.sqrt(np.sum((p2_at_time[:2]-p1_at_time[:2])**2,axis=-1))
        if d >= horizontal_sep_minimum*1852:
            break

        first_conflict_time = timestamp+time
        prev_d = d
        time += dt
        (prev_p1, prev_segment_velocity1) = (p1_at_time, segment_velocity1)
        (prev_p2, prev_segment_velocity2) = (p2_at_time, segment_velocity2)

    return first_conflict_time, prev_d, prev_p1, prev_p2, prev_segment_velocity1, prev_segment_velocity2


def cut_segments(enriched_segments, timestamp):
    idx = enriched_segments.shape[0]-1
    for i in range(enriched_segments.shape[0]):
        if enriched_segments[i, 1, 2] > timestamp:
            segment_time = enriched_segments[i, 1, 2] - enriched_segments[i, 0, 2]
            segment_h_speed_components = (enriched_segments[i, 1, :2] - enriched_segments[i, 0, :2])/segment_time
            tw = timestamp - enriched_segments[i, 0, 2]
            x_y_end = enriched_segments[i, 0, :2]+segment_h_speed_components*tw
            enriched_segments[i, 1, :2] = x_y_end
            segment_alt_speed = (enriched_segments[i, 1, 3] - enriched_segments[i, 0, 3])/segment_time
            alt_end = enriched_segments[i, 0, 3] + segment_alt_speed*tw
            enriched_segments[i, 1, 3] = alt_end
            enriched_segments[i, 1, 2] = timestamp
            if timestamp == enriched_segments[i, 0, 2]:
                idx = i-1
            else:
                idx = i
            break

    return enriched_segments[:idx+1]

def compute_cpa_w_fplans(fKey1, enriched_segments_w_alt_1, current_fplan_x_y1, current_fplan_id1, p1_init, velocity1_init,
                         fKey2, enriched_segments_w_alt_2, current_fplan_x_y2, current_fplan_id2, p2_init, velocity2_init,
                         timestamp, tw, horizontal_sep_minimum, first_conflict_point):

    p1 = p1_init[:2]
    p2 = p2_init[:2]
    velocity1 = velocity1_init[:2]
    velocity2 = velocity2_init[:2]

    proj_time_horizon = min(enriched_segments_w_alt_1[-1, 1, 2], enriched_segments_w_alt_2[-1, 1, 2])-timestamp

    if enriched_segments_w_alt_1[-1, 1, 2] < enriched_segments_w_alt_2[-1, 1, 2]:
        enriched_segments_w_alt_2 = cut_segments(enriched_segments_w_alt_2, enriched_segments_w_alt_1[-1, 1, 2])
    elif enriched_segments_w_alt_2[-1, 1, 2] < enriched_segments_w_alt_1[-1, 1, 2]:
        enriched_segments_w_alt_1 = cut_segments(enriched_segments_w_alt_1, enriched_segments_w_alt_2[-1, 1, 2])

    enriched_segments_1 = enriched_segments_w_alt_1[:, :, :3]
    enriched_segments_2 = enriched_segments_w_alt_2[:, :, :3]

    crosses, crossing_point, t_at_crossing_point, d_h_cp = crossing_point_w_fplans(enriched_segments_1,
                                                                                   enriched_segments_2)

    t_to_crossing_point = None
    if crosses:
        t_to_crossing_point = t_at_crossing_point-timestamp

    times = np.zeros((enriched_segments_1.shape[0]*enriched_segments_2.shape[0]))
    dcpas = np.zeros((enriched_segments_1.shape[0]*enriched_segments_2.shape[0]))
    idxs = np.full((enriched_segments_1.shape[0]*enriched_segments_2.shape[0]), -1)
    own_locations = np.zeros((enriched_segments_1.shape[0]*enriched_segments_2.shape[0], 2))
    neighbour_locations = np.zeros((enriched_segments_1.shape[0]*enriched_segments_2.shape[0], 2))
    segment_velocities1 = np.zeros((enriched_segments_1.shape[0] * enriched_segments_2.shape[0], 2))
    segment_velocities2 = np.zeros((enriched_segments_1.shape[0] * enriched_segments_2.shape[0], 2))

    idx = 0
    for idx1, enriched_segment_1 in enumerate(enriched_segments_1):
        for idx2, enriched_segment_2 in enumerate(enriched_segments_2):
            segment_velocity1 = (enriched_segment_1[1, :2] - enriched_segment_1[0, :2]) / \
                                np.abs((enriched_segment_1[1][2] - enriched_segment_1[0][2]))
            segment_velocity2 = (enriched_segment_2[1, :2] - enriched_segment_2[0, :2]) / \
                                np.abs((enriched_segment_2[1][2] - enriched_segment_2[0][2]))

            if enriched_segment_1[1][2] < enriched_segment_2[0][2] or\
                    enriched_segment_2[1][2] < enriched_segment_1[0][2]:

                continue
            if enriched_segment_1[0][2] < enriched_segment_2[0][2]:
                p2 = enriched_segment_2[0, :2]
                p1 = enriched_segment_1[0, :2] + segment_velocity1*np.abs(
                    enriched_segment_2[0][2]-enriched_segment_1[0][2])
                time = enriched_segment_2[0][2]
            else:
                p1 = enriched_segment_1[0, :2]
                p2 = enriched_segment_2[0, :2] + segment_velocity2 * np.abs(
                    enriched_segment_1[0][2] - enriched_segment_2[0][2])
                time = enriched_segment_1[0][2]

            tcpa = utils.compute_tcpa(p1, p2, segment_velocity1, segment_velocity2)

            if tcpa >= 0:
                min_time_bound = min(enriched_segment_1[1][2], enriched_segment_2[1][2])

                if time+tcpa > min_time_bound:

                    tcpa = min_time_bound-time

            else:
                tcpa = 0

            dcpa, d, ownship_location, neighbour_location = utils.compute_dcpa(p1, p2, segment_velocity1,
                                                                               segment_velocity2, tcpa)

            flat_id = idx1 * enriched_segments_2.shape[0] + idx2
            times[flat_id] = time + tcpa-timestamp
            dcpas[flat_id] = d
            idxs[flat_id] = idx
            own_locations[flat_id, :] = ownship_location
            neighbour_locations[flat_id, :] = neighbour_location
            segment_velocities1[flat_id, :] = segment_velocity1
            segment_velocities2[flat_id, :] = segment_velocity2

            idx += 1

    mask = (idxs != -1)
    dcpas = dcpas[mask]
    times = times[mask]
    idxs = idxs[mask]
    own_locations = own_locations[mask]
    neighbour_locations = neighbour_locations[mask]
    segment_velocities1 = segment_velocities1[mask]
    segment_velocities2 = segment_velocities2[mask]

    h_conflicts_idx = np.where(dcpas < horizontal_sep_minimum*1852)[0]

    h_conflict = False
    if h_conflicts_idx.shape[0] > 0:
        time_idx = np.zeros((times.shape[0], 2))
        time_idx[:, 0] = times
        time_idx[:, 1] = idxs

        h_conflicts_time_idxs = time_idx[h_conflicts_idx]
        cpa_idx = np.argmin(h_conflicts_time_idxs[:, 0])
        cpa_idx = int(h_conflicts_time_idxs[cpa_idx][1])

        h_conflict = True
    else:
        cpa_idx = np.argmin(dcpas)

    tcpa = times[cpa_idx]
    dcpa = dcpas[cpa_idx]

    first_point_info = None
    last_point_info = None
    if h_conflict and first_conflict_point:
        first_tcpa, first_dcpa, first_ownship_location_cpa, first_neighbour_location_cpa, first_segment_velocity1,\
        first_segment_velocity2 = \
            search_first_conflict_point(timestamp+tcpa, tcpa, enriched_segments_1, enriched_segments_2,
                                        horizontal_sep_minimum, False, proj_time_horizon)
        first_point_info = np.concatenate((np.array([first_tcpa-timestamp, first_dcpa]), first_ownship_location_cpa,
                                           first_neighbour_location_cpa, first_segment_velocity1,
                                           first_segment_velocity2), axis=0)

        last_tcpa, last_dcpa, last_ownship_location_cpa, last_neighbour_location_cpa, last_segment_velocity1,\
        last_segment_velocity2 = \
            search_first_conflict_point(timestamp + tcpa, tcpa, enriched_segments_1, enriched_segments_2,
                                        horizontal_sep_minimum, True, proj_time_horizon)
        last_point_info = np.concatenate(
            (np.array([last_tcpa-timestamp, last_dcpa]), last_ownship_location_cpa, last_neighbour_location_cpa), axis=0)

    ownship_location_cpa = own_locations[cpa_idx]
    neighbour_location_cpa = neighbour_locations[cpa_idx]
    segment_velocity1 = segment_velocities1[cpa_idx]
    segment_velocity2 = segment_velocities2[cpa_idx]

    return tcpa, dcpa, ownship_location_cpa, neighbour_location_cpa, segment_velocity1, segment_velocity2,\
           h_conflict, crossing_point, t_to_crossing_point, d_h_cp, crosses, proj_time_horizon, \
           enriched_segments_w_alt_1, enriched_segments_w_alt_2, \
           first_point_info, last_point_info

def enrich_segments_w_time(segments_np, timestamp, speed_magnitude, alt, alt_speed, tw):

    times = []
    time = timestamp
    alts = []
    for i, line_segment in enumerate(segments_np):
        times.append(time)
        alts.append(alt+(time-timestamp)*alt_speed)
        time0 = time
        distance = np.sqrt(np.sum((line_segment[1]-line_segment[0])**2, axis=-1))
        segment_time = distance/speed_magnitude
        time1 = time0 + segment_time
        if time1 - timestamp > tw:
            segment_speed = (line_segment[1] - line_segment[0]) / segment_time
            segments_np[i, 1] = line_segment[0]+segment_speed*(tw-(time0-timestamp))
            times.append(timestamp+tw)
            alts.append(alt + tw * alt_speed)

            break
        else:
            times.append(time1)
            alts.append(alt + (time1-timestamp) * alt_speed)

        time += segment_time

    if times[-1] == times[-2]:
        times = times[:-2]
        alts = alts[:-2]

    if len(times) == 0:
        return np.empty((0,0,4))

    if times[0] == times[1]:
        times = times[2:]
        alts = alts[2:]
        segments_np = segments_np[1:]

    for i in range(len(times)):
        if i % 2 == 1 and times[i] == times[i-1]:

            times = times[:i-1] + times[i+1:]
            alts = alts[:i-1] + alts[i+1:]
            segments_np = np.delete(segments_np, int(i/2), axis=0)
            segments_np[int(i/2)-1, 1, :2] = segments_np[int(i/2), 0, :2]

    times_np = np.array(times)
    times_np = times_np.reshape((int(times_np.shape[0]/2), 2, 1))
    alts_np = np.array(alts)
    alts_np = alts_np.reshape((int(alts_np.shape[0]/2), 2, 1))

    enriched_segments_np = np.concatenate((segments_np[:times_np.shape[0]], times_np, alts_np), axis=-1)

    return enriched_segments_np

def isclose(arr1, arr2, rtol=1e-05, atol=1e-08):

    return np.abs(arr1 - arr2) <= (atol + rtol * np.abs(arr2))

def point_close_to_segment(flightKey, flight_plan_id, flight_plan_x_y, p, alt, velocity, alt_speed, timestamp, tw,
                           projectionID):

    fplan_used_flag = False
    fplan_used = None
    intersection_point = None
    int_p_alt_aircraft = None
    flight_plan_course = None
    flight_plan_alt = None
    flight_plan_h_speed = None
    flight_plan_v_speed = None
    threshold = 2000
    course_threshold = 20
    enriched_segments_np = np.empty((0, 2, 4))

    course, _ = utils.mybearing_speed_components(velocity[0], velocity[1])
    p0 = p[:2]
    segment_idx = None
    course_line_segment = np.zeros((1, 2, 2)).astype(np.float64)
    course_line_segment[0, 0, :] = p0
    course_line_segment[0, 1, :] = p0 + tw * velocity
    course_fplan_line_segments = course_line_segment
    speed = np.sqrt(np.sum(velocity ** 2))

    line_segments_np = geometry_utils.poly_line_segments(flight_plan_x_y, dim2=4, convex=False)

    for idx, line_segment in enumerate(line_segments_np):
        l2 = np.sum((line_segment[1,:2]-line_segment[0,:2])**2, axis=-1)
        if l2 == 0:
            closest_segment_point = line_segment[0, :2]
        else:
            t = max(0, np.dot(p0 - line_segment[0, :2], line_segment[1, :2] - line_segment[0, :2]) / l2)
            if t > 1:
                continue
            closest_segment_point = line_segment[0, :2] + t*(line_segment[1, :2]-line_segment[0, :2])

        distance = np.sqrt(np.sum((p0-closest_segment_point)**2, axis=-1))

        if distance <= threshold:
            segment_course, _ = utils.mybearing(line_segment[0, :2], line_segment[1, :2])
            abs_course = abs(course - segment_course)
            if abs_course > 180:
                abs_course = 360 - abs_course

            if abs_course < course_threshold:
                time_to_next_wp = np.sqrt(np.sum((line_segment[1, :2]-p0)**2, axis=-1))/speed
                if time_to_next_wp > 0:
                    if idx < line_segments_np.shape[0] - 1:
                        course_fplan_line_segments = \
                            np.concatenate([[[p0, line_segment[1, :2]]], line_segments_np[idx+1:, :, :2]], axis=0)
                    else:
                        course_fplan_line_segments = np.array([[p0, line_segment[1, :2]]])
                    segment_idx = idx
                elif idx < line_segments_np.shape[0] - 1:
                    if idx < line_segments_np.shape[0] - 2:
                        course_fplan_line_segments = \
                            np.concatenate([[[p0, line_segments_np[idx+1, 1, :2]]], line_segments_np[idx + 2:, :, :2]],
                                           axis=0)
                    else:
                        course_fplan_line_segments = np.array([[p0, line_segments_np[idx+1, 1, :2]]])

                    segment_idx = idx+1

                break

    if segment_idx is not None:
        line_segment = line_segments_np[segment_idx]
        fplan_used_flag = True
        fplan_used = flight_plan_id
        intersection_point = line_segments_np[segment_idx, 1, :2]
        int_p_alt_aircraft = alt+alt_speed*(np.sqrt(np.sum((intersection_point-p0)**2, axis=-1))/speed)
        flight_plan_course, _ = utils.mybearing(line_segment[0, :2], line_segment[1, :2])
        flight_plan_alt = line_segments_np[segment_idx, 1, 2]
        segment_h_distance = \
            np.sqrt(np.sum((line_segments_np[segment_idx, 1, :2] - line_segments_np[segment_idx, 0, :2])**2, axis=-1))
        segment_time = (line_segments_np[segment_idx, 1, 3] - line_segments_np[segment_idx, 0, 3])
        if segment_time == 0:
            segment_time = 30
        flight_plan_h_speed = segment_h_distance/segment_time
        flight_plan_v_speed = np.abs(line_segments_np[segment_idx, 1, 2] - line_segments_np[segment_idx, 0, 2])/segment_time

    enriched_segments_np = enrich_segments_w_time(course_fplan_line_segments, timestamp, speed,
                                                  alt, alt_speed, tw)

    return enriched_segments_np, fplan_used_flag, fplan_used, intersection_point, int_p_alt_aircraft, flight_plan_course,\
           flight_plan_alt, flight_plan_h_speed, flight_plan_v_speed

def course_fplan_intersection(flightKey, flight_plan_id, flight_plan_x_y, p0, alt, velocity, alt_speed, timestamp, tw ):
    """
    :param flight_plans: flight plan structure
    :param flightKey: flight key of the flight
    :param p0: current point of aircraft
    :param velocity: the velocity of the aircraft
    :return: [flight_plan_segment_idx, distance from intersection_point, intersection point], closest intersecting
     segment
    """

    fplan_used_flag = False
    fplan_used = None
    intersection_point = None
    int_p_alt_aircraft = None
    flight_plan_course = None
    flight_plan_alt = None
    flight_plan_h_speed = None
    flight_plan_v_speed = None
    if flight_plan_x_y.shape[0] == 0:
        course_line_segment = np.zeros((1, 2, 2)).astype(np.float64)
        course_line_segment[0, 0, :] = p0
        course_line_segment[0, 1, :] = p0 + tw * velocity
        course_fplan_line_segments_np = course_line_segment
        speed_magnitude = np.sqrt(np.sum(velocity ** 2))
    else:
        course_line_segment = np.zeros((2, 2)).astype(np.float64)
        course_line_segment[0, :] = p0
        course_line_segment[1, :] = p0 + tw * velocity

        fplan_used = flight_plan_id

        line_segments = geometry_utils.poly_line_segments(flight_plan_x_y, dim2=4, convex=False)

        intersecting_segments = []
        line_segments_np = line_segments
        speed_magnitude = np.sqrt(np.sum(velocity ** 2))
        intersect_flag = False
        for idx, line_segment in enumerate(line_segments_np):
            segment_dist = np.sqrt(np.sum((line_segment[1]-line_segment[0])**2))
            segment_time = segment_dist/speed_magnitude

            l_passing_s1_interesects_s2, intersect, intersection_p = \
                geometry_utils.line_segments_intersect2(course_line_segment, line_segment)

            if intersect:
                dist = np.sqrt(np.sum((intersection_p-p0)**2, axis=-1))
                intersecting_segments.append([idx, dist, intersection_p[0], intersection_p[1],
                                              line_segment[0][0], line_segment[0][1], line_segment[0][2], line_segment[0][3],
                                              line_segment[1][0], line_segment[1][1], line_segment[1][2], line_segment[1][3]])
        if len(intersecting_segments) > 0:
            intersecting_segments_np = np.array(intersecting_segments)
            intersection_idx = np.argmin(intersecting_segments_np[:, 1])
            intersection_elem = intersecting_segments_np[intersection_idx]
            intersection_point = intersection_elem[2:4]
            time_to_int_p = intersection_elem[1]/speed_magnitude
            int_p_alt_aircraft = alt+alt_speed*time_to_int_p
            flight_plan_course, _ = utils.mybearing(intersection_elem[4:6], intersection_elem[8:10])

            fp_segment_td = intersection_elem[11]-intersection_elem[7]
            if fp_segment_td == 0:
                fp_segment_td = 30
            fp_segment_dist = np.sqrt(np.sum((intersection_elem[8:10]-intersection_elem[4:6])**2))
            flight_plan_h_speed = fp_segment_dist/fp_segment_td
            flight_plan_v_speed = (intersection_elem[10]-intersection_elem[6])/fp_segment_td
            fp_time_to_int_p = np.sqrt(np.sum((intersection_elem[2:4]-intersection_elem[4:6])**2))/flight_plan_h_speed
            flight_plan_alt = intersection_elem[6] + fp_time_to_int_p * flight_plan_v_speed

            if np.all(isclose(intersection_elem[2:4], line_segments_np[int(intersection_elem[0]), 1, :2], rtol=1e-08)):
                course_fplan_line_segments_np = np.array([p0[0], p0[1], line_segments_np[int(intersection_elem[0]), 1, 0],
                                                          line_segments_np[int(intersection_elem[0]), 1, 1]])
                course_fplan_line_segments_np = np.reshape(course_fplan_line_segments_np, (1, 2, 2))

            else:
                course_fplan_line_segments_np = np.array([p0[0], p0[1], intersection_elem[2], intersection_elem[3],
                                                          intersection_elem[2], intersection_elem[3],
                                                          line_segments_np[int(intersection_elem[0]), 1, 0],
                                                          line_segments_np[int(intersection_elem[0]), 1, 1]])
                course_fplan_line_segments_np = np.reshape(course_fplan_line_segments_np, (2, 2, 2))

            course_fplan_line_segments_np = \
                np.concatenate((course_fplan_line_segments_np, line_segments[int(intersection_elem[0])+1:, :, :2]),
                               axis=0)

            fplan_used_flag = True

        else:
            course_fplan_line_segments_np = np.reshape(course_line_segment, (1, 2, 2))

    enriched_segments_np = enrich_segments_w_time(course_fplan_line_segments_np, timestamp, speed_magnitude, alt,
                                                  alt_speed, tw)

    return enriched_segments_np, fplan_used_flag, fplan_used, intersection_point, int_p_alt_aircraft, flight_plan_course,\
           flight_plan_alt, flight_plan_h_speed, flight_plan_v_speed

def find_current(fplans, timestamp):
    for i, (tstamp, plan, fpkey_time, p_start, p_end, phase) in enumerate(fplans):
        if timestamp >= tstamp:
            return i


def reset_current(fplans, timestamp):
    for fkey in fplans:
        fplans[fkey]['current'] = find_current(fplans[fkey]['fplans'], timestamp)


def read_flight_plans(flight_df, flight_plans):
    flight_key = int(flight_df['flightKey'].iloc[0])
    timestamp = flight_df['timestamp'].iloc[0]
    dt_object = datetime.utcfromtimestamp(timestamp)
    yearmonth = dt_object.strftime("%Y%m")

    fplans = read_flight_plans_of_month(flight_key, yearmonth)

    current_fplan_idx = find_current(fplans, timestamp)

    flight_plans[flight_key] = {}
    flight_plans[flight_key]['current'] = current_fplan_idx
    flight_plans[flight_key]['fplans'] = fplans
    print_current_fplan(flight_key, flight_plans)

    return flight_plans


def print_current_fplan(flightKey, flight_plans):
    debug_folder = fplan_utils_global['debug_folder']

    if len(flight_plans[int(flightKey)]['fplans']) == 0:
        return

    fplan_df = flight_plans[int(flightKey)]['fplans'][
        flight_plans[int(flightKey)]['current']][1]

    Path(debug_folder + '/fplans/').mkdir(parents=True, exist_ok=True)
    fplan_df.to_csv(debug_folder + '/fplans/' + str(flightKey) +
                    '_current_fp.csv', index=False, sep='\t')

def compute_rank(vector_a, vector_b):
    return np.dot(vector_a, vector_b)/np.sqrt(np.sum(vector_b**2))

def read_flight_plans_of_month(flightKey, month):
    files = []
    fplan_path = fplan_utils_global['fplan_path']

    for idx, file in enumerate(glob.glob(fplan_path + month + '/' + str(flightKey) + '/*.fp')):
        fpkey_tstamp = file.split('/')[-1].split('.')[0].split('_')[1:]

        fplan_df = pd.read_csv(file, usecols=['lat', 'lon', 'time', 'alt', 'sectorID', 'wayPointOrder'], sep=';').\
            rename(columns={'lat': 'latitude', 'lon': 'longitude', 'alt': 'altitude', 'time': 'timestamp'})

        fplan_df = fplan_df.sort_values(['wayPointOrder']).drop_duplicates().drop(['wayPointOrder'], axis=1)

        fplan_df['timestamp'] = pd.to_datetime(fplan_df['timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        fplan_df['timestamp'] = fplan_df['timestamp'].values.astype(np.int64) // 10 ** 9

        fplan_df = flight_utils.transform_coordinates(fplan_df)
        p_start = fplan_df[['x', 'y']].iloc[0].values
        p_end = fplan_df[['x', 'y']].iloc[-1].values

        rank = np.apply_along_axis(compute_rank, 1, fplan_df[['x', 'y']].values-p_start, p_end-p_start)
        fplan_df['rank'] = rank
        f_phase_df = pd.read_csv(file[:-3] + '.phase', sep=';')
        files.append([int(fpkey_tstamp[1]), fplan_df, '_'.join(fpkey_tstamp), p_start, p_end, f_phase_df])

        if idx == 0:
            fpkey = fpkey_tstamp[0]
        else:
            assert fpkey_tstamp[0] == fpkey

    files = sorted(files, key=lambda l: l[0], reverse=True)

    return files
