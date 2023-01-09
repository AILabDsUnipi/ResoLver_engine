"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pandas as pd
from datetime import datetime, timezone
from math import ceil, floor
import time
import numpy as np
import faulthandler; faulthandler.enable()
import glob
import pickle
import os
from pathlib import Path
import copy
from math import sin, cos, radians, degrees

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from env_config import env_config
import flight_utils as flight_utils
import utils
import flight_plan_utils
import sector_utils

class Environment(object):
    """
    self.flight_plans: dictionary {flightKey: { current index: int,
                                                fplans: list of fplans in *reverse* order according to timestamp}}
    self.flight_index: dict {flightKey: { 'idx', df with first_point, df with last point}}
                       'idx' is the index of the specific flight in the self.flight_arr array
    self.flight_entry_index: dict {timestamp: set(flightKeys of flights activated at timestamp)}
    self.active_flights_mask: boolean np array reporting active flights
    self.activated_mask: boolean np array reporting if flight has been activated at a current timestamp or in the past
    self.flight_arr: np array containing all flights (active and innactive) of the scenario
                     features of self.flight_arr : ['flightKey', 'longitude', 'latitude',
                                                    'x', 'y', 'altitude',
                                                    'speed_x_component', 'speed_y_component',
                                                    'course', 'speed', 'alt_speed','bearing_relative_to_exit_wp',
                                                    'vertical_distance_to_exit_point','horizontal_distance_to_exit_point']
    self.edges: [flights_id_i, flights_id_j, tcpa, dcpa, intersection_angle,
                 bearing_cpa_ownship_neighbour_relative, v_d_cpa,
                 distance at crossing point, time_to_crossing_point,
                 horizontal_distance, vertical_distance, considered_flag]
    self.exit_wps: [ewp_f1, ewp_f2 ... ewp_fn] where ewp_fi is the exit point of flight i w.r.t. the order
                   flights are recorded in self.flight_arr
                   ewp: [lat lon timestamp alt Sector x y flightKey]
    """

    def __init__(self, scenario='A3_1675139981_LECBVNI_AAF765'):
        self.step_num = 0
        self.sector_data = sector_utils.SectorData()
        self.debug_list = []
        self.total_losses_of_separation = 0
        self.total_alerts = 0
        self.total_conflicts = 0
        self.total_losses_of_separation_per_episode = []
        self.total_alerts_per_episode = []
        self.log_states = []
        self.episode_num = 0
        self.alerts_debug = 0
        Path(env_config['log_dir']).mkdir(parents=True, exist_ok=True)
        self.scenario = scenario+'_interpolated.rdr'
        self.debug_folder = self.scenario+'_debug_files' if os.path.exists('./../environment') \
                                                         else './env/environment/' + self.scenario+'_debug_files'

        Path(self.debug_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists('./env/training_data'):
            self.training_path = './env/training_data'
        elif os.path.exists('./../training_data'):
            self.training_path = './../training_data'
        else:
            raise ValueError('None of the options for the path of the training data ("./env/training_data" or "../training_data") is valid!!')

        self.fplan_path = './env/fplans/' if os.path.exists('./env/fplans') else './../fplans/'
        flight_plan_utils.fplan_utils_global['fplan_path'] = self.fplan_path
        flight_plan_utils.fplan_utils_global['debug_folder'] = self.debug_folder
        self.dt = 30
        self.flight_index = {}
        # self.flight_entry_index: dict {timestamp: set(flightKeys of flights activated at timestamp)}
        self.flight_entry_index = {}
        self.flight_plans = {}
        self.exit_points_dict = {}
        # self.active_flights_mask: boolean np array reporting active flights
        self.active_flights_mask = None
        self.active_historical_mask = None
        self.new_active_flights_mask = None
        self.flight_plan_updated_mask = None
        self.just_deactivated_mask = None
        # self.activated_mask: boolean np array reporting if flight has been activated at a current timestamp or in the past
        self.activated_mask = None
        # self.flight_arr: np array containing all flights (active and innactive of the scenario
        self.flight_arr = None
        self.init_timestamp = None
        self.current_sector = None
        self.downstream_sector = None
        self.flight_phases = []
        # self.exit_wps: [wp_f1, wp_f2 ... wp_fn] where wp_fi is the exit point of flight i w.r.t. the order
        #                flights are recorded in self.flight_arr. ewp: [lat lon timestamp alt Sector x y flightKey]
        self.exit_wps = None
        self.next_wps = None
        self.available_wps = None
        self.dcourse_to_next_wp = None
        self.wp_per_tstamp = None
        self.direct_to = None
        self.wp_rank = None
        self.towards_wp_rank = None
        self.point_rank = None
        self.next_FL = None
        self.change_FL_vspeed = None
        self.change_FL_hspeed = None
        self.sectorIDs = []

        self.read_flights_training_set()
        self.init_flight_plans = copy.deepcopy(self.flight_plans)

        self.features = ['flightKey', 'longitude', 'latitude',
                         'x', 'y', 'altitude',
                         'speed_x_component', 'speed_y_component',
                         'course', 'speed', 'alt_speed']

        self.flight_sectors = None
        self.feature_size = len(self.features)
        self.flight_num = len(self.flight_index)
        self.main_info_list = []
        self.conflict_info_list = []
        self.conflict_params_list = []
        self.projection_list = []
        self.projection_dict = {}

        if not os.path.exists(env_config['dependency_files_directory']):
            raise ValueError('The selected path for the dependency files ("{}") does not exist!!!'.
                             format(env_config['dependency_files_directory']))

        with open(env_config['dependency_files_directory']+'/bins_median_dict.pickle', 'rb') as handle:
            bins_median_dict = pickle.load(handle)

        d_angles_lst = bins_median_dict['d_angles_transformed'].tolist()
        d_speeds_lst = bins_median_dict['d_h_speeds_transformed'].tolist()
        try:
            d_angles_lst.remove(0.)
        except ValueError as ve:
            pass

        try:
            d_speeds_lst.remove(0.)
        except ValueError as ve:
            pass

        # d_angles 0 and d_speeds 0 must always be first in the list
        utils.utils_global['d_angles'] = np.array([0.] + d_angles_lst)
        utils.utils_global['d_speeds'] = np.array([0.] + d_speeds_lst)

        if not env_config['simulate_uncertainty']:
            utils.utils_global['d_speeds'] = np.array([0.])
            utils.utils_global['d_angles'] = np.array([0.])

        self.d_angles = utils.utils_global['d_angles']
        self.d_speeds = utils.utils_global['d_speeds']
        self.initialize()
        self.platform_states = pd.read_csv(env_config['validation_exercise_input_dir']+'/states_out.csv')
        self.platform_states_dict = dict(tuple(self.platform_states.groupby('timestamp')))

        self.init_output_files()

    def init_output_files(self):
        with open(env_config['log_dir']+'/non_conformance_events.csv', 'w') as f:
            f.write('ID,ResolutionActionID,NonConformaceType,DesiredValue,ActualValue\n')

        with open(env_config['log_dir'] + '/dcourse.csv', 'w') as f:
            f.write('RTkey,p0x,p0y,p1x,p1y,p2x,p2y,timestamp,course,new_course,x_speed,y_speed,new_x_speed,new_y_speed,b_before,b_after,duration\n')

    def impact_conflicts(self, RTkey, res_action_id):
        active_flights_idxs = np.where((self.active_flights_mask) == True)[0]
        active_flights = self.flight_arr[active_flights_idxs]

        # edges: [0: RTkey, 1: x, 2: y, 3: x_speed, 4: y_speed, 5: altitudeInFeets, 6: alt_speed, 7: max_altitude]

        main_info, conflict_info, conflict_params,\
        projections, edges = utils.compute_conflicts(active_flights[:, np.array([0, 3, 4, 6, 7, 5, 10])],
                                                     self.flight_plans, self.timestamp,
                                                     [self.current_sector, self.downstream_sector],
                                                     self.sector_data, None, self.flight_index, RTkey,
                                                     res_action_id, self.projection_dict,
                                                     self.exit_wps, self.action_duration, self.action_starting_time_point,
                                                     self.executing_course_change, self.finished_course_change_mask,
                                                     self.executing_direct_to, self.towards_wp_idxs,
                                                     self.towards_wp_fp_idx, True)

        if len(main_info) > 0:
            main_info_np = np.array(main_info)

            main_info_np[:, 13:15] = flight_utils.transform_coordinates_np(main_info_np[:, 13:15], inverse=True)
            main_info_np[:, 16:18] = flight_utils.transform_coordinates_np(main_info_np[:, 16:18], inverse=True)
            self.main_info_list.extend(main_info_np.tolist())

        if len(conflict_info) > 0:
            conflict_info_np = np.array(conflict_info)
            conflict_info_np[:, 2:4] = flight_utils.transform_coordinates_np(conflict_info_np[:, 2:4], inverse=True)
            mask = conflict_info_np[:, 8].astype(np.float64) != np.nan
            if conflict_info_np[mask].shape[0] > 0:
                conflict_info_np[mask, 8:10] = \
                    flight_utils.transform_coordinates_np(conflict_info_np[mask, 8:10].astype(np.float64), inverse=True)
                conflict_info_np[mask, 14:16] = \
                    flight_utils.transform_coordinates_np(conflict_info_np[mask, 14:16].astype(np.float64), inverse=True)

            self.conflict_info_list.extend(conflict_info_np.tolist())
            self.conflict_params_list.extend(conflict_params)

        if len(projections) > 0:
            self.projection_list.extend(projections)

        return edges

    def edges_between_active_flights(self, res_act_ID):

        active_flights_idxs = np.where((self.active_flights_mask) == True)[0]
        active_flights = self.flight_arr[active_flights_idxs]

        # edges: [0: RTkey, 1: x, 2: y, 3: x_speed, 4: y_speed, 5: altitudeInFeets, 6: alt_speed, 7: max_altitude]

        main_info, conflict_info,\
        conflict_params, projections, edges = utils.compute_conflicts(active_flights[:, np.array([0, 3, 4, 6, 7, 5, 10])],
                                                                      self.flight_plans, self.timestamp,
                                                                      [self.current_sector, self.downstream_sector],
                                                                      self.sector_data, res_act_ID, self.flight_index,
                                                                      None, None, self.projection_dict,
                                                                      self.exit_wps, self.action_duration,
                                                                      self.action_starting_time_point,
                                                                      self.executing_course_change,
                                                                      self.finished_course_change_mask,
                                                                      self.executing_direct_to,
                                                                      self.towards_wp_idxs,
                                                                      self.towards_wp_fp_idx, False)
        if len(main_info) > 0:
            main_info_np = np.array(main_info)

            main_info_np[:, 13:15] = flight_utils.transform_coordinates_np(main_info_np[:, 13:15], inverse=True)
            main_info_np[:, 16:18] = flight_utils.transform_coordinates_np(main_info_np[:, 16:18], inverse=True)
            self.main_info_list.extend(main_info_np.tolist())

        if len(conflict_info) > 0:
            conflict_info_np = np.array(conflict_info)

            conflict_info_np[:, 2:4] = flight_utils.transform_coordinates_np(conflict_info_np[:, 2:4], inverse=True)
            mask = conflict_info_np[:, 8].astype(np.float64) != np.nan
            if conflict_info_np[mask].shape[0] > 0:
                conflict_info_np[mask, 8:10] = \
                    flight_utils.transform_coordinates_np(conflict_info_np[mask, 8:10].astype(np.float64), inverse=True)
                conflict_info_np[mask, 14:16] = \
                    flight_utils.transform_coordinates_np(conflict_info_np[mask, 14:16].astype(np.float64), inverse=True)
            self.conflict_info_list.extend(conflict_info_np.tolist())
            self.conflict_params_list.extend(conflict_params)

        if len(projections) > 0:
            self.projection_list.extend(projections)

        return edges

    def in_sector(self, sectorID):

        return np.apply_along_axis(self.sector_data.point_in_sector, 1, self.flight_arr[:, 3:6],
                                   timestamp=self.timestamp,
                                   sectorID=sectorID, transformed=True)

    def update_fplans_w_mask(self, mask):
        self.flight_plan_updated_mask.fill(False)
        for flight in self.flight_arr[self.active_flights_mask]:
            flightKey = flight[0]
            idx = self.flight_plans[flightKey]['current']
            if idx == 0:
                continue

            if len(self.flight_plans[flightKey]['fplans']) == 0:
                continue
            if self.timestamp >= self.flight_plans[flightKey]['fplans'][idx - 1][0]:
                self.flight_plans[flightKey]['current'] = idx - 1
                self.flight_plan_updated_mask[self.flight_index[flightKey]['idx']] = True

        return

    def update_fplans(self):
        self.flight_plan_updated_mask.fill(False)
        for flight in self.flight_arr[self.active_flights_mask]:
            flightKey = flight[0]
            idx = self.flight_plans[flightKey]['current']
            if idx == 0:
                continue

            if len(self.flight_plans[flightKey]['fplans']) == 0:
                continue
            if self.timestamp >= self.flight_plans[flightKey]['fplans'][idx-1][0]:
                self.flight_plans[flightKey]['current'] = idx-1
                self.flight_plan_updated_mask[self.flight_index[flightKey]['idx']] = True

        return

    def new_active_flights_ve(self, next_states):
        prev_active_flights_mask = np.copy(self.active_flights_mask)
        self.active_flights_mask.fill(False)
        self.new_active_flights_mask.fill(False)

        for next_state in next_states:

            idx = self.flight_index[next_state[0]]['idx']
            self.flight_arr[idx] = next_state
            self.active_flights_mask[idx] = True
            if prev_active_flights_mask[idx] == False:
                self.new_active_flights_mask[idx] = True
            self.activated_mask[idx] = True
            self.active_historical_mask[idx] = True

        in_current_sector = self.in_sector(self.current_sector)
        in_downstream_sector = self.in_sector(self.downstream_sector)
        in_sector_flights_mask = (in_current_sector[:, 0] | in_downstream_sector[:, 0])
        alt_mask = self.flight_arr[:, 5] <= env_config['max_alt']
        self.flight_sectors[in_current_sector[:, 0]] = "C"
        self.flight_sectors[in_downstream_sector[:, 0]] = "D"
        self.active_flights_mask = \
            self.active_flights_mask & in_sector_flights_mask & alt_mask & self.active_historical_mask
        self.flight_sectors[~self.active_flights_mask] = "N"
        self.just_deactivated_mask = prev_active_flights_mask & ~self.active_flights_mask

    def new_active_flights(self):
        # ASSERT first points are in sectors
        self.new_active_flights_mask.fill(False)
        prev_active_flights_mask = np.copy(self.active_flights_mask)

        try:
            for flightKey in self.flight_entry_index[self.timestamp]:
                dict_entry = self.flight_index[int(flightKey)]
                if self.activated_mask[dict_entry['idx']]:
                    continue

                self.flight_arr[dict_entry['idx'], :] = np.append(dict_entry['df'][self.features].iloc[0].values,
                                                                  [0, 0, 0])
                self.active_flights_mask[dict_entry['idx']] = True
                self.active_historical_mask[dict_entry['idx']] = True
                self.activated_mask[dict_entry['idx']] = True
                self.new_active_flights_mask[dict_entry['idx']] = True

        except KeyError:
            pass

        in_current_sector = self.in_sector(self.current_sector)
        in_downstream_sector = self.in_sector(self.downstream_sector)
        in_sector_flights_mask = (in_current_sector[:, 0] | in_downstream_sector[:, 0])
        alt_mask = self.flight_arr[:, 5] <= env_config['max_alt']
        update_wp_mask = (self.flight_sectors != self.current_sector) & \
                         (self.flight_sectors != 'N') & (in_current_sector[:, 0])
        update_wp_mask = update_wp_mask | ((self.flight_sectors != self.downstream_sector) &
                                           (self.flight_sectors != 'N') & in_downstream_sector[:, 0])
        self.flight_sectors[in_current_sector[:, 0]] = self.current_sector
        self.flight_sectors[in_downstream_sector[:, 0]] = self.downstream_sector
        self.active_flights_mask = \
            self.active_flights_mask & in_sector_flights_mask & alt_mask & self.active_historical_mask
        self.flight_sectors[~self.active_flights_mask | ~in_sector_flights_mask] = "N"
        self.just_deactivated_mask = prev_active_flights_mask & ~self.active_flights_mask

        return update_wp_mask

    def update_flight_phases(self):
        flight_phases = []
        for i, flight in enumerate(self.flight_arr):
            if not self.active_flights_mask[i]:
                flight_phases.append('innactive')
                continue
            try:
                flight_phase = utils.find_flight_phase(flight[0], self.flight_plans, self.timestamp)
            except:
                print(flight)
                print(self.timestamp)
                assert False
            flight_phases.append(flight_phase)

        return flight_phases

    def initialize(self):
        self.alerts_debug = 0
        self.total_conflicts = 0
        self.log_states=[]
        self.log_flight_phases = []
        self.main_info_list = []
        self.conflict_info_list = []
        self.conflict_params_list = []
        self.projection_list = []
        self.projection_dict = {}
        self.loss_list = []
        self.alert_list = []
        self.timestamp = self.init_timestamp
        # +2 for bearing relative to exit point, altitude_diff w.r.t. exit point, flight_phase
        self.flight_arr = np.zeros([self.flight_num, self.feature_size+3])
        self.sectorIDs = []
        self.active_flights_mask = np.zeros(self.flight_num).astype(bool)
        self.active_historical_mask = np.zeros(self.flight_num).astype(bool)
        self.new_active_flights_mask = np.zeros(self.flight_num).astype(bool)
        self.flight_plan_updated_mask = np.zeros(self.flight_num).astype(bool)
        self.activated_mask = np.zeros(self.flight_num).astype(bool)
        self.just_deactivated_mask = np.zeros(self.flight_num).astype(bool)
        self.flight_sectors = np.array((["C"] * self.flight_arr.shape[0]), dtype=object)
        self.exit_wps = np.zeros((self.flight_arr.shape[0], 9)).astype(object)

        self.next_wps = np.zeros((self.flight_arr.shape[0], 4, 3)).astype(object)
        self.available_wps = np.zeros((self.flight_arr.shape[0], 4))
        self.towards_wps = np.zeros((self.flight_arr.shape[0], 3))
        self.towards_wps_dcourse = np.zeros((self.flight_arr.shape[0]))
        self.dcourse_to_next_wps = np.zeros((self.flight_arr.shape[0], 4))
        self.distance_to_next_wps = np.zeros((self.flight_arr.shape[0], 4))
        self.vertical_distance_to_next_wps = np.zeros((self.flight_arr.shape[0], 4))
        self.executing_direct_to = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.executing_resume_to_fplan = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.executing_course_change = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.action_duration = np.zeros((self.flight_arr.shape[0]))
        self.action_starting_time_point = np.zeros((self.flight_arr.shape[0]))
        self.wp_rank = np.zeros((self.flight_arr.shape[0], 4))
        self.towards_wp_rank = np.zeros((self.flight_arr.shape[0]))
        self.point_rank = np.zeros((self.flight_arr.shape[0]))
        self.next_FL = np.zeros((self.flight_arr.shape[0])).astype(np.int64)
        self.FL_change_sign = np.zeros((self.flight_arr.shape[0]))
        self.executing_FL_change = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.change_FL_vspeed = np.zeros((self.flight_arr.shape[0]))
        self.change_FL_hspeed = np.zeros((self.flight_arr.shape[0], 2))

        self.finished_FL_change = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.finished_direct_to = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.finished_resume_to_fplan = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.finished_course_change_mask = np.zeros((self.flight_arr.shape[0])).astype(bool)
        self.direct_to_idxs = np.zeros((self.flight_arr.shape[0], 4))
        self.direct_to_fp_idx = np.zeros((self.flight_arr.shape[0]))
        self.towards_wp_idxs = np.zeros((self.flight_arr.shape[0]))
        self.towards_wp_fp_idx = np.zeros((self.flight_arr.shape[0]))
        self.wp_per_tstamp = []
        self.episode_num += 1
        self.flight_plans = copy.deepcopy(self.init_flight_plans)
        self.new_active_flights()
        flight_plan_utils.reset_current(self.flight_plans, self.timestamp)
        self.find_current_exit_p_w_fplans((self.active_flights_mask),
                                          self.flight_sectors)
        self.update_features_wrt_exit_point()

        self.total_losses_of_separation = 0
        self.total_alerts = 0
        res_act_ID = np.zeros(self.flight_arr.shape[0])
        edges = self.edges_between_active_flights(res_act_ID)

        edges_w_los, edges_w_conflict, edges_w_conf_not_alerts, edges_w_alerts = self.conflicting_flights(edges)
        edges = np.append(edges,
                          np.array([edges_w_los, edges_w_conflict,
                                    edges_w_conf_not_alerts, edges_w_alerts]).T.astype(float),
                          axis=1)
        edges = np.delete(edges, [11, 13], axis=1)

        for flightKey in self.flight_index:
            self.flight_index[flightKey]['complete_df_idx'] = 0

        self.find_next_wp(self.active_flights_mask)
        self.flight_phases = self.update_flight_phases()

        concated_states = np.concatenate([self.flight_arr, -self.dcourse_to_next_wps,
                                          self.distance_to_next_wps, self.vertical_distance_to_next_wps],
                                         axis=1)

        return concated_states, \
               edges, \
               self.available_wps, \
               self.flight_phases, \
               self.finished_FL_change, \
               self.finished_direct_to, \
               self.finished_resume_to_fplan,\
               self.executing_FL_change, \
               self.executing_direct_to, \
               self.executing_resume_to_fplan

    def update_features_wrt_exit_point(self):
        # compute bearing relative to exit point

        if np.all(~self.active_flights_mask):
            return
        vectorized_mybearing_speed_components = np.vectorize(utils.mybearing_speed_components)
        # ATTENTION: is this already computed in flights_arr ?
        bearing, _ = vectorized_mybearing_speed_components(self.flight_arr[self.active_flights_mask, 6],
                                                           self.flight_arr[self.active_flights_mask, 7])
        fake_s_components = self.exit_wps[self.active_flights_mask, 5:7].astype(np.float64) - \
                            self.flight_arr[self.active_flights_mask, 3:5]
        bearing_to_exit_point, _ = vectorized_mybearing_speed_components(fake_s_components[:, 0],
                                                                         fake_s_components[:, 1])

        self.flight_arr[self.active_flights_mask, 12] = (self.exit_wps[:, 3] - self.flight_arr[:, 5])[self.active_flights_mask]

        self.flight_arr[self.active_flights_mask, 13] = \
            np.sqrt(np.sum(((self.flight_arr[:, 3:5] - self.exit_wps[:, 5:7])
                    [self.active_flights_mask])**2, axis=1).astype(np.float32))

        # ATTENTION: absolute value maybe?
        bearing_relative_to_exit_point = bearing - bearing_to_exit_point
        bearing_relative_to_exit_point = np.where(np.abs(bearing_relative_to_exit_point) > 180,
                                                  (360-np.abs(bearing_relative_to_exit_point))*np.sign(bearing_relative_to_exit_point)*(-1),
                                                  bearing_relative_to_exit_point)

        self.flight_arr[self.active_flights_mask, 11] = bearing_relative_to_exit_point

    def move_flights(self, actions, mask):

        self.flight_arr[mask, 8:11] += actions[mask]
        self.flight_arr[mask, 6] = (np.sin(np.radians(self.flight_arr[mask, 8]))) * self.flight_arr[mask, 9]
        self.flight_arr[mask, 7] = (np.cos(np.radians(self.flight_arr[mask, 8]))) * self.flight_arr[mask, 9]

        self.flight_arr[mask, 3] += self.flight_arr[mask, 6] * self.dt
        self.flight_arr[mask, 4] += self.flight_arr[mask, 7] * self.dt
        self.flight_arr[mask, 5] += self.flight_arr[mask, 10] * self.dt

    def reward_fun(self, actions, edges):
        """
        Computes the reward function
        :param actions: #N x [dcourse,dspeed,d_alt_speed]
        :param edges: Edges between active agents matrix #N x 11
        :return: Reward for each active agent
        """
        actions_not_zero_mask = (actions != 0).astype(int)

        actions_rw = -actions_not_zero_mask * 1

        actions_reward = np.sum(-actions_not_zero_mask*1, axis=1)
        actions_reward[~self.active_flights_mask] = 0
        actions_rw[~self.active_flights_mask] = 0

        flight_edges_reward = np.zeros(self.flight_arr.shape[0])
        hsm = env_config['horizontal_sep_minimum']*1852
        vsm_init = 1000
        drift_reward = -0.5*(np.abs(self.flight_arr[:, 11])/180)
        drift_reward[~self.active_flights_mask] = 0

        altitude_reward = -0.5*(np.abs(self.flight_arr[:, 5] - self.exit_wps[:, 3])/1000)
        altitude_reward[~self.active_flights_mask] = 0

        total_losses_of_separation = 0
        total_alerts = 0
        losses_rw = np.zeros(shape=[self.flight_arr.shape[0], 1])
        alerts_rw = np.zeros(shape=[self.flight_arr.shape[0], 1])
        for edge in edges:
            if edge[0] == edge[1]:
                continue
            f1_key = self.flight_index[edge[0]]['idx']
            f2_key = self.flight_index[edge[1]]['idx']
            vsm = vsm_init

            if self.flight_arr[int(f1_key)][5] >= 41000 or self.flight_arr[int(f2_key)][5] >= 41000:
                vsm = 2000

            los = 0
            if edge[9] < hsm and edge[10] < vsm:
                los = 1
                total_losses_of_separation += 1
                losses_rw[f1_key] -= 10
                self.loss_list.append(self.flight_arr[f1_key].tolist()+[self.timestamp])

            c_alert = 0
            c_vsm = edge[11]

            if edge[12] == 1 and edge[2] < 2 * 60 and edge[2] >= 0 and edge[3] < hsm and np.abs(edge[6]) < c_vsm:
                total_alerts += 1
                c_alert = 1
                alerts_rw[int(f1_key)] -= 5
                self.alert_list.append(self.flight_arr[f1_key].tolist() + [self.timestamp])

            r = -(los*10 + c_alert*5)

            flight_edges_reward[int(f1_key)] += r

        r_per_factor = np.concatenate([actions_rw, drift_reward[:, np.newaxis],
                                       altitude_reward[:, np.newaxis], losses_rw, alerts_rw],
                                      axis=1)

        total_r = drift_reward + actions_reward + flight_edges_reward + altitude_reward

        # /2 because there are 2 edges for each loss of separation
        self.total_losses_of_separation += total_losses_of_separation/2
        self.total_alerts += total_alerts/2

        return total_r, r_per_factor

    def find_exit_wp(self):
        for key in self.flight_index:
            flightKey = int(key)

            exit_point = self.flight_index[flightKey]['last_point'].values[0, 1:]
            # exit point: lat lon timestamp alt Sector x y flightKey
            exit_point = exit_point[np.array([0, 1, 3, 2, 4])]

            x, y = utils.utils_global['transformer'].transform(exit_point[1], exit_point[0])
            exit_point = np.append(exit_point, [x, y, flightKey])

            utc_date = datetime.utcfromtimestamp(exit_point[2])
            date_str = utc_date.strftime("%Y-%m-%d %H:%M:%S")
            exit_point[2] = date_str

            self.exit_wps[self.flight_index[flightKey]['idx']] = exit_point

    def exit_point_from_radar(self, flightKey):
        exit_point = self.last_points[flightKey][0, 1:]
        # exit point: lat lon timestamp alt Sector x y flightKey
        exit_point = exit_point[np.array([0, 1, 3, 2, 4])]
        x, y = utils.utils_global['transformer'].transform(exit_point[1], exit_point[0])
        exit_point = np.append(exit_point, [x, y, np.inf, flightKey])

        return exit_point

    def exit_point_from_radar_2(self, flightKey, sector):
        exit_point = self.last_points[flightKey][0, 1:]
        df = self.flight_index[flightKey]['complete_df']
        try:
            exit_point = df[df['Sector'] == sector].tail(1)
            if exit_point.shape[0] == 0:
                exit_point = df.tail(1).values[0,1:]
            else:
                exit_point = exit_point.values[0,1:]
        except Exception as e:
            print(e)
        # exit point: lat lon timestamp alt Sector x y flightKey
        try:
            exit_point = exit_point[np.array([0, 1, 3, 2, 4])]
        except Exception as e:
            print(e)
        x, y = utils.utils_global['transformer'].transform(exit_point[1], exit_point[0])
        exit_point = np.append(exit_point, [x, y, np.inf, flightKey])

        return exit_point

    def find_current_exit_p_w_fplans(self, flights_mask, sectorID_np):
        for flight_feature in self.flight_arr[flights_mask]:

            flightKey = int(flight_feature[0])
            idx = self.flight_index[flightKey]['idx']
            sector = sectorID_np[idx]
            flight_plans = self.flight_plans[flightKey]

            if len(flight_plans['fplans']) == 0:
                self.exit_wps[self.flight_index[flightKey]['idx']] = self.exit_point_from_radar_2(flightKey, sector)
                continue

            current_sector = False
            downstream_sector = False
            exit_point_idx = 0

            for idx, wp in enumerate(flight_plans['fplans'][flight_plans['current']][1].values):

                if wp[4] == sector:
                    current_sector = True
                    exit_point = wp
                    exit_point_idx = idx

                else:
                    if current_sector:
                        exit_point = wp
                        exit_point_idx = idx
                        break

            if exit_point_idx > 0:
                prev_wp = flight_plans['fplans'][flight_plans['current']][1].values[exit_point_idx - 1]
                t_start = prev_wp[2]
                t_end = exit_point[2]

                line_segment = np.array([prev_wp[np.array([5, 6, 3])], exit_point[np.array([5, 6, 3])]])
                segment_distance = np.sqrt(np.sum((line_segment[1][:2] - line_segment[0][:2]) ** 2))
                if t_end - t_start == 0:
                    t_end += 30
                speed_magnitude = segment_distance / (t_end - t_start)

                flag, keyFalse, sector_active, exit_point = \
                    self.sector_data.line_segment_sector_intersection(line_segment, t_start, sector, True)

                if flag:
                    lon, lat = utils.utils_global['inverse_transformer'].transform(exit_point[0], exit_point[1])

                    exit_point_dist = np.sqrt(np.sum((exit_point[:2] - line_segment[0][:2]) ** 2))
                    exit_point_time = t_start + exit_point_dist / speed_magnitude

                    p_start = flight_plans['fplans'][flight_plans['current']][3]
                    p_end = flight_plans['fplans'][flight_plans['current']][4]
                    rank = flight_plan_utils.compute_rank(exit_point[:2] - p_start, p_end - p_start)

                    exit_point = np.array([lat, lon, exit_point_time, exit_point[2],
                                           sector, exit_point[0], exit_point[1], rank, flightKey],
                                          dtype=object)

                else:
                    exit_point = self.exit_point_from_radar_2(flightKey, sector)

            else:
                exit_point = self.exit_point_from_radar_2(flightKey, sector)

            self.exit_wps[self.flight_index[flightKey]['idx']] = exit_point

    def find_exit_wp_w_fplans(self, flights_mask):
        for flight_feature in self.flight_arr[flights_mask]:
            flightKey = int(flight_feature[0])
            flight_plans = self.flight_plans[flightKey]

            if len(flight_plans['fplans']) == 0:

                self.exit_wps[self.flight_index[flightKey]['idx']] = self.exit_point_from_radar(flightKey)
                continue

            current_sector = False
            downstream_sector = False
            exit_point_idx = 0
            sector = None
            for idx, wp in enumerate(flight_plans['fplans'][flight_plans['current']][1].values):

                if wp[4] == self.current_sector:
                    current_sector = True
                    sector = self.current_sector
                    exit_point = wp
                    exit_point_idx = idx
                elif wp[4] == self.downstream_sector:
                    sector = self.downstream_sector
                    downstream_sector = True
                    exit_point = wp
                    exit_point_idx = idx
                else:
                    if current_sector or downstream_sector:
                        exit_point = wp
                        exit_point_idx = idx
                        break

            if exit_point_idx > 0:
                prev_wp = flight_plans['fplans'][flight_plans['current']][1].values[exit_point_idx-1]
                t_start = prev_wp[2]
                t_end = exit_point[2]

                line_segment = np.array([prev_wp[np.array([5, 6, 3])], exit_point[np.array([5, 6, 3])]])
                segment_distance = np.sqrt(np.sum((line_segment[1][:2]-line_segment[0][:2])**2))
                if t_end - t_start == 0:
                    t_end += 30
                speed_magnitude = segment_distance/(t_end - t_start)

                flag, keyFalse, sector_active, exit_point = \
                    self.sector_data.line_segment_sector_intersection(line_segment, t_start, sector, True)

                if flag:
                    lon, lat = utils.utils_global['inverse_transformer'].transform(exit_point[0], exit_point[1])

                    exit_point_dist = np.sqrt(np.sum((exit_point[:2]-line_segment[0][:2])**2))
                    exit_point_time = t_start + exit_point_dist/speed_magnitude

                    p_start = flight_plans['fplans'][flight_plans['current']][3]
                    p_end = flight_plans['fplans'][flight_plans['current']][4]
                    rank = flight_plan_utils.compute_rank(exit_point[:2]-p_start, p_end-p_start)

                    exit_point = \
                        np.array([lat, lon, exit_point_time, exit_point[2],
                                  sector, exit_point[0], exit_point[1], rank, flightKey],
                                 dtype=object)

                    flight_plans['fplans'][flight_plans['current']][1].iloc[exit_point_idx] = exit_point[:-1]
                    flight_plans['fplans'][flight_plans['current']][1] = \
                        flight_plans['fplans'][flight_plans['current']][1].iloc[:exit_point_idx+1]

                else:
                    exit_point = self.exit_point_from_radar(flightKey)
                    new_row = \
                        {col: exit_point[i] for i, col in enumerate(flight_plans['fplans'][flight_plans['current']][1])}

                    flight_plans['fplans'][flight_plans['current']][1] = \
                        flight_plans['fplans'][flight_plans['current']][1].append(new_row, ignore_index=True)

            else:
                exit_point = self.exit_point_from_radar(flightKey)
                new_row = {col: exit_point[i] for i, col in
                           enumerate(flight_plans['fplans'][flight_plans['current']][1])}

                flight_plans['fplans'][flight_plans['current']][1] = \
                    flight_plans['fplans'][flight_plans['current']][1].append(new_row, ignore_index=True)

            self.exit_wps[self.flight_index[flightKey]['idx']] = exit_point

    def update_lon_lat_from_x_y(self, x, y):

        lon, lat = utils.utils_global['inverse_transformer'].transform(x, y)

        return lon, lat

    def update_lon_lat_from_x_y_vec(self):
        if np.all(~self.active_flights_mask):
            return
        update_lon_lat_from_x_y_vectorized = np.vectorize(self.update_lon_lat_from_x_y)
        active_flights = self.flight_arr[self.active_flights_mask]
        lon, lat = update_lon_lat_from_x_y_vectorized(active_flights[:, 3], active_flights[:, 4])
        self.flight_arr[self.active_flights_mask, 1] = lon
        self.flight_arr[self.active_flights_mask, 2] = lat

    def clip_actions(self, actions):
        active_actions = actions[self.active_flights_mask]
        total_speeds = self.flight_arr[self.active_flights_mask, 9:11] + active_actions[:, 1:]
        clipped_max_h_speeds = \
            np.where(total_speeds[:, 0] > env_config['max_h_speed'],
                     active_actions[:, 1]-(total_speeds[:, 0]-env_config['max_h_speed']),
                     active_actions[:, 1])
        clipped_max_min_h_speeds = \
            np.where(total_speeds[:, 0] < env_config['min_h_speed'],
                     clipped_max_h_speeds-(total_speeds[:, 0]-env_config['min_h_speed']),
                     clipped_max_h_speeds)
        clipped_max_alt_speeds = \
            np.where(total_speeds[:, 1] > env_config['max_alt_speed'],
                     active_actions[:, 2]-(total_speeds[:, 1]-env_config['max_alt_speed']),
                     active_actions[:, 2])
        clipped_max_min_alt_speeds = \
            np.where(total_speeds[:, 1] < env_config['min_alt_speed'],
                     clipped_max_alt_speeds - (total_speeds[:, 1]-env_config['min_alt_speed']),
                     clipped_max_alt_speeds)

        clipped_actions = np.copy(actions)
        clipped_actions[self.active_flights_mask, 1] = clipped_max_min_h_speeds
        clipped_actions[self.active_flights_mask, 2] = clipped_max_min_alt_speeds

        return clipped_actions

    def conflicting_flights(self, edges):

        edges_w_los = \
            np.where((np.abs(edges[:, 10]) < edges[:, 13]) & (edges[:, 12] == 1),
                     edges[:, 9] < env_config['horizontal_sep_minimum'] * 1852,
                     False)

        # edges: [flights_id_i, flights_id_j, tcpa, dcpa, intersection_angle,
        #  bearing_cpa_ownship_neighbour_relative, v_d_cpa,
        #  distance at crossing point, time_to_crossing_point,
        #  horizontal_distance, vertical_distance, v_sep_minimum_cpa, 1, v_sep_minimum ]

        edges_w_conflict = np.where((np.abs(edges[:, 6]) < edges[:, 11]) & (edges[:, 12] == 1),
                                    edges[:, 3] < env_config['horizontal_sep_minimum'] * 1852,
                                    False)

        edges_w_alerts = edges_w_conflict & (edges[:, 2] < 2 * 60) & (edges[:, 2] >= 0)
        edges_w_conf_not_alerts = edges_w_conflict & ((edges[:, 2] >= 2 * 60) | (edges[:, 2] < 0))
        self.total_conflicts += (np.where(edges_w_conflict)[0].shape[0])/2

        return edges_w_los, edges_w_conflict, edges_w_conf_not_alerts, edges_w_alerts

    def find_next_from_historical_data(self, mask):
        """
        ['flightKey', 'longitude', 'latitude',
         'x', 'y', 'altitude',
         'speed_x_component', 'speed_y_component',
         'course', 'speed', 'alt_speed','bearing_relative_to_exit_wp',
         'vertical_distance_to_exit_point','horizontal_distance_to_exit_point']

        :return:
        """
        speed_of_next_points = []

        for flight in self.flight_arr[self.active_historical_mask & mask]:
            flightKey = flight[0]
            current_point = \
                self.flight_index[flightKey]['complete_df'].iloc[self.flight_index[flightKey]['complete_df_idx']]

            if self.flight_index[flightKey]['complete_df'].shape[0]-1 == self.flight_index[flightKey]['complete_df_idx']:
                self.active_historical_mask[self.flight_index[flightKey]['idx']] = False
                self.flight_index[flightKey]['complete_df_idx'] = 0

                continue

            next_point = \
                self.flight_index[flightKey]['complete_df'].iloc[self.flight_index[flightKey]['complete_df_idx']+1]
            next_point_np = next_point.values
            self.flight_index[flightKey]['complete_df_idx'] += 1
            idx = self.flight_index[flightKey]['idx']

            self.flight_arr[idx, 6:11] = next_point[np.array([10, 11, 8, 12, 9])]
            #speeds 10,11,8,12,9
            self.flight_arr[idx, 1:6] = next_point_np[np.array([2, 1, 6, 7, 3])]

            self.speed_of_next_point = next_point_np[np.array([10, 11, 8, 12, 9])]
            speed_of_next_points.append(self.speed_of_next_point)

        return np.array(speed_of_next_points)

    def detect_conflicts_between_active_flights(self):
        active_flights_idxs = np.where((self.active_historical_mask) == True)[0]
        active_flights = self.flight_arr[active_flights_idxs]

        # [0: RTkey, 1: x, 2: y, 3: x_speed, 4: y_speed, 5: altitudeInFeets, 6: alt_speed, 7: max_altitude]
        main_info, conflict_info, \
        conflict_params, projections, edges = \
            utils.compute_conflicts(active_flights[:, np.array([0, 3, 4, 6, 7, 5, 10])],
                                    self.flight_plans, self.timestamp,
                                    [self.current_sector, self.downstream_sector],
                                    self.sector_data, self.exit_wps, self.action_duration,
                                    self.action_starting_time_point, self.executing_course_change,
                                    self.finished_course_change_mask, self.executing_direct_to,
                                    self.towards_wp_idxs, self.towards_wp_fp_idx)

        # change x,y in main and conflict info to lon lat
        if len(main_info) > 0:
            main_info_np = np.array(main_info)

            main_info_np[:, 13:15] = flight_utils.transform_coordinates_np(main_info_np[:, 13:15], inverse=True)
            main_info_np[:, 16:18] = flight_utils.transform_coordinates_np(main_info_np[:, 16:18], inverse=True)

            conflict_info_np = np.array(conflict_info)
            conflict_info_np[:, 2:4] = flight_utils.transform_coordinates_np(conflict_info_np[:, 2:4], inverse=True)

            self.main_info_list.extend(main_info_np.tolist())
            self.conflict_info_list.extend(conflict_info_np.tolist())
            self.conflict_params_list.extend(conflict_params)
            self.projection_list.extend(projections)

    def write_loss_conf_main(self):

        states_np = np.concatenate(self.log_states, axis=0)
        states_df = pd.DataFrame(states_np, columns=['RTkey', 'longitude', 'latitude',
                                                     'x', 'y', 'altitude',
                                                     'speed_x_component', 'speed_y_component',
                                                     'course', 'speed', 'alt_speed',
                                                     'relative_bearing_to_exit_point',
                                                     'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                     ['wp_dcourse_' + str(i + 1) for i in range(4)] +
                                                     ['wp_hdistance_' + str(i + 1) for i in range(4)] +
                                                     ['wp_vdistance_' + str(i + 1) for i in range(4)] +
                                                     ['timestamp'])
        states_df['flight_phase'] = np.concatenate(self.log_flight_phases, axis=0)
        states_df = states_df.astype({'timestamp': 'int64', 'RTkey': 'int64'})
        states_df.to_csv(env_config['log_dir'] + '/states_out_' + str(self.episode_num) + '.csv', index=False)

        loss_df = pd.DataFrame(self.loss_list, columns=['flightKey', 'longitude', 'latitude',
                                                        'x', 'y', 'altitude',
                                                        'speed_x_component', 'speed_y_component',
                                                        'course', 'speed', 'alt_speed',
                                                        'relative_bearing_to_exit_point',
                                                        'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point',
                                                        'step'])
        loss_df.to_csv(env_config['log_dir'] + '/loss_out_' + str(self.episode_num) + '.csv', index=False)

        alert_df = pd.DataFrame(self.alert_list, columns=['flightKey', 'longitude', 'latitude',
                                                          'x', 'y', 'altitude',
                                                          'speed_x_component', 'speed_y_component',
                                                          'course', 'speed', 'alt_speed',
                                                          'relative_bearing_to_exit_point',
                                                          'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point',
                                                          'step'])
        alert_df.to_csv(env_config['log_dir'] + '/alert_out_' + str(self.episode_num) + '.csv', index=False)
        # add conflict type, flight phase1, flight phase2, projection time horizon
        main_info_df = pd.DataFrame(self.main_info_list, columns=['TimePoint', 'RTkey1', 'RTkey2',
                                                                  'fp_projection_flag1', 'fp_id1',
                                                                  'fp_projection_flag2', 'fp_id2',
                                                                  'course1', 'course2', 'speed_h1', 'speed_h2',
                                                                  'speed_v1', 'speed_v2', 'lon1', 'lat1', 'alt1',
                                                                  'lon2', 'lat2', 'alt2', 'conflict_ID',
                                                                  'flight_phase_1', 'flight_phase_2',
                                                                  'projection_time_horizon', 'event_type', 'due_to_flight_1',
                                                                  'due_to_flight_2', 'command_category'])
        main_info_df = main_info_df.astype({'TimePoint': 'int64', 'RTkey1': 'int64', 'RTkey2': 'int64'})
        main_info_df.to_csv(env_config['log_dir'] + '/main.csv', index=False)
        # add due to, Sector ID
        conflict_info_df = pd.DataFrame(self.conflict_info_list, columns=['conflict_ID', 'RTkey',
                                                                          'conflict_lon', 'conflict_lat',
                                                                          'conflict_alt',
                                                                          'time_to_conflict',
                                                                          'h_distance_at_conflict',
                                                                          'v_distance_at_conflict',
                                                                          'first_conflict_lon', 'first_conflict_lat',
                                                                          'first_conflict_alt',
                                                                          'time_to_first_conflict',
                                                                          'h_distance_at_first_conflict',
                                                                          'v_distance_at_first_conflict',
                                                                          'last_conflict_lon', 'last_conflict_lat',
                                                                          'last_conflict_alt',
                                                                          'time_to_last_conflict',
                                                                          'h_distance_at_last_conflict',
                                                                          'v_distance_at_last_conflict',
                                                                          'crossing_point_lon', 'crossing_point_lat',
                                                                          't_to_crossing_point',
                                                                          'd_h_cp', 'd_v_cp', 'sectorID', 'due_to_flight_1',
                                                                          'due_to_flight_2', 'command_category',
                                                                          'projection_ID'])
        conflict_info_df = conflict_info_df.astype({'RTkey': 'int64'})
        conflict_info_df.to_csv(env_config['log_dir'] + '/conflicts.csv', index=False)

        conflict_params_df = pd.DataFrame(self.conflict_params_list, columns=['conflict_ID', 'RTkey',
                                                                              'FP_Track_Cross_Point_Long',
                                                                              'FP_Track_Cross_Point_Lat',
                                                                              'FP_Track_Cross_Point_Alt',
                                                                              'Track_Course',
                                                                              'Flight_Plan_Course',
                                                                              'Flight_Plan_Alt',
                                                                              'Flight_Plan_HSpeed',
                                                                              'Flight_Plan_VSpeed',
                                                                              'due_to_flight_1',
                                                                              'due_to_flight_2',
                                                                              'command_category'])
        conflict_params_df = conflict_params_df.astype({'RTkey': 'int64'})
        conflict_params_df.to_csv(env_config['log_dir'] + '/conflict_params.csv', index=False)

        projections_df = pd.DataFrame(self.projection_list, columns=['projection_ID', 'RTkey', 'TimePoint',
                                                                     'resolution_action_type_value', 'sequence_number',
                                                                     'lon', 'lat', 'timestamp', 'alt'])
        projections_df = projections_df.astype({'RTkey': 'int64',
                                                'timestamp': 'float64',
                                                'TimePoint': 'float64'}).astype({'timestamp': 'int64',
                                                                                 'TimePoint': 'int64'})
        projections_df.to_csv(env_config['log_dir'] + '/points_of_projection.csv', index=False)
        exit_points_df = pd.DataFrame(self.wp_per_tstamp, columns=['lat', 'lon', 'eto', 'alt', 'Sector',
                                                                   'x', 'y', 'rank', 'RTkey', 'timepoint'])
        exit_points_df = exit_points_df.astype({'RTkey': 'int64',
                                                'timepoint': 'float64'}).astype({'timepoint': 'int64'})
        exit_points_df.to_csv(env_config['log_dir'] + '/exit_points.csv', index=False)

        # parameters for detecting conflicts
        # ConflictID, RTkey, FP_Track_Cross_Point_Long
        # FP_Track_Cross_Point_Lat
        # FP_Track_Cross_Point_Alt
        # Track_Course
        # Flight_Plan_Course
        # Flight_Plan_Alt
        # Flight_Plan_VSpeed
        # Flight_Plan_HSpeed
        # Points of projection

    def step_from_historical_data(self):
        mask = np.ones((self.flight_arr.shape[0])).astype(bool)
        speed_of_next_points_np = self.find_next_from_historical_data(mask)
        self.update_fplans_w_mask(self.active_historical_mask)
        prev_active_historical = np.copy(self.active_historical_mask)
        self.new_active_flights()

        done = False
        if ~np.any(self.active_historical_mask) and np.all(self.activated_mask):
            done = True

            states_np = np.concatenate(self.log_states, axis=0)
            states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                         'x', 'y', 'altitude',
                                                         'speed_x_component', 'speed_y_component',
                                                         'course', 'speed', 'alt_speed',
                                                         'relative_bearing_to_exit_point',
                                                         'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point',
                                                         'step'])
            states_df.to_csv(env_config['log_dir'] + '/states_out_' + str(self.episode_num) + '.csv', index=False)
            loss_df = pd.DataFrame(self.loss_list, columns=['flightKey', 'longitude', 'latitude',
                                                            'x', 'y', 'altitude',
                                                            'speed_x_component', 'speed_y_component',
                                                            'course', 'speed', 'alt_speed',
                                                            'relative_bearing_to_exit_point',
                                                            'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point',
                                                            'step'])
            loss_df.to_csv(env_config['log_dir'] + '/loss_out_' + str(self.episode_num) + '.csv', index=False)

            main_info_df = pd.DataFrame(self.main_info_list, columns=['TimePoint', 'RTkey1', 'RTkey2',
                                                                      'fp_projection_flag1', 'fp_id1',
                                                                      'fp_projection_flag2', 'fp_id2',
                                                                      'course1', 'course2', 'speed_h1', 'speed_h2',
                                                                      'speed_v1', 'speed_v2', 'lon1', 'lat1', 'alt1',
                                                                      'lon2', 'lat2', 'alt2', 'conflict_ID',
                                                                      'flight_phase_1', 'flight_phase_2',
                                                                      'projection_time_horizon', 'type'])
            main_info_df = main_info_df.astype({'TimePoint': 'int64', 'RTkey1': 'int64', 'RTkey2': 'int64'})
            main_info_df.to_csv(env_config['log_dir'] + '/main.csv', index=False)
            conflict_info_df = pd.DataFrame(self.conflict_info_list, columns=['conflict_ID', 'RTkey',
                                                                              'conflict_lon', 'conflict_lat',
                                                                              'conflict_alt',
                                                                              'time_to_conflict',
                                                                              'h_distance_at_conflict',
                                                                              'v_distance_at_conflict',
                                                                              'crossing_point_lon',
                                                                              'crossing_point_lat',
                                                                              't_to_crossing_point',
                                                                              'd_h_cp', 'd_v_cp', 'sectorID'])
            conflict_info_df = conflict_info_df.astype({'RTkey': 'int64'})
            conflict_info_df.to_csv(env_config['log_dir'] + '/conflicts.csv', index=False)
        self.find_exit_wp_w_fplans((self.new_active_flights_mask | self.flight_plan_updated_mask))
        self.detect_conflicts_between_active_flights()
        step_arr = np.array([[self.timestamp]] * self.flight_arr[self.active_historical_mask].shape[0])
        if self.flight_arr[self.active_historical_mask].shape[0] > 0:
            self.log_states.append(np.concatenate([self.flight_arr[self.active_historical_mask],
                                                   step_arr], axis=1))
            self.log_flight_phases.append(np.array(self.flight_phases)[self.active_historical_mask])

        if speed_of_next_points_np.shape[0] > 0:
            self.flight_arr[prev_active_historical, 6:11] = speed_of_next_points_np

        self.timestamp += self.dt

        return done

    def compute_shift_from_exit_point(self, RTkey, res_act):
        """[dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
            direct_to_3rd, direct_to_4th, continue_action, duration, resume_to_flight_plan]
         ewp: [lat lon timestamp alt Sector x y flightKey]
         """

        agent_i = self.flight_index[RTkey]['idx']
        course_speed = np.empty((3)).astype(np.float64)
        course_speed[:2] = self.flight_arr[agent_i, 8:10] + res_act[:2]
        course_speed[2] = self.flight_arr[agent_i, 10] + res_act[2]

        direct_to = res_act[np.array([3, 5, 6, 7])].astype(bool)
        if np.any(direct_to):
            course_to_next_wp = self.dcourse_to_next_wps[agent_i, direct_to]
            abs_course = abs(course_to_next_wp)
            if abs_course > 180:
                abs_course = 360 - abs_course
            if abs_course <= 90:
                course_speed[0] = course_speed[0]+course_to_next_wp

        speed_x = (np.sin(np.radians(course_speed[0]))) * course_speed[1]
        speed_y = (np.cos(np.radians(course_speed[0]))) * course_speed[1]
        course_speed[:2] = [speed_x, speed_y]

        p = self.flight_arr[agent_i, 3:6]
        next_p = p+course_speed*self.dt

        bearing, _ = utils.mybearing(p[:2], next_p[:2])
        exit_point = self.exit_wps[agent_i, np.array([5, 6, 3])]
        bearing_to_exit_point, _ = utils.mybearing(next_p[:2], exit_point[:2].astype(np.float64))

        bearing_relative_to_exit_point = bearing - bearing_to_exit_point
        if abs(bearing_relative_to_exit_point) > 180:
            bearing_relative_to_exit_point = \
                (360 - np.abs(bearing_relative_to_exit_point)) * np.sign(bearing_relative_to_exit_point) * (-1)

        v_shift_from_exit_point = next_p[2]-exit_point[2]

        return bearing_relative_to_exit_point, v_shift_from_exit_point, bearing

    def actions_impact(self, action_dict):
        """

        :param action_dict:  {"res_acts": np arr Nx3x5, "res_acts_ID": np arr Nx3,
                              "filt_out_mask": Nx3 (True for filtered actions)}
        :return:
        """

        for agent_i, top_res_acts in enumerate(action_dict['res_acts']):
            if not self.active_flights_mask[agent_i]:
                continue
            for act_i, res_act in enumerate(top_res_acts):
                if action_dict['filt_out_mask'][agent_i, act_i] or action_dict['res_acts_ID'][agent_i][act_i] == 0:
                    continue

                state_i = np.copy(self.flight_arr[agent_i])
                duration_i = np.copy(self.action_duration[agent_i])
                action_starting_time_point_i = np.copy(self.action_starting_time_point[agent_i])
                self.action_duration[agent_i] = res_act[9]
                self.action_starting_time_point[agent_i] = self.timestamp
                # RECONSIDER: for action vertical speed change
                self.flight_arr[agent_i, 8:11] += res_act[:3]
                self.flight_arr[agent_i, 6] = \
                    (np.sin(np.radians(self.flight_arr[agent_i, 8]))) * self.flight_arr[agent_i, 9]
                self.flight_arr[agent_i, 7] = \
                    (np.cos(np.radians(self.flight_arr[agent_i, 8]))) * self.flight_arr[agent_i, 9]
                edges = self.impact_conflicts(self.flight_arr[agent_i][0], action_dict['res_acts_ID'][agent_i][act_i])
                self.flight_arr[agent_i] = state_i
                self.action_duration[agent_i] = duration_i
                self.action_starting_time_point[agent_i] = action_starting_time_point_i

    def compute_point_ranks(self, mask):
        point_rank_arr = np.zeros(self.point_rank.shape[0])
        for fkey_point in self.flight_arr[mask][:, np.array([0, 3, 4, 5])]:
            flightKey = fkey_point[0]
            point = fkey_point[1:]
            if len(self.flight_plans[flightKey]['fplans']) == 0:
                p_start = self.flight_index[flightKey]['df'][['x', 'y']].values[0]
                p_end = self.flight_index[flightKey]['last_point'][['x', 'y']].values[0]
                point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)
                point_rank_arr[self.flight_index[flightKey]['idx']] = point_rank
                continue

            p_start = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][3]
            p_end = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][4]
            point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)
            point_rank_arr[self.flight_index[flightKey]['idx']] = point_rank

        return point_rank_arr

    def next_wp_per_flight(self, flight_idx, point, course):

        flightKey = self.flight_arr[flight_idx][0]

        if len(self.flight_plans[flightKey]['fplans']) == 0:
            next_wp = self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6])]
            dcourse_to_next_wp = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

            p_start = self.flight_index[flightKey]['df'][['x', 'y']].values[0]
            p_end = self.flight_index[flightKey]['last_point'][['x', 'y']].values[0]
            p_end_rank = flight_plan_utils.compute_rank(p_end - p_start, p_end - p_start)
            point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

            abs_dcourse = abs(dcourse_to_next_wp)
            if abs_dcourse > 180:
                abs_dcourse = 360 - abs_dcourse

            if point_rank < p_end_rank and abs_dcourse <= 90:
                return next_wp, -1
            else:
                return None, -1

        fplan = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][1]
        p_start = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][3]
        p_end = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][4]
        point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

        idx = np.searchsorted(fplan['rank'].values, point_rank)
        if idx >= fplan.shape[0]:

            next_wp = self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6])]
            dcourse_to_next_wp = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

            p_start = self.flight_index[flightKey]['df'][['x', 'y']].values[0]
            p_end = self.flight_index[flightKey]['last_point'][['x', 'y']].values[0]
            p_end_rank = flight_plan_utils.compute_rank(p_end - p_start, p_end - p_start)
            point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

            abs_dcourse = abs(dcourse_to_next_wp)
            if abs_dcourse > 180:
                abs_dcourse = 360 - abs_dcourse

            if point_rank < p_end_rank and abs_dcourse <= 90:
                return next_wp, -1
            else:
                return None, -1

        else:
            number_of_next = fplan.shape[0] - idx

            for i in range(number_of_next):
                next_wp = fplan[['x', 'y']].iloc[idx+i].values
                bearing_to_next, _ = utils.mybearing(point, next_wp.astype(np.float64))
                dcourse_to_next_wp = bearing_to_next - course
                abs_dcourse = abs(dcourse_to_next_wp)
                if abs_dcourse > 180:
                    abs_dcourse = 360 - abs_dcourse

                if abs_dcourse <= 90:
                    return next_wp, idx+i

            next_wp = self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6])]
            dcourse_to_next_wp = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

            p_start = self.flight_index[flightKey]['df'][['x', 'y']].values[0]
            p_end = self.flight_index[flightKey]['last_point'][['x', 'y']].values[0]
            p_end_rank = flight_plan_utils.compute_rank(p_end - p_start, p_end - p_start)
            point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)

            abs_dcourse = abs(dcourse_to_next_wp)
            if abs_dcourse > 180:
                abs_dcourse = 360 - abs_dcourse

            if point_rank < p_end_rank and abs_dcourse <= 90:
                return next_wp, -1
            else:
                return None, -1

    def find_next_wp(self, mask):
        self.next_wps.fill(0)
        self.distance_to_next_wps.fill(np.nan)
        self.vertical_distance_to_next_wps.fill(np.nan)
        self.dcourse_to_next_wps.fill(np.nan)

        for fkey_point in self.flight_arr[mask][:, np.array([0, 3, 4, 5])]:
            flightKey = fkey_point[0]
            point = fkey_point[1:]
            self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [0, 0, 0, 0]
            self.direct_to_fp_idx[self.flight_index[flightKey]['idx']] = -1
            if len(self.flight_plans[flightKey]['fplans']) == 0:
                self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :] = \
                    self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6, 3])]
                dcourse = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

                p_start = self.flight_index[flightKey]['df'][['x', 'y']].values[0]
                p_end = self.flight_index[flightKey]['last_point'][['x', 'y']].values[0]
                p_end_rank = flight_plan_utils.compute_rank(p_end - p_start, p_end - p_start)
                point_rank = flight_plan_utils.compute_rank(point[:2] - p_start, p_end - p_start)
                self.point_rank[self.flight_index[flightKey]['idx']] = point_rank

                self.available_wps[self.flight_index[flightKey]['idx']] = [1, 0, 0, 0]
                self.distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = np.sqrt(
                    np.sum((self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :2] - point[:2]) ** 2, axis=-1))
                self.vertical_distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = \
                    self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, 2] - point[2]
                self.dcourse_to_next_wps[self.flight_index[flightKey]['idx'], 0] = dcourse
                abs_dcourse = abs(dcourse)
                if abs(dcourse) > 180:
                    abs_dcourse = 360 - abs(dcourse)
                if abs_dcourse >= 90:
                    self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-2, 0, 0, 0]
                else:
                    self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-1, 0, 0, 0]

                self.wp_rank[self.flight_index[flightKey]['idx'], 0] = p_end_rank

                continue

            fplan = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][1]
            p_start = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][3]
            p_end = self.flight_plans[flightKey]['fplans'][self.flight_plans[flightKey]['current']][4]
            self.direct_to_fp_idx[self.flight_index[flightKey]['idx']] = self.flight_plans[flightKey]['current']
            point_rank = flight_plan_utils.compute_rank(point[:2]-p_start, p_end-p_start)
            self.point_rank[self.flight_index[flightKey]['idx']] = point_rank
            idx = np.searchsorted(fplan['rank'].values, point_rank)

            if idx >= fplan.shape[0]:

                self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :] = \
                    self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6, 3])]
                dcourse = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

                self.available_wps[self.flight_index[flightKey]['idx']] = [1, 0, 0, 0]
                self.wp_rank[self.flight_index[flightKey]['idx'], 0] = flight_plan_utils.compute_rank(
                    self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6])]-p_start, p_end-p_start)
                self.dcourse_to_next_wps[self.flight_index[flightKey]['idx'], 0] = dcourse
                self.distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = np.sqrt(
                    np.sum((self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :2] - point[:2]) ** 2, axis=-1))
                self.vertical_distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = \
                    self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, 2] - point[2]
                abs_dcourse = abs(dcourse)
                if abs(dcourse) > 180:
                    abs_dcourse = 360 - abs(dcourse)
                if abs_dcourse >= 90:
                    self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-2, 0, 0, 0]
                else:
                    self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-1, 0, 0, 0]
            else:

                self.available_wps[self.flight_index[flightKey]['idx']] = [0, 0, 0, 0]
                number_of_next = fplan.shape[0]-idx
                saved = 0
                skipped = 0
                for i in range(number_of_next):
                    self.next_wps[self.flight_index[int(flightKey)]['idx'], i-skipped, :] = \
                        fplan[['x', 'y', 'altitude']].iloc[idx+i]
                    self.wp_rank[self.flight_index[flightKey]['idx'], i-skipped] = \
                        flight_plan_utils.compute_rank(self.next_wps[
                                                       self.flight_index[int(flightKey)]['idx'], i-skipped, :2
                                                                    ] -
                                                       p_start, p_end - p_start)
                    bearing_to_next, _ = \
                        utils.mybearing(self.flight_arr[self.flight_index[flightKey]['idx'], 3:5].astype(np.float64),
                                        self.next_wps[
                                        self.flight_index[int(flightKey)]['idx'], i-skipped, :2
                                                     ].astype(np.float64))
                    dcourse = bearing_to_next - self.flight_arr[self.flight_index[int(flightKey)]['idx'], 8]
                    abs_dcourse = abs(dcourse)
                    if abs_dcourse > 180:
                        abs_dcourse = 360 - abs(dcourse)
                    if abs_dcourse >= 90 and i < fplan[['x', 'y', 'altitude']].shape[0]:
                        skipped += 1
                        continue
                    self.dcourse_to_next_wps[self.flight_index[flightKey]['idx'], i-skipped] = dcourse
                    self.distance_to_next_wps[self.flight_index[flightKey]['idx'], i-skipped] = \
                        np.sqrt(np.sum((self.next_wps[self.flight_index[int(flightKey)]['idx'],
                                        i-skipped, :2]-point[:2])**2,
                                       axis=-1))
                    self.vertical_distance_to_next_wps[self.flight_index[flightKey]['idx'], i-skipped] = \
                        self.next_wps[self.flight_index[int(flightKey)]['idx'], i-skipped, 2] - point[2]
                    saved += 1
                    self.available_wps[self.flight_index[flightKey]['idx'], i-skipped] = 1
                    self.direct_to_idxs[self.flight_index[flightKey]['idx'], i-skipped] = idx+i
                    if saved >= 4:
                        break
                if saved == 0:
                    self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :] = \
                        self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6, 3])]
                    dcourse = - self.flight_arr[self.flight_index[flightKey]['idx'], 11]

                    self.available_wps[self.flight_index[flightKey]['idx']] = [1, 0, 0, 0]
                    self.wp_rank[self.flight_index[flightKey]['idx'], 0] = flight_plan_utils.compute_rank(
                        self.exit_wps[self.flight_index[flightKey]['idx'], np.array([5, 6])]-p_start, p_end-p_start)
                    self.dcourse_to_next_wps[self.flight_index[flightKey]['idx'], 0] = dcourse
                    self.distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = \
                        np.sqrt(np.sum((self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, :2] -
                                        point[:2]) ** 2,
                                       axis=-1))
                    self.vertical_distance_to_next_wps[self.flight_index[flightKey]['idx'], 0] = \
                        self.next_wps[self.flight_index[int(flightKey)]['idx'], 0, 2] - point[2]
                    abs_dcourse = abs(dcourse)
                    if abs(dcourse) > 180:
                        abs_dcourse = 360 - abs(dcourse)
                    if abs_dcourse >= 90:
                        self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-2, 0, 0, 0]
                    else:
                        self.direct_to_idxs[self.flight_index[flightKey]['idx']] = [-1, 0, 0, 0]

    def correct_FLchange_finished(self):
        mask = \
            self.finished_FL_change & (self.flight_arr[:, 5] * self.FL_change_sign > self.next_FL * self.FL_change_sign)
        idxs = np.where(mask)[0]

        for idx in idxs:
            vspeed = self.change_FL_vspeed[idx]
            alt = self.flight_arr[idx, 5]
            f_alt = self.next_FL[idx]
            t = np.abs(f_alt-alt)/vspeed
            current_xyspeed = self.flight_arr[idx, 6:8]
            pointxy = self.flight_arr[idx, 3:5] - t*current_xyspeed
            if pointxy[0] is np.nan:
                print(t)
                print(vspeed)
                assert False
            self.flight_arr[idx, 9] = np.sqrt(np.sum(self.change_FL_hspeed[idx] ** 2))
            final_point = pointxy + t*self.change_FL_hspeed[idx]
            self.flight_arr[idx, 3:5] = final_point
            self.flight_arr[idx, 6:8] = self.change_FL_hspeed[idx]
            self.flight_arr[idx, 5] = self.next_FL[idx]

        if np.any(mask):
            self.vertical_distance_to_next_wps[mask] = \
                (self.next_wps[mask, :, 2] - self.flight_arr[mask, 5][:, np.newaxis])*\
                (np.where(self.available_wps[mask] == 1, self.available_wps[mask], np.nan))

    def correct_resume_to_fplan(self):
        mask = self.finished_resume_to_fplan & \
               (self.flight_arr[:, 5] * self.FL_change_sign > self.next_FL * self.FL_change_sign)
        self.flight_arr[mask, 5] = self.next_FL[mask]
        if np.any(mask):
            self.vertical_distance_to_next_wps[mask] = \
                (self.next_wps[mask, :, 2] - self.flight_arr[mask, 5][:, np.newaxis])*\
                (np.where(self.available_wps[mask] == 1, self.available_wps[mask], np.nan))

        return
        # COMMENT: code for using horizontal speed before change of altitude
        idxs = np.where(mask)[0]
        for idx in idxs:
            vspeed = self.change_FL_vspeed[idx]
            alt = self.flight_arr[idx, 5]
            f_alt = self.next_FL[idx]
            t = np.abs(f_alt - alt) / vspeed
            current_xyspeed = self.flight_arr[idx, 6:8]
            pointxy = self.flight_arr[idx, 3:5] - t * current_xyspeed
            if pointxy[0] is np.nan:
                print(t)
                print(vspeed)
                assert False
            self.flight_arr[idx, 9] = np.sqrt(np.sum(self.change_FL_hspeed[idx] ** 2))

            self.flight_arr[idx, 6] = (np.sin(np.radians(self.flight_arr[idx, 8]))) * self.flight_arr[idx, 9]
            self.flight_arr[idx, 7] = (np.cos(np.radians(self.flight_arr[idx, 8]))) * self.flight_arr[idx, 9]

            final_point = pointxy + t * self.flight_arr[idx, 6:8]
            self.flight_arr[idx, 3:5] = final_point
            self.flight_arr[idx, 5] = self.next_FL[idx]

    def correct_direct_to_finished(self):

        mask = \
            self.finished_direct_to & (self.flight_arr[:, 5] * self.FL_change_sign > self.next_FL * self.FL_change_sign)
        self.flight_arr[mask, 5] = self.next_FL[mask]

        if np.any(mask):
            self.vertical_distance_to_next_wps[mask] = \
                (self.next_wps[mask, :, 2] - self.flight_arr[mask, 5][:, np.newaxis]) * \
                (np.where(self.available_wps[mask] == 1, self.available_wps[mask], np.nan))
        return
        #COMMENT: code for using horizontal speed before change of altitude
        idxs = np.where(mask)[0]
        for idx in idxs:
            vspeed = self.change_FL_vspeed[idx]
            alt = self.flight_arr[idx, 5]
            f_alt = self.next_FL[idx]
            t = np.abs(f_alt-alt)/vspeed
            current_xyspeed = self.flight_arr[idx, 6:8]
            pointxy = self.flight_arr[idx, 3:5] - t*current_xyspeed
            if pointxy[0] is np.nan:
                print(t)
                print(vspeed)
                assert False
            self.flight_arr[idx, 9] = np.sqrt(np.sum(self.change_FL_hspeed[idx] ** 2))

            self.flight_arr[idx, 6] = (np.sin(np.radians(self.flight_arr[idx, 8]))) * self.flight_arr[idx, 9]
            self.flight_arr[idx, 7] = (np.cos(np.radians(self.flight_arr[idx, 8]))) * self.flight_arr[idx, 9]

            final_point = pointxy + t*self.flight_arr[idx, 6:8]
            self.flight_arr[idx, 3:5] = final_point
            self.flight_arr[idx, 5] = self.next_FL[idx]

    def step(self, unclipped_actions, res_act_ID):
        """
        Applies actions and returns s' and rewards.
        Action examples:
           continue_action: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
           FL_change: [0, 0, -17, 0, 0, 0, 0, 0, 0, 0, 0]
           direct to 4th wp: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
           resume to flight plan: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        :param actions: N x [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                             direct_to_3rd, direct_to_4th, continue_action, duration, resume_to_flight_plan] matrix where N is the number of agents.

        :return: Next state s', edges between agents and rewards, available_wps: N X 4 [1,1,1,0],
        flight phases:['climbing'/'cruising'/'descending'].
        """
        self.projection_dict = {}
        unclipped_actions = unclipped_actions.astype(np.float64)
        unclipped_actions[~self.active_flights_mask] = [0]*11
        # COMMENT: Handling direct to waypoint. START
        to_next_wp_action = unclipped_actions[:, 3]
        to_next_wp_action_mask = (to_next_wp_action == 1) & (self.active_flights_mask)
        direct_to_wp_mask = unclipped_actions[:, np.array([3, 5, 6, 7])].astype(bool)
        course_change_mask = (unclipped_actions[:, 0] != 0)
        resume_to_flight_plan_mask = unclipped_actions[:, 10].astype(bool)
        resume_to_flight_plan_4D_mask = np.array([[False, False, False, False]]*self.flight_arr.shape[0])
        resume_to_flight_plan_4D_mask[resume_to_flight_plan_mask] = [True, False, False, False]
        self.executing_course_change = self.executing_course_change | course_change_mask
        self.action_duration[course_change_mask] = unclipped_actions[course_change_mask, 9]
        self.action_starting_time_point[course_change_mask] = self.timestamp

        continue_action = unclipped_actions[:, 8]
        if np.any((direct_to_wp_mask[:, 1:] & ~self.available_wps.astype(bool)[:, 1:]) &
                  (self.active_flights_mask[:, np.newaxis])):
            print('Waypoint not available')
            assert False
        direct_to_all_mask_1d = (self.active_flights_mask) & np.any(direct_to_wp_mask, axis=1)

        self.towards_wp_idxs[direct_to_all_mask_1d] = self.direct_to_idxs[direct_to_wp_mask]
        self.towards_wp_fp_idx[direct_to_all_mask_1d] = self.direct_to_fp_idx[direct_to_all_mask_1d]
        self.executing_direct_to = (self.executing_direct_to & continue_action.astype(bool)) | \
                                   np.any(direct_to_wp_mask, axis=1)
        self.executing_resume_to_fplan = (self.executing_resume_to_fplan & continue_action.astype(bool)) | \
                                         resume_to_flight_plan_mask

        self.towards_wps[direct_to_all_mask_1d | resume_to_flight_plan_mask] = self.next_wps[direct_to_wp_mask |
                                                                                             resume_to_flight_plan_4D_mask]
        self.towards_wp_rank[np.any(direct_to_wp_mask, axis=1) | resume_to_flight_plan_mask] = \
            self.wp_rank[direct_to_wp_mask | resume_to_flight_plan_4D_mask]

        direct_to_wp_mask_padded = \
            np.array([m if np.any(m) else [True, False, False, False] for m in direct_to_wp_mask])
        next_wps = self.next_wps[direct_to_wp_mask_padded]
        step_from_historical_action_mask = (unclipped_actions[:, 4] == 1) & self.active_historical_mask

        distance_from_next_wp = np.sqrt(np.sum((self.flight_arr[:, 3:5].astype(np.float64) -
                                                next_wps[:, :2].astype(np.float64))**2, axis=1))
        speed_magnitude = np.copy(self.flight_arr[:, 9])
        speed_magnitude[speed_magnitude == 0] = 1
        t_to_next_wp = distance_from_next_wp/speed_magnitude
        v_dist_to_exit_point = next_wps[:, 2] - self.flight_arr[:, 5]
        t_to_next_wp_0_mask = (t_to_next_wp == 0)

        temp_mask = (self.active_flights_mask) & (resume_to_flight_plan_mask) & ~t_to_next_wp_0_mask
        temp_mask_2 = (self.active_flights_mask) & (direct_to_all_mask_1d | resume_to_flight_plan_mask) & t_to_next_wp_0_mask

        try:
            unclipped_actions[temp_mask, 2] = \
                v_dist_to_exit_point[temp_mask] / \
                t_to_next_wp[temp_mask] - \
                self.flight_arr[temp_mask, 10]
        except:
            print(temp_mask)
            print(self.flight_arr[temp_mask, 10])
            print(t_to_next_wp[temp_mask])
            print(speed_magnitude[temp_mask])
            print(distance_from_next_wp[temp_mask])
            print(self.flight_arr[temp_mask, 3:5])
            print(next_wps[temp_mask, :2])
            assert False

        self.next_FL[temp_mask] = next_wps[temp_mask, 2]
        self.FL_change_sign[temp_mask] = \
            np.sign(unclipped_actions[temp_mask, 2]+self.flight_arr[temp_mask, 10])
        self.change_FL_vspeed[temp_mask] = unclipped_actions[temp_mask, 2]+self.flight_arr[temp_mask, 10]
        self.change_FL_hspeed[temp_mask] = self.flight_arr[temp_mask, 6:8]

        v_dist_not_0 = self.flight_arr[(v_dist_to_exit_point != 0) & (temp_mask)]

        unclipped_actions[temp_mask_2, 2] = 0
        unclipped_actions[temp_mask_2, 0] = 0

        direct_to_or_resume_mask = direct_to_wp_mask | resume_to_flight_plan_4D_mask

        unclipped_actions[direct_to_all_mask_1d | resume_to_flight_plan_mask, 0] = \
            np.where(abs(self.dcourse_to_next_wps[direct_to_or_resume_mask]) < 90,
                     self.dcourse_to_next_wps[direct_to_or_resume_mask], 0)

        rows = np.where(continue_action.astype(bool))[0]
        for r in rows:
            unclipped_actions[r, np.array([0, 1, 2])] = 0
        # COMMENT: Handling direct to waypoint. END

        # COMMENT: Handling FL change. START
        FLchange_mask = (unclipped_actions[:, 2] != 0) & self.active_flights_mask & ~self.executing_resume_to_fplan

        self.FL_change_sign[FLchange_mask] = np.sign(unclipped_actions[FLchange_mask, 2])
        self.next_FL[FLchange_mask] = \
            ((self.flight_arr[FLchange_mask, 5] + self.FL_change_sign[FLchange_mask] * 1000)/1000).astype(np.int64)*1000
        self.executing_FL_change = (self.executing_FL_change & continue_action.astype(bool)) | FLchange_mask
        self.change_FL_vspeed[FLchange_mask] = unclipped_actions[FLchange_mask, 2]
        self.change_FL_hspeed[FLchange_mask] = self.flight_arr[FLchange_mask, 6:8]
        unclipped_actions[FLchange_mask, 2] = unclipped_actions[FLchange_mask, 2] - self.flight_arr[FLchange_mask, 10]
        # COMMENT: Handling FL change. END

        actions = self.clip_actions(unclipped_actions[:, :3])

        active_actions = self.active_flights_mask.astype(int)[:, np.newaxis]*actions

        if self.flight_arr[self.active_flights_mask].shape[0] > 0:
            step_arr = np.array([[self.timestamp]] * self.flight_arr[self.active_flights_mask].shape[0])
            log_np = np.concatenate([self.flight_arr[self.active_flights_mask],
                                     -self.dcourse_to_next_wps[self.active_flights_mask],
                                     self.distance_to_next_wps[self.active_flights_mask],
                                     self.vertical_distance_to_next_wps[self.active_flights_mask],
                                     step_arr], axis=1)
            self.log_states.append(log_np)
            self.log_flight_phases.append(np.array(self.flight_phases)[self.active_flights_mask])
            self.wp_per_tstamp.extend(np.concatenate([self.exit_wps[self.active_flights_mask],
                                                      log_np[:, np.array([26])]],
                                                     axis=1))

        self.move_flights(active_actions, ~step_from_historical_action_mask)
        self.find_next_from_historical_data(step_from_historical_action_mask)

        self.timestamp += self.dt
        changed_sector_mask = self.new_active_flights()
        self.update_fplans()
        self.find_current_exit_p_w_fplans((self.new_active_flights_mask | self.flight_plan_updated_mask |
                                           changed_sector_mask), self.flight_sectors)

        # COMMENT: handling finished actions w duration. START
        self.find_next_wp(self.active_flights_mask)
        self.finished_direct_to = (self.point_rank >= self.towards_wp_rank) & self.executing_direct_to
        self.finished_resume_to_fplan = (self.point_rank >= self.towards_wp_rank) & self.executing_resume_to_fplan
        self.executing_direct_to = self.executing_direct_to & ~self.finished_direct_to
        self.executing_resume_to_fplan = self.executing_resume_to_fplan & ~self.finished_resume_to_fplan
        self.finished_FL_change = \
            self.executing_FL_change & (self.flight_arr[:, 5] * self.FL_change_sign >= self.next_FL *
                                        self.FL_change_sign)
        self.executing_FL_change = self.executing_FL_change & ~self.finished_FL_change
        self.flight_arr[self.finished_FL_change | self.finished_resume_to_fplan, 10] = 0
        # COMMENT: handling finished actions w duration. END

        self.correct_FLchange_finished()
        self.correct_resume_to_fplan()

        self.finished_course_change_mask = (self.action_starting_time_point + self.action_duration <= self.timestamp) & self.executing_course_change
        self.executing_course_change = self.executing_course_change & ~self.finished_course_change_mask

        done = False
        self.update_features_wrt_exit_point()
        self.update_lon_lat_from_x_y_vec()

        edges = self.edges_between_active_flights(res_act_ID)

        reward, reward_per_factor = self.reward_fun(actions, edges)
        edges_w_los, edges_w_conflict, edges_w_conf_not_alerts, edges_w_alerts = self.conflicting_flights(edges)
        edges = np.append(edges, np.array([edges_w_los, edges_w_conflict,
                                           edges_w_conf_not_alerts,
                                           edges_w_alerts]).T.astype(float), axis=1)
        edges = np.delete(edges, [11, 13], axis=1)
        if edges[edges[:, -1] == 1].shape[0] > 0:
            self.alerts_debug += 1

        if self.timestamp > self.timestamp_end or (not np.any(self.active_flights_mask) and np.all(self.activated_mask)):
            self.total_losses_of_separation_per_episode.append(self.total_losses_of_separation)
            self.total_alerts_per_episode.append(self.total_alerts)
            self.write_loss_conf_main()
            done = True

        self.flight_phases = self.update_flight_phases()
        concated_states = \
            np.concatenate([self.flight_arr, -self.dcourse_to_next_wps,
                            self.distance_to_next_wps, self.vertical_distance_to_next_wps],
                           axis=1)

        return concated_states, \
               edges, \
               reward, \
               reward_per_factor, \
               done, \
               actions, \
               self.available_wps, \
               self.flight_phases,\
               self.finished_FL_change, \
               self.finished_direct_to, \
               self.finished_resume_to_fplan, \
               self.executing_FL_change,\
               self.executing_direct_to, \
               self.executing_resume_to_fplan

    def check_conformance(self, prev_flight_arr, next_states, unclipped_actions, res_act_ID):
        """self.flight_arr : ['flightKey', 'longitude', 'latitude',
                              'x', 'y', 'altitude',
                              'speed_x_component', 'speed_y_component',
                              'course', 'speed', 'alt_speed','bearing_relative_to_exit_wp',
                              'vertical_distance_to_exit_point','horizontal_distance_to_exit_point']"""
        labels = ['course', 'horizontal speed', 'vertical speed']
        for next_state in next_states:
            idx = self.flight_index[next_state[0]]['idx']
            if not self.active_flights_mask[idx]:
                continue
            prev_state = prev_flight_arr[idx]
            course_speed_diff = next_state[8:11] - prev_state[8:11]

            for i in range(course_speed_diff.shape[0]):
                if (unclipped_actions[idx, i] == 0 and np.abs(course_speed_diff[i]) > 1e-4) or \
                        (unclipped_actions[idx, i] != 0 and np.abs(course_speed_diff[i]) < 1e-12):

                    id = str(int(self.timestamp)) + '_' + str(int(next_state[0]))

                    with open(env_config['log_dir'] + '/non_conformance_events.csv', 'a') as f:
                        print(unclipped_actions[idx, i])
                        print(course_speed_diff[i])
                        print(prev_state[8+i])
                        print(next_state[8+i])
                        f.write(id+','+res_act_ID[idx]+','+labels[i]+',' +
                                str(prev_state[8+i]+unclipped_actions[idx, i])+','+str(next_state[8+i])+'\n')

                elif not (unclipped_actions[idx, i] == 0 and np.abs(course_speed_diff[i]) < 1e-4):
                    if np.sign(unclipped_actions[idx, i]) != np.sign(course_speed_diff[i]):
                        id = str(int(self.timestamp)) + '_' + str(int(next_state[0]))
                        print(unclipped_actions[idx, i])
                        print(np.sign(unclipped_actions[idx, i]))
                        print(np.sign(course_speed_diff[i]))
                        print(course_speed_diff[i])
                        exit(0)
                        with open(env_config['log_dir'] + '/non_conformance_events.csv', 'a') as f:
                            f.write(id + ',' + res_act_ID[idx] + ',' + labels[i] + ',' +
                                    str(prev_state[8 + i]+unclipped_actions[idx, i]) + ',' + str(next_state[8 + i]) + '\n')

    def copy_move_flights_for_conformance(self, active_actions, step_from_historical_action_mask):

        flights_cp = np.copy(self.flight_arr)
        finished_direct_to_cp = np.copy(self.finished_direct_to)
        finished_FLchange_cp = np.copy(self.finished_FL_change)

        self.move_flights(active_actions, ~step_from_historical_action_mask)
        self.find_next_from_historical_data(step_from_historical_action_mask)

        # COMMENT: handling finished actions w duration. START
        point_rank_arr = self.compute_point_ranks(self.active_flights_mask)
        self.finished_direct_to = (point_rank_arr >= self.towards_wp_rank) & self.executing_direct_to
        self.finished_FL_change = \
            self.executing_FL_change & (self.flight_arr[:, 5] *
                                        self.FL_change_sign >= self.next_FL * self.FL_change_sign)
        self.flight_arr[self.finished_FL_change | self.finished_direct_to, 10] = 0
        # COMMENT: handling finished actions w duration. END

        self.correct_FLchange_finished()
        self.correct_direct_to_finished()

        conformance_flight_arr = np.copy(self.flight_arr)
        self.flight_arr = flights_cp
        self.finished_FL_change = finished_FLchange_cp
        self.finished_direct_to = finished_direct_to_cp

        return conformance_flight_arr

    def get_wp_by_index(self, RTkey, idx_start, idx_end):
        fplan = self.flight_plans[RTkey]['fplans'][self.flight_plans[RTkey]['current']][1]
        wps = fplan[['x', 'y']].iloc[idx_start: idx_end+1].values

        return wps

    def compute_additional_nautical_miles_course_change(self, RTkey, action):
        """
        :param RTkey: RTkey of flight
        :param action: 1D np array [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                                    direct_to_3rd, direct_to_4th, continue_action, duration]
        :return: additional nautical miles
        """
        flight_idx = self.flight_index[RTkey]['idx']
        course = self.flight_arr[flight_idx, 8]
        position = self.flight_arr[flight_idx, 3:5]
        speed = self.flight_arr[flight_idx, 9]
        new_course = course + action[0]
        duration = action[9]
        x_speed_component = (np.sin(np.radians(new_course))) * speed
        y_speed_component = (np.cos(np.radians(new_course))) * speed
        r_position = position+np.array([x_speed_component, y_speed_component])*duration
        wp_r, wp_idx_r = self.next_wp_per_flight(flight_idx, r_position, new_course)
        wp, wp_idx = self.next_wp_per_flight(flight_idx, position, course)

        if wp_r is None or wp is None: # wp_idx_r == -1:
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
            #       'ALERT: Return to flight plan wp not found\n'
            #       '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

            return 0, 0

        wp_r = wp_r[:2]
        wp = wp[:2]
        manoeuvre_nm = np.sqrt(np.sum((r_position-position)**2)) + np.sqrt(np.sum((wp_r-r_position)**2))
        if wp_idx == -1 or wp_idx_r == -1:
            intermediate_wp = [wp_r]
        else:
            intermediate_wp = self.get_wp_by_index(RTkey, wp_idx, wp_idx_r)
        p = position
        fp_distance = 0
        for i_wp in intermediate_wp:
            fp_distance += np.sqrt(np.sum((i_wp-p)**2))
            p = wp

        additional_nm = (manoeuvre_nm-fp_distance)/1852

        additional_duration = additional_nm/speed

        return additional_nm, additional_duration

    def compute_additional_nautical_miles_direct_to(self, RTkey, action):
        """
        :param RTkey: RTkey of flight
        :param action: 1D np array [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                                    direct_to_3rd, direct_to_4th, continue_action, duration]
        :return: additional nautical miles
        """
        direct_to_wp_mask = action[np.array([3, 5, 6, 7])]
        wp_idxs = np.where(direct_to_wp_mask == 1)[0]
        if wp_idxs.shape[0] != 1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
                  'ALERT: Invalid direct to action\n'
                  '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        wp_idx = wp_idxs[0]
        flight_idx = self.flight_index[RTkey]['idx']
        flight_p = self.flight_arr[self.flight_index[RTkey]['idx'], 3:5]
        p = flight_p
        fp_distance = 0

        sorted_idx = np.argsort(self.wp_rank[flight_idx, :wp_idx + 1])
        sorted_next_wps = self.next_wps[flight_idx, :wp_idx+1, :2][sorted_idx]

        for wp in sorted_next_wps:
            fp_distance += np.sqrt(np.sum((wp-p)**2))
            p = wp
            if np.allclose(wp.astype(np.float32), self.next_wps[flight_idx, wp_idx, :2].astype(np.float32)):
                break

        direct_to_distance = np.sqrt(np.sum((self.next_wps[flight_idx, wp_idx, :2]-flight_p)**2))
        additional_nm = (direct_to_distance-fp_distance)/1852

        speed = self.flight_arr[flight_idx, 9]

        additional_duration = additional_nm/speed

        return additional_nm, additional_duration

    def compute_additional_duration_FL_change(self, RTkey, action):
        return 0

    def compute_additional_duration_speed_change(self, RTkey, action):
        """
        :param RTkey: RTkey of flight
        :param action: 1D np array [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                                    direct_to_3rd, direct_to_4th, continue_action, duration]
        :return: additional duration
        """
        action_duration = action[9]
        d_speed = action[1]
        flight_idx = self.flight_index[RTkey]['idx']
        speed = self.flight_arr[flight_idx, 9]
        dist1 = speed*action_duration
        action_speed = speed+d_speed

        duration2 = dist1/action_speed

        return duration2 - action_duration

    def validation_exercise_step(self, unclipped_actions, res_act_ID):
        """
        Applies actions and returns s' and rewards.
        Action examples:
           continue_action: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
           FL_change: [0, 0, -17, 0, 0, 0, 0, 0, 0, 0]
           direct to 4th wp: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        :param unclipped_actions: N x [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                                        direct_to_3rd, direct_to_4th, continue_action, duration] matrix where N is the number of agents.

        :return: Next state s', edges between agents and rewards, available_wps: N X 4 [1,1,1,0],
        flight phases:['climbing'/'cruising'/'descending'].
        """
        self.projection_dict = {}

        # COMMENT: Handling direct to waypoint. START
        to_next_wp_action = unclipped_actions[:, 3]
        to_next_wp_action_mask = (to_next_wp_action == 1) & (self.active_flights_mask)
        direct_to_wp_mask = unclipped_actions[:, np.array([3, 5, 6, 7])].astype(bool)
        continue_action = unclipped_actions[:, 8]
        if np.any((direct_to_wp_mask & ~self.available_wps.astype(bool)) &
                  (self.active_flights_mask[:, np.newaxis])):
            print('Waypoint not available')
            assert False
        self.executing_direct_to = \
            (self.executing_direct_to & continue_action.astype(bool)) | np.any(direct_to_wp_mask, axis=1)

        self.towards_wps = self.next_wps[direct_to_wp_mask]
        self.towards_wp_rank[np.any(direct_to_wp_mask, axis=1)] = self.wp_rank[direct_to_wp_mask]
        direct_to_wp_mask_padded = \
            np.array([m if np.any(m) else [True, False, False, False] for m in direct_to_wp_mask])
        next_wps = self.next_wps[direct_to_wp_mask_padded]
        step_from_historical_action_mask = (unclipped_actions[:, 4] == 1) & self.active_historical_mask
        distance_from_next_wp = np.sqrt(np.sum((self.flight_arr[:, 3:5].astype(np.float64) -
                                                next_wps[:, :2].astype(np.float64)) ** 2, axis=1))
        speed_magnitude = np.copy(self.flight_arr[:, 9])
        speed_magnitude[speed_magnitude == 0] = 1
        t_to_next_wp = distance_from_next_wp / speed_magnitude
        v_dist_to_exit_point = next_wps[:, 2] - self.flight_arr[:, 5]
        t_to_next_wp_0_mask = (t_to_next_wp == 0)

        direct_to_all_mask_1d = (self.active_flights_mask) & np.any(direct_to_wp_mask, axis=1)

        temp_mask = (self.active_flights_mask) & (direct_to_all_mask_1d) & ~t_to_next_wp_0_mask
        temp_mask_2 = (self.active_flights_mask) & (direct_to_all_mask_1d) & t_to_next_wp_0_mask

        try:
            unclipped_actions[temp_mask, 2] = \
                v_dist_to_exit_point[temp_mask] / \
                t_to_next_wp[temp_mask] - \
                self.flight_arr[temp_mask, 10]
        except:
            print(temp_mask)
            print(self.flight_arr[temp_mask, 10])
            print(t_to_next_wp[temp_mask])
            print(speed_magnitude[temp_mask])
            print(distance_from_next_wp[temp_mask])
            print(self.flight_arr[temp_mask, 3:5])
            print(next_wps[temp_mask, :2])
            assert False

        self.next_FL[temp_mask] = next_wps[temp_mask, 2]
        self.FL_change_sign[temp_mask] = np.sign(unclipped_actions[temp_mask, 2])
        self.change_FL_vspeed[temp_mask] = unclipped_actions[temp_mask, 2]
        self.change_FL_hspeed[temp_mask] = self.flight_arr[temp_mask, 6:8]

        v_dist_not_0 = self.flight_arr[(v_dist_to_exit_point != 0) & (temp_mask)]

        unclipped_actions[temp_mask_2, 2] = 0
        unclipped_actions[temp_mask_2, 0] = 0

        unclipped_actions[direct_to_all_mask_1d, 0] = \
            np.where(abs(self.dcourse_to_next_wps[direct_to_wp_mask]) < 90,
                     self.dcourse_to_next_wps[direct_to_wp_mask], 0)

        rows = np.where(continue_action.astype(bool))[0]
        for r in rows:
            unclipped_actions[r, np.array([0, 1, 2])] = 0
        # COMMENT: Handling direct to waypoint. END

        # COMMENT: Handling FL change. START
        FLchange_mask = (unclipped_actions[:, 2] != 0) & self.active_flights_mask & ~self.executing_direct_to

        self.FL_change_sign[FLchange_mask] = np.sign(unclipped_actions[FLchange_mask, 2])
        self.next_FL[FLchange_mask] = ((self.flight_arr[FLchange_mask, 5] +
                                        self.FL_change_sign[FLchange_mask] * 1000) / 1000).astype(np.int64) * 1000
        self.executing_FL_change = (self.executing_FL_change & continue_action.astype(bool)) | FLchange_mask
        self.change_FL_vspeed[FLchange_mask] = unclipped_actions[FLchange_mask, 2]
        self.change_FL_hspeed[FLchange_mask] = self.flight_arr[FLchange_mask, 6:8]
        # COMMENT: Handling FL change. END

        actions = self.clip_actions(unclipped_actions[:, :3])

        active_actions = self.active_flights_mask.astype(int)[:, np.newaxis] * actions

        if self.flight_arr[self.active_flights_mask].shape[0] > 0:

            step_arr = np.array([[self.timestamp]] * self.flight_arr[self.active_flights_mask].shape[0])
            log_np = np.concatenate([self.flight_arr[self.active_flights_mask], step_arr], axis=1)
            self.log_states.append(log_np)
            self.log_flight_phases.append(np.array(self.flight_phases)[self.active_flights_mask])
            self.wp_per_tstamp.extend(np.concatenate([self.exit_wps[self.active_flights_mask],
                                                      log_np[:, np.array([14])]],
                                                     axis=1))

        prev_flight_arr = np.copy(self.flight_arr)

        conformance_flight_arr = self.copy_move_flights_for_conformance(active_actions, step_from_historical_action_mask)

        unclipped_actions[:, 2] += (conformance_flight_arr[:, 10] - prev_flight_arr[:, 10])

        self.timestamp += self.dt

        """COMMUNICATION W PLATFORM
        read next state
        """
        try:
            next_states = \
                self.platform_states_dict[self.timestamp][['RTkey', 'longitude', 'latitude', 'x', 'y', 'altitude',
                                                           'speed_x_component', 'speed_y_component', 'course', 'speed',
                                                           'alt_speed', 'relative_bearing_to_exit_point',
                                                           'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point']].values
        except KeyError:
            next_states = np.empty((0, 14))

        self.check_conformance(prev_flight_arr, next_states, unclipped_actions, res_act_ID)

        self.update_fplans()
        self.new_active_flights_ve(next_states)

        # COMMENT: handling finished actions w duration. START
        self.find_next_wp(self.active_flights_mask)
        self.finished_direct_to = (self.point_rank >= self.towards_wp_rank) & self.executing_direct_to
        self.executing_direct_to = self.executing_direct_to & ~self.finished_direct_to
        self.finished_FL_change = self.executing_FL_change & \
                                  (self.flight_arr[:, 5] * self.FL_change_sign >= self.next_FL * self.FL_change_sign)
        self.executing_FL_change = self.executing_FL_change & ~self.finished_FL_change
        self.flight_arr[self.finished_FL_change | self.finished_direct_to, 10] = 0
        self.correct_FLchange_finished()
        self.correct_direct_to_finished()
        # COMMENT: handling finished actions w duration. END

        self.find_exit_wp_w_fplans((self.new_active_flights_mask | self.flight_plan_updated_mask))
        done = False
        self.update_features_wrt_exit_point()
        self.update_lon_lat_from_x_y_vec()

        edges = self.edges_between_active_flights(res_act_ID)

        reward, reward_per_factor = self.reward_fun(actions, edges)
        edges_w_los, edges_w_conflict, edges_w_conf_not_alerts, edges_w_alerts = self.conflicting_flights(edges)
        edges = np.append(edges, np.array([edges_w_los, edges_w_conflict,
                                           edges_w_conf_not_alerts, edges_w_alerts]).T.astype(float), axis=1)

        edges = np.delete(edges, [11, 13], axis=1)
        if edges[edges[:, -1] == 1].shape[0] > 0:
            self.alerts_debug += 1

        if self.timestamp > self.timestamp_end or (not np.any(self.active_flights_mask) and np.all(self.activated_mask)):
            self.total_losses_of_separation_per_episode.append(self.total_losses_of_separation)
            self.total_alerts_per_episode.append(self.total_alerts)
            self.write_loss_conf_main()

            done = True

        self.flight_phases = self.update_flight_phases()

        return self.flight_arr, \
               edges, \
               reward, \
               reward_per_factor, \
               done, \
               actions, \
               self.available_wps, \
               self.flight_phases, \
               self.finished_FL_change, \
               self.finished_direct_to, \
               self.executing_FL_change, \
               self.executing_direct_to

    def check_neighbouring_time_interval(self):
        """
        Debug helper function: Checks if neighbours are in the R U DR time interval
        R: time interval in which the ownship crosses the current sector
        DR: time interval in which the ownchip crosses the Downstream sector
        :return:
        """
        for file in glob.glob(self.training_path + '/*.rdr'):

            neighbour_flights_file = file.split('/')[-1].split('.')[0]

            own_df = pd.read_csv(file, sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                                 'Lon': 'longitude', 'Alt': 'altitude',
                                                                 'Unixtime': 'timestamp'})
            own_df = own_df.sort_values('timestamp')
            neighbour_flights = pd.read_csv(self.training_path+'/'+neighbour_flights_file,
                                            sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                                      'Lon': 'longitude', 'Alt': 'altitude',
                                                                      'Unixtime': 'timestamp'})

            flightKeys_in_interval = \
                neighbour_flights[(neighbour_flights['timestamp'] >= own_df['timestamp'].values[0]) |
                                  (neighbour_flights['timestamp'] <= own_df['timestamp'].values[-1])]['flightKey'].unique()

            not_in_interval_df = (neighbour_flights[~neighbour_flights['flightKey'].isin(flightKeys_in_interval)])
            if not_in_interval_df.shape[0] > 0:
                print(not_in_interval_df)
                exit(0)

    def print_active_trajectories(self):
        f = self.training_path + '/'+self.scenario
        neighbour_flights_file = f.split('/')[-1].split('.')[0]
        neighbour_flights = pd.read_csv(self.training_path + '/' + neighbour_flights_file,
                                        sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                                  'Lon': 'longitude', 'Alt': 'altitude',
                                                                  'Unixtime': 'timestamp'})

        active_flights_idxs = np.where(self.active_flights_mask == True)[0]
        active_flights = self.flight_arr[active_flights_idxs]
        active_flight_ids = list(map(int, active_flights[:, 0]))
        for name, group in neighbour_flights[neighbour_flights['flightKey'].isin(active_flight_ids)].groupby('flightKey'):
            group.to_csv(self.debug_folder+'/'+str(name)+'.csv', index=False)
        own_df = pd.read_csv(f, sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                             'Lon': 'longitude', 'Alt': 'altitude',
                                                             'Unixtime': 'timestamp'})
        own_df.to_csv(self.debug_folder+'/own.csv', index=False)

    def read_flights_training_set(self):

        self.last_points = {}
        for file in glob.glob(self.training_path+'/'+self.scenario):

            neighbour_flights_file = file.split('/')[-1].split('.')[0]
            neighbour_last_points_file = neighbour_flights_file.split('/')[-1].split('_interpolated')[0]
            own_complete_df = pd.read_csv(file, sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                                          'Lon': 'longitude', 'Alt': 'altitude',
                                                                          'Unixtime': 'timestamp'})

            own_complete_df = own_complete_df[(own_complete_df['Sector'] != 'N/A') & (~own_complete_df['Sector'].isna())]
            own_complete_df = own_complete_df.sort_values('timestamp')

            self.current_sector = own_complete_df['Sector'].head(1).values[0]
            self.downstream_sector = own_complete_df['Sector'].tail(1).values[0]

            self.init_timestamp = own_complete_df['timestamp'].values[0]
            self.timestamp_end = own_complete_df['timestamp'].values[-1]

            neighbour_flights = pd.read_csv(self.training_path+'/'+neighbour_flights_file,
                                            sep='\t').rename(columns={'RTKey': 'flightKey', 'Lat': 'latitude',
                                                                      'Lon': 'longitude', 'Alt': 'altitude',
                                                                      'Unixtime': 'timestamp'})

            neighbour_flights = neighbour_flights[(neighbour_flights['Sector'] != 'N/A') &
                                                  (~neighbour_flights['Sector'].isna())]

            neighbour_flights = neighbour_flights[neighbour_flights['timestamp'] >= self.init_timestamp]
            last_points_neighbours_df = pd.read_csv(self.training_path+'/'+neighbour_last_points_file + '_final_points')

            flight_idx = 0
            for name, group in neighbour_flights.groupby(['flightKey']):
                complete_df = group[group['timestamp'] >= self.init_timestamp]

                if complete_df.shape[0] <= 1:
                    continue
                complete_df = flight_utils.transform_compute_speed_bearing(complete_df)
                last_row = complete_df.tail(1)
                flight_df = complete_df.iloc[[0, 1]]

                flight_df['timestamp'] = flight_df['timestamp'].astype(int)
                first_row = flight_df.iloc[0]
                entry_timestamp = first_row['timestamp']
                flightKey = int(first_row['flightKey'])
                if entry_timestamp not in self.flight_entry_index:
                    self.flight_entry_index[entry_timestamp] = set([flightKey])
                else:
                    self.flight_entry_index[entry_timestamp].add(flightKey)

                self.flight_index[int(flightKey)] = {'idx': flight_idx,
                                                     'df': flight_df,
                                                     'complete_df': complete_df,
                                                     'complete_df_idx': 0,
                                                     'last_point': last_row}
                self.last_points[int(flightKey)] = \
                    last_points_neighbours_df[last_points_neighbours_df['RTKey'] == int(flightKey)].values

                flight_idx += 1
                self.flight_plans = flight_plan_utils.read_flight_plans(flight_df, self.flight_plans)

            own_complete_df = flight_utils.transform_compute_speed_bearing(own_complete_df)
            own_last_row = own_complete_df.tail(1)
            own_df = own_complete_df.iloc[[0, 1]]
            own_first_row = own_complete_df.iloc[0]

            own_df = own_df.iloc[[0]]

            ownKey = int(own_first_row['flightKey'])
            self.last_points[ownKey] = \
                pd.read_csv(self.training_path+'/'+neighbour_last_points_file+'_final_points.rdr').values

            self.flight_plans = flight_plan_utils.read_flight_plans(own_df, self.flight_plans)

            self.flight_index[ownKey] = {'idx': flight_idx,
                                         'df': own_df,
                                         'complete_df': own_complete_df,
                                         'complete_df_idx': 0,
                                         'last_point': own_last_row}

            entry_timestamp = own_first_row['timestamp']

            if entry_timestamp not in self.flight_entry_index:
                self.flight_entry_index[entry_timestamp] = set([ownKey])
            else:
                self.flight_entry_index[entry_timestamp].add(ownKey)

    def read_flights_placeholder(self, init_timestamp):
        self.init_timestamp = 1498765120
        own_columns = ['flightKey',	'FPkey', 'callsign', 'adep', 'ades', 'aircraft', 'latitude', 'longitude',
                       'altitude', 'timestamp', 'cruiseAltitudeInFeet', 'cruiseSpeed', 'aircraftWake', 'flightType',
                       'timestamp_start', 'timestamp_end', 'atco_event_time_annotation', 'mwm_codigo', 'EntryExitKey_x',
                       'myID', 'flightKeyRels', 'EntryTimeRels', 'ExitTimeRels', 'EntryExitKey_y', 'SectorID',
                       'flightKeyRel', 'entryTime', 'exitTime']
        rel_columns = ['RTkey', 'FPkey', 'callsign', 'departureAerodrome', 'arrivalAerodrome', 'cruiseAltitudeInFeets',
                       'cruiseSpeed', 'flightType', 'aircraftType', 'aircraftWake', 'timestamp', 'longitude', 'latitude',
                       'altitudeInFeets', 'heading', 'moduleSpeedInKnots', 'xSpeedInKnots', 'ySpeedInKnots',
                       'verticalSpeedInFeetsPerSecond']

        own_df = pd.read_csv('/media/hdd/Documents/TP4CFT_Datasets/enriched_w_sectors/TAPAS_env_samples/'
                             '1000392_enriched_w_sectors.csv', sep='\t', names=own_columns)
        own_df = own_df[(own_df['timestamp'] >= self.init_timestamp) &
                        (own_df['timestamp'] <= self.init_timestamp+5)]

        flight_idx = 0
        for idx, neighbour_flight in own_df[['flightKeyRels', 'EntryTimeRels', 'ExitTimeRels']].iterrows():
            flight_key_rels = list(map(int, neighbour_flight['flightKeyRels'].split(',')))
            entry_time_rels = neighbour_flight['EntryTimeRels'].split(',')
            exit_time_rels = neighbour_flight['ExitTimeRels'].split(',')
            for i, flight_key in enumerate(flight_key_rels):
                if flight_key not in self.flight_index:
                    flight_df = \
                        pd.read_csv('/media/hdd/Documents/TP4CFT_Datasets/enriched_w_sectors/TAPAS_env_samples/' +
                                    str(flight_key)+'.csv', sep=',', names=rel_columns)
                    flight_df = flight_df.rename(columns={'RTkey': 'flightKey', 'altitudeInFeets': 'altitude'})

                    flight_df['timestamp'] = flight_df['timestamp'].\
                        apply(lambda x: (datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')).
                              replace(tzinfo=timezone.utc).timestamp())
                    flight_df = flight_df[(flight_df['timestamp'] >= int(entry_time_rels[i])) &
                                          (flight_df['timestamp'] <= int(exit_time_rels[i]))]

                    if not env_config['historical_transition']:
                        flight_df = flight_df[(flight_df['timestamp'] >= int(entry_time_rels[i])) &
                                              (flight_df['timestamp'] <= int(exit_time_rels[i])) &
                                              (flight_df['timestamp'] >= int(init_timestamp))].sort_values('timestamp')
                        flight_df = flight_df.iloc[[0, 1]]

                    flight_df = flight_utils.transform_compute_speed_bearing(flight_df)
                    flight_df['timestamp'] = flight_df['timestamp'].astype(int)
                    if not env_config['historical_transition']:
                        flight_df = flight_df.iloc[[0]]
                    first_row = flight_df.iloc[0]
                    entry_timestamp = first_row['timestamp']
                    flightKey = int(first_row['flightKey'])
                    if entry_timestamp not in self.flight_entry_index:
                        self.flight_entry_index[entry_timestamp] = set([flightKey])
                    else:
                        self.flight_entry_index[entry_timestamp].add(flightKey)

                    flight_df = flight_df[['flightKey', 'longitude', 'latitude', 'x', 'y', 'altitude', 'timestamp',
                                           'speed_x_component', 'speed_y_component', 'course', 'speed', 'alt_speed']]

                    self.flight_index[int(flight_key)] = {'idx': flight_idx,
                                                          'entryTime': int(entry_time_rels[i]),
                                                          'exitTime': int(exit_time_rels[i]),
                                                          'df': flight_df}
                    flight_idx += 1

        if not env_config['historical_transition']:
            own_df = own_df[(own_df['timestamp'] >= int(init_timestamp))].sort_values('timestamp')
            own_df = own_df.iloc[[0, 1]]
        own_first_row = own_df.iloc[0]
        own_df = flight_utils.transform_compute_speed_bearing(own_df)[['flightKey', 'longitude', 'latitude',
                                                                       'x', 'y', 'altitude', 'timestamp',
                                                                       'speed_x_component', 'speed_y_component',
                                                                       'course', 'speed', 'alt_speed']]
        own_df = own_df.iloc[[0]]

        ownKey = int(own_first_row['flightKey'])
        self.flight_index[ownKey] = {'idx': flight_idx,
                                     'entryTime': own_first_row['entryTime'],
                                     'exitTime': own_first_row['exitTime'],
                                     'df': own_df}

        entry_timestamp = own_first_row['timestamp']

        if entry_timestamp not in self.flight_entry_index:
            self.flight_entry_index[entry_timestamp] = set([ownKey])
        else:
            self.flight_entry_index[entry_timestamp].add(ownKey)

def step_from_historical_data_unit_test(env):

    done = False
    i = 0
    ep_time_start = time.time()
    while not done:
        s_time = time.time()
        done = env.step_from_historical_data()

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))

def one_episode_FL_change(env, episode_num, actions, ve_step_flag):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                  direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0]).astype(object)
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2, 0])

    while not done:
        if env.timestamp < 1564761030:
            entry = [0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            entry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for idx, flightKey in enumerate(env.flight_arr[:, 0]):
            act_id = str(env.timestamp) + '_' + str(env.flight_arr[idx, 0]) + '_A1_0_0'
            res_acts_ID[idx] = act_id
        actions = np.array([entry for i in range(env.flight_arr.shape[0])]).astype(np.float64)
        actions[env.executing_FL_change] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        res_acts_ID[env.executing_FL_change] = str(env.timestamp) + '_flightKey_continue_action_0'

        s_time = time.time()

        if not ve_step_flag:
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
            finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
            executing_resume_to_fplan = env.step(actions, res_acts_ID)
        else:
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases, \
            finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
            executing_resume_to_fplan = env.validation_exercise_step(actions, res_acts_ID)

        step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask], step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude', 'speed_x_component',
                                                 'speed_y_component', 'course', 'speed', 'alt_speed',
                                                 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_' + str(i + 1) for i in range(4)] +
                                                ['wp_hdistance_' + str(i + 1) for i in range(4)] +
                                                ['wp_vdistance_' + str(i + 1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y','rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def one_episode_direct_to_4th_wp(env, episode_num, actions, ve_step_flag):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                 direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0])
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2, 0])

    while not done:

        arr = np.array([[1, 2, 3, 4] for i in range(env.flight_arr.shape[0])])*env.available_wps
        arr2 = np.max(arr, axis=1).astype(np.int64)
        entries = []

        for a in arr2:
            entry = [0, 0, 0, 0]
            if a > 0:
                entry[a-1] = 1
            entries.append(entry)

        entries_np = np.array(entries)
        actions = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(env.flight_arr.shape[0])]).astype(np.float64)
        actions[:, np.array([3, 5, 6, 7])] = entries_np

        for idx, flight in enumerate(env.flight_arr):
            if flight[0] == 4803126 and np.any(entries_np[idx] == 1) :
                env.compute_additional_nautical_miles_direct_to(flight[0], actions[idx])

        actions[env.executing_direct_to] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        s_time = time.time()

        if not ve_step_flag:
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
            finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
            executing_resume_to_fplan = env.step(actions, res_acts_ID)
            step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])
        else:
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
            finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
            executing_resume_to_fplan = env.validation_exercise_step(actions, res_acts_ID)
            step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask], step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude', 'speed_x_component', 'speed_y_component',
                                                 'course', 'speed', 'alt_speed', 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_' + str(i + 1) for i in range(4)] +
                                                ['wp_hdistance_' + str(i + 1) for i in range(4)] +
                                                ['wp_vdistance_' + str(i + 1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y', 'rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def one_episode_dcourse(env, episode_num, actions):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                  direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = ['res_act_dummy_ID'] * env.flight_arr.shape[0]
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2,0])
    print('{0}_{1}_{2}_{3}'.format(int(env.timestamp), int(1), 'continue', '0'))

    while not done:
        if i % 2 == 0:
            dc = 20
        else:
            dc = -20
        actions = np.array([[dc, 0, 0, 0, 0, 0, 0, 0, 0, 210, 0] for l in range(env.flight_arr.shape[0])])
        for idx, flight in enumerate(env.flight_arr):
            if env.executing_course_change[idx]:
                res_acts_ID[idx] = '{0}_{1}_{2}_{3}'.format(int(env.timestamp), int(flight[0]), 'continue', '0')
            else:
                res_acts_ID[idx] = '{0}_{1}_{2}_{3}'.format(int(env.timestamp), int(flight[0]), 'S2', str(dc))

        res_acts_ID = np.array(res_acts_ID)
        actions[env.executing_course_change] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        s_time = time.time()

        for idx in np.where((actions[:, 0] != 0) & (~env.executing_course_change) & (env.active_flights_mask))[0]:
            env.compute_additional_nautical_miles_course_change(env.flight_arr[idx, 0], actions[idx])

        flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
        finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
        executing_resume_to_fplan = env.step(actions, res_acts_ID)
        step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask],
                                         step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude', 'speed_x_component',
                                                 'speed_y_component', 'course', 'speed', 'alt_speed',
                                                 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point','h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_' + str(i + 1) for i in range(4)] +
                                                ['wp_hdistance_' + str(i + 1) for i in range(4)] +
                                                ['wp_vdistance_' + str(i + 1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y', 'rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def one_episode_historical(env, episode_num, actions):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                  direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0])
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2, 0])

    while not done:

        actions = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] for l in range(env.flight_arr.shape[0])])

        s_time = time.time()

        flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
        finished_FL_change, finished_direct_to, finsihed_resume_to_fplan, executing_FL_change, executing_direct_to,\
        executing_resume_to_fplan = env.step(actions, res_acts_ID)
        step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask], step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude', 'speed_x_component', 'speed_y_component',
                                                 'course', 'speed', 'alt_speed', 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_'+str(i+1) for i in range(4)] +
                                                ['wp_hdistance_'+str(i+1) for i in range(4)] +
                                                ['wp_vdistance_'+str(i+1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y','rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def one_episode_direct_to_next_wp(env, episode_num, actions):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                  direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0]).astype(object)
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2, 0])

    while not done:
        arr = np.array([[1, 2, 3, 4] for i in range(env.flight_arr.shape[0])])*env.available_wps
        arr2 = np.min(arr, axis=1).astype(np.int64)
        entries = []

        for idx, available_wps in enumerate(env.available_wps):
            act_id = str(env.timestamp) + '_' + str(env.flight_arr[idx, 0]) + '_A3_'
            if available_wps[1] == 1:
                entry = [0, 1, 0, 0]
                act_id += '2_0'
            elif available_wps[0] == 1:
                entry = [1, 0, 0, 0]
                act_id += '1_0'
            else:
                entry = [0, 0, 0, 0]
                act_id += '0_0'
            res_acts_ID[idx] = act_id
            entries.append(entry)

        actions = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(env.flight_arr.shape[0])])
        actions[:, np.array([3, 5, 6, 7])] = np.array(entries)

        actions[env.executing_direct_to] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if env.active_flights_mask[0] and np.any(np.array(entries[0]) == 1) and not env.executing_direct_to[0]:
            print(env.compute_additional_nautical_miles_direct_to(env.flight_arr[0, 0], actions[0]))
        res_acts_ID[env.executing_direct_to] = str(env.timestamp)+'_flightKey_continue_action_0'
        s_time = time.time()

        flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
        finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
        executing_resume_to_fplan = env.step(actions, res_acts_ID)
        step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask], step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: '+str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude', 'speed_x_component', 'speed_y_component',
                                                 'course', 'speed', 'alt_speed', 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_'+str(i+1) for i in range(4)] +
                                                ['wp_hdistance_'+str(i+1) for i in range(4)] +
                                                ['wp_vdistance_'+str(i+1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y', 'rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def one_episode_resume_fp(env, episode_num, actions):
    """ actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
                  direct_to_3rd, direct_to_4th, continue_action, duration] """
    res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0])
    done = False
    i = 0
    states = []
    ep_time_start = time.time()
    print(env.flight_arr[:2, 0])

    while not done:

        actions = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] for i in range(env.flight_arr.shape[0])])

        actions[env.executing_resume_to_fplan] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        s_time = time.time()

        flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases,\
        finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
        executing_resume_to_fplan = env.step(actions, res_acts_ID)
        step_arr = np.array([[env.timestamp]] * flight_arr[env.active_flights_mask].shape[0])

        if flight_arr[env.active_flights_mask].shape[0] > 0:
            states.append(np.concatenate([flight_arr[env.active_flights_mask],
                                         step_arr], axis=1))

        print('step: '+str(i)+', time: '+str(time.time()-s_time))

        i += 1

    print('episode_time: ' + str(time.time()-ep_time_start))
    states_np = np.concatenate(states, axis=0)
    states_df = pd.DataFrame(states_np, columns=['flightKey', 'longitude', 'latitude',
                                                 'x', 'y', 'altitude',
                                                 'speed_x_component', 'speed_y_component',
                                                 'course', 'speed', 'alt_speed', 'relative_bearing_to_exit_point',
                                                 'alt_diff_wrt_exit_point', 'h_diff_wrt_exit_point'] +
                                                ['wp_dcourse_'+str(i+1) for i in range(4)] +
                                                ['wp_hdistance_'+str(i+1) for i in range(4)] +
                                                ['wp_vdistance_'+str(i+1) for i in range(4)] +
                                                ['step'])
    states_df.to_csv('./states_out_'+str(episode_num)+'.csv', index=False)

    pd.DataFrame(env.exit_wps, columns=['lat', 'lon', 'timestamp', 'alt', 'Sector', 'x', 'y', 'rank', 'flightKey']).\
        to_csv('./exit_wp_'+str(episode_num)+'.csv', index=False)

def run(act_dict):
    env = Environment()
    for episode in act_dict:
        print('Episode: '+str(episode))
        env.initialize()
        for step in act_dict[episode]:
            print('Step: ' + str(step))
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases = \
                env.step(act_dict[episode][step])
            all_agents = np.unique(edges[:, 0])
            if episode == 2 and step == 485:
                print('id, alt,v speed')
                print(flight_arr[:, np.array([0, 5, 10])])
                print(reward_per_factor)
                print(reward)
                print('loss')
                edges_loss = edges[edges[:, -4] == 1]

                print(edges_loss[:, np.array([0, 1, 2, 3, 6, 12, 13, 14, 15])].tolist())
                print('other conflicts')
                edges_confs = edges[edges[:, -2] == 1]
                print(edges_confs[:, np.array([0, 1, 2, 3, 6, 12, 13, 14, 15])].tolist())
                print('alerts')
                edges_alerts = edges[edges[:, -1] == 1]
                print('tcpa,dcpa,dvcpa')
                print(edges_alerts[:, np.array([0, 1, 2, 3, 6, 12, 13, 14, 15])].tolist())
                for edge in edges_alerts:
                    print(edge[0])
                    print(reward_per_factor[env.flight_index[int(edge[0])]['idx']])
                exit(0)

def step_unit_test():
    env = Environment()

    """ [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
         direct_to_3rd, direct_to_4th, continue_action, duration] """
    for i in range(1):
        env.initialize()

        actions = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for i in range(env.flight_arr.shape[0])])

        one_episode_direct_to_next_wp(env, 1, actions)

    print('total_losses_of_separation_per_episode')
    print(env.total_losses_of_separation_per_episode)
    print('total_alerts_per_episode')
    print(env.total_alerts_per_episode)
    print('total_conflicts')
    print(env.total_conflicts)

def course_fplan_intersection_unit_test():
    env = Environment()
    env.initialize()

    print(env.flight_arr[0])
    velocity1 = env.flight_index[4803044]['df'][['speed_x_component', 'speed_y_component',  'alt_speed']].values[0]

    velocity1[0] = -velocity1[0]
    p1 = env.flight_index[4803044]['df'][['x', 'y',  'altitude']].values[0]

    print(env.flight_plans[4803044]['fplans'][env.flight_plans[4803044]['current']])
    env.flight_plans[4803044]['fplans'][env.flight_plans[4803044]['current']][1].to_csv('4803044_current_fp.csv')

    p2 = np.array([947264.4622083987, 621095.2574438052, 37000])

    vel3 = np.array([100, 150, 10])
    velocity2 = np.array([-100, -150, 10])
    p3 = p2+vel3*(45+100)
    p3 = p2 + vel3 * (100 + 100)
    print(p3)

    print(np.sqrt(np.sum((p3[:2]-p2[:2])**2))/1852)
    exit(0)

    p01 = p1+velocity1*(-50)
    p02 = p2 + velocity2 * (-50)

    print(p1)
    print(p2)
    print(velocity1)
    print(velocity2)

    print(flight_plan_utils.detect_conflicts_w_fplan(4803044.0, env.flight_plans, p1, velocity1,
                                                     4803344.0, env.flight_plans, p2, velocity2, env.timestamp))

def both_zero_alt_speed_unit_test():
    env = Environment()
    env.initialize()

    velocity1 = env.flight_index[4803044]['df'][['speed_x_component', 'speed_y_component', 'alt_speed']].values[0]
    velocity1[0] = -velocity1[0]
    p1 = env.flight_index[4803044]['df'][['x', 'y', 'altitude']].values[0]
    velocity2 = env.flight_index[4803126]['df'][['speed_x_component', 'speed_y_component', 'alt_speed']].values[0]
    p2 = env.flight_index[4803126]['df'][['x', 'y', 'altitude']].values[0]
    print(env.flight_plans[4803044]['fplans'][env.flight_plans[4803044]['current']])
    env.flight_plans[4803044]['fplans'][env.flight_plans[4803044]['current']][1].to_csv('4803044_current_fp.csv')
    env.flight_plans[4803126]['fplans'][env.flight_plans[4803126]['current']][1].to_csv('4803044_current_fp.csv')

    p01 = p1 + velocity1 * (-50)
    p02 = p2 + velocity2 * (-50)

    velocity1 = env.flight_arr[env.flight_index[4803288]['idx'], np.array([6, 7, 10])]
    p1 = env.flight_arr[env.flight_index[4803288]['idx'], 3:6]
    velocity2 = env.flight_arr[env.flight_index[4803286]['idx'], np.array([6, 7, 10])]
    p2 = env.flight_arr[env.flight_index[4803286]['idx'], 3:6]

    velocity2[0] = -15
    p1[2] = 39000
    p2[2] = 39000
    print(p1)
    print(p2)
    print(velocity1)
    print(velocity2)

    print(flight_plan_utils.detect_conflicts_w_fplan(4803288.0, env.flight_plans, p1, velocity1,
                                                     4803286.0, env.flight_plans, p2, velocity2, env.timestamp))

def one_ascends_other_descends_unit_test():
    env = Environment()
    env.initialize()

    print(env.flight_arr[0])
    velocity1 = env.flight_index[4803266]['df'][['speed_x_component', 'speed_y_component', 'alt_speed']].values[0]
    p1 = env.flight_index[4803266]['df'][['x', 'y', 'altitude']].values[0]
    velocity2 = env.flight_index[4803344]['df'][['speed_x_component', 'speed_y_component', 'alt_speed']].values[0]
    p2 = env.flight_index[4803344]['df'][['x', 'y', 'altitude']].values[0]

    p1 = np.array([935856., 682675., 32000.])

    p2 = np.array([940000., 680000., 32500.])

    velocity1 = np.array([28., 143., 20.])
    velocity2 = np.array([-28., -143., -10.])

    print(np.sqrt(np.sum((p2[:2]-p1[:2])**2))/1852)

    p01 = p1 + velocity1 * (-50)
    p02 = p2 + velocity2 * (-50)

    p1 = p01
    p2 = p02

    print(p1)
    print(p2)
    print(velocity1)
    print(velocity2)

    print(flight_plan_utils.detect_conflicts_w_fplan(4803266.0, env.flight_plans, p1, velocity1,
                                                     4803344.0, env.flight_plans, p2, velocity2, env.timestamp))

def one_flight_follows_fplan_other_ascends_unit_test():
    env = Environment()
    env.initialize()

    velocity1 = env.flight_index[4803044]['df'][['speed_x_component', 'speed_y_component',  'alt_speed']].values[0]
    velocity1[0] = -velocity1[0]
    p1 = env.flight_index[4803044]['df'][['x', 'y',  'altitude']].values[0]

    p2 = np.array([961764.4622084,  642845.25744381,  38000.])

    velocity2 = np.array([-100, -150, 10])
    print(p1)
    print(p2)
    print(velocity1)
    print(velocity2)

    print(flight_plan_utils.detect_conflicts_w_fplan(4803044.0, env.flight_plans, p1, velocity1,
                                                     4803344.0, env.flight_plans, p2, velocity2, env.timestamp))

def unit_test_actions_impact():

    env = Environment()

    # {"res_acts": np arr Nx3x5, "res_acts_ID": np arr Nx3,
    #  "filt_out_mask": Nx3(True for filtered actions)}

    with open('actions_dict.pkl', 'rb') as f:
        act_dict = pickle.load(f)

    for i in range(1):
        env.initialize()
        time_start = time.time()
        for step in act_dict:
            print('step:', step)
            actions_dict = act_dict[step]

            env.actions_impact(actions_dict)
            if env.timestamp == 1564760190:
                print(env.compute_additional_nautical_miles_course_change(6127213, [-10, 0, 0, 0, 0, 0, 0, 0, 0, 30]))
                print(env.compute_additional_nautical_miles_course_change(6127213, [-20, 0, 0, 0, 0, 0, 0, 0, 0, 30]))

            res_acts = np.zeros((env.flight_arr.shape[0],10))
            res_act_IDs = [0]*env.flight_arr.shape[0]
            for i, mask in enumerate(actions_dict['filt_out_mask']):
                if np.all(mask == True):
                    res_acts[i, :] = actions_dict['res_acts'][i, 0, :]
                    res_act_IDs[i] = actions_dict['res_acts_ID'][i, 0]
                    continue
                for j, act in enumerate(mask):
                    if act == False:
                        res_acts[i, :] = actions_dict['res_acts'][i, j, :]
                        res_act_IDs[i] = actions_dict['res_acts_ID'][i, j]
                        break

            res_act_IDs_np = np.array(res_act_IDs)
            time_before = time.time()
            flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases, \
            finished_FL_change, finished_direct_to, finished_resume_to_fplan, executing_FL_change, executing_direct_to,\
            executing_resume_to_fplan = env.step(res_acts, res_act_IDs_np)
            print(time.time()-time_before)

        print(time.time()-time_start)

def debug_training():
    """
    [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
     direct_to_3rd, direct_to_4th, continue_action, duration, resume_to_flight_plan]"""
    env = Environment()

    with open('dict_actions_debug.pkl', 'rb') as f:
        act_dict = pickle.load(f)

    for i in range(1):

        time_start = time.time()

        for episode in act_dict:
            env.initialize()
            for step in act_dict[episode]:
                print('step:', step)
                env.step_num = step
                actions_dict = act_dict[episode][step]

                res_acts = actions_dict['res_acts']
                res_act_IDs = actions_dict['res_acts_ID']

                res_act_IDs_np = np.array(res_act_IDs)
                time_before = time.time()

                for idx, action in enumerate(res_acts):
                    if env.active_flights_mask[idx]:
                        env.compute_additional_nautical_miles_course_change(env.flight_arr[idx, 0],
                                                                            np.array([-20., 0., 0., 0., 0., 0., 0., 0., 0., 60., 0.]))
                        print(env.compute_shift_from_exit_point(env.flight_arr[idx, 0],
                                                                np.array([-20., 0., 0., 0., 0., 0., 0., 0., 0., 60., 0.])))
                        print(env.compute_shift_from_exit_point(env.flight_arr[idx, 0],
                                                                np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 60., 0.])))

                flight_arr, edges, reward, reward_per_factor, done, clipped_actions, available_wps, flight_phases, \
                finished_FL_change, finished_direct_to, finished_resum, executing_FL_change, executing_direct_to, \
                executing_resume = env.step(res_acts, res_act_IDs_np)
                print(time.time()-time_before)

                if step == 17:
                    env.write_loss_conf_main()
                    print(env.compute_additional_nautical_miles_course_change(6124498,
                                                                              np.array([-20., 0., 0., 0., 0., 0., 0., 0., 0., 60., 0.])))

        print(time.time()-time_start)


if __name__ == '__main__':

    debug_training()