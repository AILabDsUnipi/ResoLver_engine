import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["KMP_WARNINGS"] = "FALSE"

import pickle
import argparse
import time
import numpy as np
import random
from datetime import datetime, timezone

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from env.environment import environment
from env.environment.env_config import env_config
from enhanced_DGN.ReplayBuffer import ReplayBuffer
from enhanced_DGN.DGN_v2 import DGN_model as DGN_model_v2
from enhanced_DGN.DGN_config import dgn_config
from enhanced_DGN.prioritized_replay_buffer.prioritized_experience_replay import Memory as PER
from CDR_config import cdr_config

from keras.utils import np_utils, to_categorical
import pandas as pd
import copy
import psutil
import subprocess

def compute_ROC(edges_features, max_rateOfClosureHV, j_):

    # seconds after first conflict point for which the computation of velocity
    # will be performed in order to calculate relSpeedV for RoC
    dt = 1

    # Columns of edges:
    # 12 time to first conflict point
    # 13 h distance at first conflict point
    # 14 v distance at first conflict point
    # 15 x coordinate agent i at first point of conflict
    # 16 y coordinate agent i at first point of conflict
    # 17 altitude agent i at first point of conflict
    # 18 x coordinate agent j at first point of conflict
    # 19 y coordinate agent j at first point of conflict
    # 20 altitude agent j at first point of conflict
    # 21 velocity_x_component_agent_i at first point of conflict
    # 22 velocity_y_component_agent_i at first point of conflict
    # 23 vertical_speed_component_agent_i at first point of conflict
    # 24 velocity_x_component_agent_j at first point of conflict
    # 25 velocity_y_component_agent_j at first point of conflict
    # 26 vertical_speed_component_agent_j at first point of conflict
    # 27 time to the closest conflict point
    # 28 h distance at closest conflict point
    # 29 v distance at closest conflict point

    # Vertical relative speed
    lower_vert_speed_fcp = \
        edges_features[j_, 23] if edges_features[j_, 17] <= edges_features[j_, 20] else edges_features[j_, 26]
    higher_vert_speed_fcp = \
        edges_features[j_, 26] if edges_features[j_, 17] <= edges_features[j_, 20] else edges_features[j_, 23]

    relSpeedV = (lower_vert_speed_fcp - higher_vert_speed_fcp) * 60
    rateOfClosureV = 0 if (relSpeedV <= 0) \
        else (1 if 0 < relSpeedV <= 1000
              else (2 if 1000 < relSpeedV <= 2000
                    else (4 if 2000 < relSpeedV <= 4000 else 5)))

    # Horizontal relative speed
    # Euclidean distance at the first conflict point:
    # d_fcp = square_root(
    #                       ((x coordinate agent i at first point of conflict - x coordinate agent j at first point of conflict)**2) +
    #                       ((y coordinate agent i at first point of conflict - y coordinate agent j at first point of conflict)**2)
    #                     )
    d_fcp = np.sqrt(((edges_features[j_, 15] - edges_features[j_, 18]) ** 2) +
                    ((edges_features[j_, 16] - edges_features[j_, 19]) ** 2))

    # Computation of coordinates after dt time from first conflict point
    # dt_x_coord_fcp_agent_A = x coordinate agent A at first point of conflict + (velocity_x_component_agent_A at first point of conflict * dt)
    # dt_x_coord_fcp_agent_A = x coordinate agent A at first point of conflict + (velocity_x_component_agent_A at first point of conflict * dt)
    # dt_y_coord_fcp_agent_A = y coordinate agent A at first point of conflict + (velocity_y_component_agent_A at first point of conflict * dt)
    # dt_y_coord_fcp_agent_A = y coordinate agent A at first point of conflict + (velocity_y_component_agent_A at first point of conflict * dt)

    dt_x_coord_fcp_agent_i = edges_features[j_, 15] + (edges_features[j_, 21] * dt)
    dt_y_coord_fcp_agent_i = edges_features[j_, 16] + (edges_features[j_, 22] * dt)
    dt_x_coord_fcp_agent_j = edges_features[j_, 18] + (edges_features[j_, 24] * dt)
    dt_y_coord_fcp_agent_j = edges_features[j_, 19] + (edges_features[j_, 25] * dt)

    d_dt_fcp = np.sqrt(((dt_x_coord_fcp_agent_i - dt_x_coord_fcp_agent_j) ** 2) +
                       ((dt_y_coord_fcp_agent_i - dt_y_coord_fcp_agent_j) ** 2))

    #Convert relSpeedH from m/s to knots by dividing with 0.5144
    relSpeedH = (((d_fcp - d_dt_fcp)/0.5144) / dt)
    rateOfClosureH = 0 if (relSpeedH <= 0) \
        else (1 if 0 < relSpeedH <= 85
              else (2 if 85 < relSpeedH <= 205
                    else (4 if 205 < relSpeedH <= 700 else 5)))

    RoC = (rateOfClosureH if rateOfClosureH >= rateOfClosureV else rateOfClosureV) / max_rateOfClosureHV

    return RoC, rateOfClosureH, rateOfClosureV, relSpeedV, relSpeedH

class CDR(object):

    # "CDR" stands for "Conflicts Detection & Resolution"

    def __init__(self,
                 DGN_model_path_,
                 evaluation_,
                 batch_size_,
                 LRA_,
                 train_episodes_,
                 exploration_episodes_,
                 prioritized_replay_buffer_,
                 scenario_,
                 multi_scenario_training_,
                 debug__,
                 continue_train_,
                 with_RoC_term_reward_,
                 with_slack_notifications_,
                 send_slack_notifications_every_episode_,
                 conc_observations_edges_):

        # Evaluation specifications
        self.evaluation = evaluation_
        if self.evaluation:
            # Disable any tensorflow message
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.num_eval_episodes = dgn_config['num_eval_episodes']

        # Training specifications
        self.DGN_model_path = DGN_model_path_
        self.batch_size = batch_size_
        self.LRA = LRA_
        self.GAMMA = dgn_config['GAMMA']
        self.train_episodes = train_episodes_
        self.exploration_episodes = exploration_episodes_
        self.episode_before_train = dgn_config['episode_before_train']
        self.train_step_per_episode = dgn_config['train_step_per_episode']
        self.prioritized_replay_buffer = prioritized_replay_buffer_
        self.multi_scenario_training = multi_scenario_training_
        self.continue_train = continue_train_
        self.with_slack_notifications = with_slack_notifications_
        self.send_slack_notifications_every_episode = send_slack_notifications_every_episode_
        self.neighbors_observed = dgn_config['neighbors_observed']

        if self.evaluation is False:
            self.alpha = dgn_config['alpha']  # In the paper of DGN this parameter is denoted as 'epsilon'
        else:
            self.alpha = 0.00

        self.min_alpha = dgn_config['min_alpha']

        if self.exploration_episodes >= self.train_episodes:
            print("Exploration episodes should be less than total train episodes!!")
            exit(0)

        self.n_episode = self.train_episodes + self.episode_before_train
        # Below, we use 1e-18 instead of 0, otherwise 'alpha_decay' will be 0.
        self.alpha_decay = 0.0 if self.evaluation is True \
                               else \
                            ((self.min_alpha / self.alpha) ** (1 / self.exploration_episodes)
                             if self.min_alpha != 0.0
                             else
                             (1e-18 / self.alpha) ** (1 / self.exploration_episodes))

        # Debug mode
        self.debug_ = debug__

        # Reward preferences
        self.with_RoC_term_reward = with_RoC_term_reward_

        # Edges mode
        self.conc_observations_edges = conc_observations_edges_

        # Scenario specifications
        self.scenario = scenario_

        if self.multi_scenario_training and not self.evaluation:
            f_selected_scenarios = open('selected_scenarios.txt', 'r')
            lines = f_selected_scenarios.readlines()
            self.scenario_list = [line.split('\n')[0] for line in lines]
            print("Scenario list: {}".format(self.scenario_list))
            print("NUmber of scenarios: {}".format(len(self.scenario_list)))
        else:
            self.scenario_list = [scenario]

        # print RAM usage for each scenario
        RAM_before = psutil.virtual_memory().used / 1000000000
        print("RAM before any scenario: {} GB".format(RAM_before))

        self.env_list = []
        for scen_indx, scen in enumerate(self.scenario_list):
            self.env_list.append(environment.Environment(scen))
            if scen_indx > 0:
                RAM_before = current_RAM
            current_RAM = psutil.virtual_memory().used / 1000000000
            extra_RAM = current_RAM - RAM_before
            print("extra RAM {} GB, scenario {}".format(extra_RAM, scen))

        self.horizontal_minimum = cdr_config['horizontal_minimum']
        self.vertical_minimum = cdr_config['vertical_minimum']
        self.t_CPA_threshold = cdr_config['t_CPA_threshold']
        self.num_edges_features = cdr_config['num_edges_features']
        self.num_norm_edges_features = cdr_config['num_norm_edges_features']
        self.mask_edges_features = cdr_config['mask_edges_features']
        self.mask_self_edges_features = cdr_config['mask_self_edges_features']
        self.RoC_term_weight = cdr_config['RoC_term_weight']
        self.r_norm = cdr_config['r_norm']
        self.altitude_drift_from_exit_point = cdr_config['altitude_drift_from_exit_point']
        self.max_rateOfClosureHV = cdr_config['max_rateOfClosureHV']
        self.min_duration = cdr_config['min_duration']
        self.max_duration = cdr_config['max_duration']
        self.interval_between_two_steps = cdr_config['interval_between_two_steps']
        self.actions_delta_course = cdr_config['actions_delta_course']
        self.num_actions_dc = cdr_config['num_actions_dc']
        self.num_actions_ds = cdr_config['num_actions_ds']
        self.total_duration_values = cdr_config['total_duration_values']
        self.actions_delta_altitude_speed = cdr_config['actions_delta_altitude_speed']
        self.num_actions_as = cdr_config['num_actions_as']
        self.num_dir_wp = cdr_config['num_dir_wp']
        self.actions_list = cdr_config['actions_list']
        self.num_types_of_actions = cdr_config['num_types_of_actions']
        self.types_of_actions = cdr_config['types_of_actions']
        self.types_of_actions_for_evaluation = cdr_config['types_of_actions_for_evaluation']
        self.action_values_for_evaluation = cdr_config['action_values_for_evaluation']
        self.max_poss_actions_for_evaluation = cdr_config['max_poss_actions_for_evaluation']
        self.max_poss_actions_for_evaluation_to_be_passed_to_env = \
            cdr_config['max_poss_actions_for_evaluation_to_be_passed_to_env']
        self.n_actions = cdr_config['n_actions']
        self.end_action_array = cdr_config['end_action_array']
        self.plus_model_action = cdr_config['plus_model_action']
        self.min_alt_speed = env_config['min_alt_speed']
        self.max_alt_speed = env_config['max_alt_speed']
        self.min_h_speed = env_config['min_h_speed']
        self.max_h_speed = env_config['max_h_speed']
        self.D = env_config['D']
        self.max_alt = env_config['max_alt']
        self.max_alt_minus_exit_point_alt = env_config['max_alt-exit_point_alt']
        self.H_dij = env_config['H_dij']
        self.V_dij = env_config['V_dij']
        self.V_dcpa = env_config['V_dcpa']
        self.D_cp = env_config['D_cp']
        self.D_cpa = env_config['D_cpa']
        self.T_cpa = env_config['T_cpa']
        self.T_cp = env_config['T_cp']
        self.num_edges_feats_for_ROC = cdr_config['num_edges_feats_for_ROC']

        # Get the observation space (i.e., the number of observation features without the edges)
        # and the number of agents of each scenario
        envs_observation_space, self.envs_n_agent = self.get_observation_space()

        # Observation space is not different for different scenarios
        self.observation_space = envs_observation_space[0]

        # We consider that the number of flights is the maximum one of the different scenarios.
        # Thus, for the rest of scenarios we will perform padding.
        self.n_agent = max(self.envs_n_agent)

        print('n_actions: ' + str(self.n_actions))
        print('Observation space: ' + str(self.observation_space))
        print('Maximum agents number: ' + str(self.n_agent))
        for scen in range(len(self.scenario_list)):
            print("Scenario: {}".format(self.scenario_list[scen]))
            print("Number of agents: {}".format(self.envs_n_agent[scen]))

        # Define the model
        self.DGN_m = DGN_model_v2(self.n_agent,
                                  self.observation_space,
                                  self.num_norm_edges_features,
                                  self.n_actions + self.plus_model_action,
                                  self.evaluation,
                                  self.DGN_model_path,
                                  self.LRA,
                                  self.continue_train,
                                  self.conc_observations_edges)

        if not self.evaluation:
            # Simple replay buffer is needed even if PER is enabled because
            # we temporarily store there the samples of the current episode
            self.buff = ReplayBuffer()

            if self.prioritized_replay_buffer:
                self.priorit_buff = PER(self.buff.buffer_size, self.DGN_m, self.exploration_episodes)

        # Get the name of the host to use it when a message is sent via Slack
        self.hostname = (str(subprocess.check_output('hostname', shell=True)).split("'")[1]).split('\\n')[0]

        # Get the column names of the output files
        self.flights_positions_df_columns = cdr_config['flights_positions_df_columns']
        self.confl_resol_act_df_columns = cdr_config['confl_resol_act_df_columns']
        self.htmp_by_agent_df_columns = cdr_config['htmp_by_agent_df_columns']

        # Initialize the rest of the variables to None
        self.f = None
        self.dict_actions = None
        self.dict_actions_debug = None
        self.dict_actions_attentions = None
        self.i_episode = None
        self.loss = None
        self.total_episode_num_ATC_instr = None
        self.total_episode_score = None
        self.total_episode_losses_of_separation = None
        self.total_episode_alerts = None
        self.steps_per_scenario = None
        self.total_additional_nautical_miles = None
        self.total_confls_with_positive_tcpa = None
        self.total_additional_duration = None
        self.flight_ids_one_hot_np = None
        self.flight_ids_one_hot_dict = None
        self.flights_idxs_to_ids = None
        self.flights_positions = None
        self.confl_htmaps = None
        self.confl_resol_act = None
        self.htmp_by_agent = None
        self.losses_file = None
        self.conflicts_file = None
        self.steps = None
        self.score = None
        self.num_ATC_instr = None
        self.add_nautical_miles = None
        self.add_duration = None
        self.confls_with_positive_tcpa = None
        self.done = None
        self.history_loss_confl = None
        self.durations_of_actions = None
        self.previous_durations_of_actions = None
        self.reward_history = None
        self.previous_end_of_maneuver = None
        self.previous_fls_with_loss_of_separation = None
        self.previous_fls_with_conflicts = None
        self.previous_np_mask_climb_descend_res_fplan_with_confl_loss = None
        self.previous_np_mask_res_fplan_after_maneuv = None
        self.previous_duration_dir_to_wp_ch_FL_res_fplan = None
        self.flight_array_ = None
        self.previous_flight_array_ = None
        self.edges = None
        self.available_wps = None
        self.flight_phases = None
        self.finished_FL_change = None
        self.finished_direct_to = None
        self.finished_resume_to_fplan = None
        self.executing_FL_change = None
        self.executing_direct_to = None
        self.executing_resume_to_fplan = None
        self.active_flights_mask = None
        self.timestamp = None
        self.flight_array = None
        self.adjacency_one_hot = None
        self.adjacency_matrix = None
        self.norm_edges_feats = None
        self.fls_with_loss_of_separation = None
        self.fls_with_conflicts = None
        self.fls_with_alerts = None
        self.count_fls_with_loss_of_separation = None
        self.count_fls_with_conflicts = None
        self.count_fls_with_alerts = None
        self.count_total_conflicts_not_alerts_per_flt = None
        self.count_total_alerts_per_flt = None
        self.loss_ids = None
        self.alrts_ids = None
        self.confls_ids_with_positive_tcpa = None
        self.confls_ids_with_negative_tcpa = None
        self.MOC = None
        self.RoC = None
        self.obs = None
        self.next_adjacency_one_hot = None
        self.next_adjacency_matrix = None
        self.next_norm_edges_feats = None
        self.next_fls_with_loss_of_separation = None
        self.next_fls_with_conflicts = None
        self.next_fls_with_alerts = None
        self.count_next_fls_with_loss_of_separation = None
        self.count_next_fls_with_conflicts = None
        self.count_next_fls_with_alerts = None
        self.count_next_total_conflicts_not_alerts_per_flt = None
        self.count_next_total_alerts_per_flt = None
        self.next_loss_ids = None
        self.next_alrts_ids = None
        self.next_confls_ids_with_positive_tcpa = None
        self.next_confls_ids_with_negative_tcpa = None
        self.next_MOC = None
        self.next_RoC = None
        self.unfixed_acts = None
        self.acts = None
        self.actions = None
        self.att_all = None
        self.durations_of_actions_for_evaluation_ = None
        self.rand_choice = None
        self.env = None
        self.env_number = None
        self.dict_agents_actions = None
        self.cur_num_ATC_instr = None
        self.np_dur_of_actions = None
        self.end_of_maneuver = None
        self.np_mask_climb_descend_res_fplan_with_confl_loss = None
        self.np_mask_res_fplan_after_maneuv = None
        self.duration_dir_to_wp_ch_FL_res_fplan = None
        self.np_resolutionActionID_only_for_train = None
        self.cur_additional_nautical_miles = None
        self.cur_additional_duration = None
        self.actions_duration_to_be_passed_to_env = None
        self.true_acts = None
        self.valid_acts_to_be_executed_ID = None
        self.possible_actions = None
        self.possible_actions_type = None
        self.q_values_of_poss_actions = None
        self.filtered_out_reason = None
        self.true_actions_for_evaluation = None
        self.indices_possible_actions = None
        self.durations_of_actions_for_evaluation = None
        self.dummy_possible_actions_mask = None
        self.action_in_progress = None
        self.not_action_due_to_phase_mask = None
        self.res_acts_dict_for_evaluation = None
        self.flight_array__ = None
        self.next_flight_array__ = None
        self.next_flight_array_ = None
        self.next_edges = None
        self.reward = None
        self.reward_per_factor = None
        self.next_available_wps = None
        self.next_flight_phases = None
        self.next_finished_FL_change = None
        self.next_finished_direct_to = None
        self.next_finished_resume_to_fplan = None
        self.next_executing_FL_change = None
        self.next_executing_direct_to = None
        self.next_executing_resume_to_fplan = None
        self.next_active_flights_mask = None
        self.next_timestamp = None
        self.next_flight_array = None
        self.normalized_reward = None
        self.next_norm_edges_feats_based_on_previous_adj_matrix = None
        self.new_states = None
        self.dones = None
        self.active_flights_m = None
        self.next_fls_with_loss_of_separation_m = None
        self.next_fls_with_conflicts_m = None
        self.history_loss_confl_m = None
        self.next_active_flights_m = None
        self.durations_of_actions_b = None
        self.next_timestamp_b = None
        self.data_needed_for_delayed_update_b = None
        self.np_mask_climb_descend_res_fplan_with_confl_loss_b = None
        self.np_mask_res_fplan_after_maneuv_b = None
        self.next_executing_FL_change_b = None
        self.next_executing_direct_to_b = None
        self.next_flight_phases_b = None
        self.executing_FL_change_b = None
        self.executing_direct_to_b = None
        self.next_available_wps_b = None
        self.next_executing_resume_fplan_b = None
        self.num_of_different_matrices_needed = None
        self.target_q_values = None
        self.different_target_q_values_mask = None
        self.list_temp_episode_samples = None
        self.maxq_actions = None
        self.default_action_q_value_mask = None
        self.not_use_max_next_q_mask = None
        self.solved_conflicts = None
        self.total_conflicts = None
        self.total_loss = None
        self.number_conflicting_flight_pairs = None
        self.number_loss_flight_pairs = None
        self.number_loss_flight_pairs_without_conflict_or_loss_before = None
        self.solved_conflicts_in_groups = None
        self.conflict_resolution_duration_in_groups = None

    def run_CDR(self):

        # Define and initialize the log files
        self.initialize_log_files()

        # Sent notification to slack
        if with_slack_notifications:
            message = '"A new experiment has just started at host {}, with {} scenarios in total, ' \
                      'with batch_size {}, ' \
                      'with PER {}, with exploration_episodes {}, ' \
                      'with total train episodes {}."'.format(self.hostname, len(self.scenario_list),
                                                              self.DGN_m.batch_size, self.prioritized_replay_buffer,
                                                              self.exploration_episodes, self.train_episodes)
            self.send_slack_message(message)

        ####Flight simulation begins#####
        self.i_episode = 0
        while (self.i_episode < self.n_episode and not self.evaluation) or \
              (self.evaluation and self.i_episode < self.num_eval_episodes):

            # Sent notification to slack
            if self.with_slack_notifications and self.i_episode % self.send_slack_notifications_every_episode == 0:
                message = '"The experiment running in host {} is currently in episode {}."'\
                          .format(self.hostname, self.i_episode)
                self.send_slack_message(message)

            # Reduce alpha
            if self.i_episode >= self.episode_before_train:
                self.alpha *= self.alpha_decay
            if (self.alpha < self.min_alpha or
                (self.i_episode >= self.exploration_episodes + self.episode_before_train and
                 self.min_alpha == 0.0)) \
                and not self.evaluation:
                self.alpha = self.min_alpha

            self.i_episode = self.i_episode + 1
            print('\nEpisode: ' + str(self.i_episode))

            # Initialize variables in each episode
            self.initialize_episode_vars()

            for self.env_number, self.env in enumerate(self.env_list):

                # Initialize the appropriate variables and output files to run the current environment
                self.initialize_vars_and_out_files_for_cur_env()

                # Initialize the environment which runs the current scenario
                self.initialize_environment_and_get_returns()

                # Run a whole episode for the current environment
                self.run_episode_for_current_env()

            # Print stats of the episode for all environment and write log files
            self.write_log_files_of_episode_for_all_env()

            ##########In the following cases only continue to training##########
            if not self.evaluation and self.i_episode >= self.episode_before_train:
                self.training()

        #Close log files and delete the unnecessary ones
        self.close_log_files()

    def get_observation_space(self):
        env_obs_space = []
        env_number_of_flights = []
        for env_ in self.env_list:
            flight_arr, _, _, fls_phase, _, _, _, _, _, _ = env_.initialize()
            flight_feat = self.get_flight_features(flight_arr.copy(), env_.active_flights_mask.copy(), fls_phase.copy())
            env_obs_space.append(int(flight_feat[0].shape[0]))
            env_number_of_flights.append(int(flight_feat.shape[0]))
        return env_obs_space, env_number_of_flights

    def get_flight_features(self, flight_ar_, active_flights, phases):

        # If any way point is not available then its feature will be NaN. Thus, we should replace NaN values with 0.
        flight_ar_[:, 14:] = np.where(np.isnan(flight_ar_[:, 14:]), 0, flight_ar_[:, 14:])

        # Provide to agents their phases.
        # This is critical for each one to know the phase of the other agents and act accordingly. Specifically,
        # if the phase of an agent is 'climbing'/'descending', we assign 1, otherwise 0.
        transformed_phases = np.where((np.array(phases) == 'climbing') | (np.array(phases) == 'descending'), 1, 0)

        if ((flight_ar_[:, 5] > self.max_alt) * active_flights).any():  # Check if max_alt is maintained
            print("Flight with higher altitude than max_alt was found!!!")
            exit(0)

        # The flight features are 14 (but we keep only 7, and as the 2 of these are angles,
        # the total normalized features are 9).
        # The way point features are 3 (as we have 4 way points, the total way point features are 12,
        # and as the 4 of these are angles, the total way point normalized features are 16).
        # Plus the flight phase, the total normalized features of an agent are 26.
        flight_features = np.zeros((flight_ar_.shape[0], 26), dtype=float)
        flight_features[:, 0] = flight_ar_[:, 5] / self.max_alt  # altitude
        flight_features[:, 1] = np.cos(flight_ar_[:, 8])  # chi
        flight_features[:, 2] = np.sin(flight_ar_[:, 8])  # chi
        flight_features[:, 3] = \
            (flight_ar_[:, 9] - self.min_h_speed) / (self.max_h_speed - self.min_h_speed)  # horizontal speed
        flight_features[:, 4] = \
            (flight_ar_[:, 10] - self.min_alt_speed) / (self.max_alt_speed - self.min_alt_speed)  # altitude speed
        flight_features[:, 5] = np.cos(flight_ar_[:, 11])  # chi - psi
        flight_features[:, 6] = np.sin(flight_ar_[:, 11])  # chi - psi
        flight_features[:, 7] = flight_ar_[:, 13] / self.D  # horizontal distance from exit point
        flight_features[:, 8] = \
            flight_ar_[:, 12] / self.max_alt_minus_exit_point_alt  # vertical distance from exit point
        flight_features[:, 9:9 + self.num_dir_wp] = \
            np.cos(flight_ar_[:, 14:14 + self.num_dir_wp])  # cos wp_dcourse_x for x : 1 2 3 4
        flight_features[:, 9 + self.num_dir_wp:9 + (2 * self.num_dir_wp)] = np.sin(
            flight_ar_[:, 14:14 + self.num_dir_wp])  # sin wp_dcourse_x for x : 1 2 3 4
        flight_features[:, 9 + (2 * self.num_dir_wp):9 + (3 * self.num_dir_wp)] = \
            flight_ar_[:, 14 + self.num_dir_wp:14 + (2 * self.num_dir_wp)] / \
            self.H_dij  # wp_hdistance_k for k : 1 2 3 4
        flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)] = \
            flight_ar_[:, 14 + (2 * self.num_dir_wp):14 + (3 * self.num_dir_wp)] / \
            self.V_dij  # wp_vdistance_l for l: 1 2 3 4
        flight_features[:, 9 + (4 * self.num_dir_wp)] = transformed_phases.copy()

        # Block 'd' normalized values to be in the range [0, 20]
        flight_features[:, 8] = np.where(flight_features[:, 8] > 20.0, 20.0, flight_features[:, 8])

        # Block normalized values of vertical distance to any way point to be in the range [-20, 20]
        flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)] = \
            np.where(flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)] < -20.0, -20.0,
                     flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)])
        flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)] = \
            np.where(flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)] > 20.0, 20.0,
                     flight_features[:, 9 + (3 * self.num_dir_wp):9 + (4 * self.num_dir_wp)])

        """
        #Check if any feature of any flight is out of the range [-20,20]

        for agg in range(flight_ar_.shape[0]):
            if active_flights[agg] and ((flight_features[agg] > 20).any() or (flight_features[agg] < -20).any()):
                print("Agent {}".format(agg))
                print("flight_features {}".format(str(flight_features[agg])))
        """

        return flight_features

    @staticmethod
    def send_slack_message(message_):
        text_var_ = '"text"'
        os.system("curl -X POST -H 'Content-type: application/json' --data '{" + text_var_ + ":" + message_ + "}' " +
                  "https://hooks.slack.com/services/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")

    def flight_ids_to_one_hot(self):
        flights_idxs = np.arange(0, self.n_agent)
        self.flight_ids_one_hot_np = np.zeros((self.n_agent + 1, self.n_agent))

        for idx, item in enumerate(self.env.flight_index.items()):
            if item[1]['idx'] != idx:
                print('flight idx found to be in different order from that which is from 0 to n_flights')
                exit(0)
        self.flight_ids_one_hot_np[:self.n_agent] = to_categorical(flights_idxs, num_classes=self.n_agent)
        self.flight_ids_one_hot_np[len([*self.env.flight_index]):, :] = 0.

        self.flight_ids_one_hot_dict = {}  # One-hot representation is based on flights idxs, not to ids, but this dictionary maps this representation to ids.
        for item in self.env.flight_index.items():
            self.flight_ids_one_hot_dict[item[0]] = self.flight_ids_one_hot_np[item[1]['idx']]

    def flight_ids_to_idx(self):
        self.flights_idxs_to_ids = np.arange(0, self.n_agent + 1)
        for item in self.env.flight_index.items():
            self.flights_idxs_to_ids[item[1]['idx']] = item[0]
        self.flights_idxs_to_ids[len([*self.env.flight_index]):] = -1

    def initialize_environment_and_get_returns(self):

        # Initialize the environment and get the returns
        env_returns = self.env.initialize()
        self.flight_array__ = env_returns[0]
        edges_ = env_returns[1]
        available_wps_ = env_returns[2]
        flight_phases_ = env_returns[3]
        finished_FL_change_ = env_returns[4]
        finished_direct_to_ = env_returns[5]
        finished_resume_to_fplan_ = env_returns[6]
        executing_FL_change_ = env_returns[7]
        executing_direct_to_ = env_returns[8]
        executing_resume_to_fplan_ = env_returns[9]

        self.flight_array_ = np.append(self.flight_array__.copy(),
                                       [[0] * self.flight_array__.shape[1]
                                        for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                       axis=0) \
                                 if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                 else self.flight_array__.copy()
        self.previous_flight_array_ = self.flight_array_.copy()
        self.edges = edges_.copy()
        self.available_wps = np.append(available_wps_.copy(),
                                       [[0] * available_wps_.shape[1]
                                        for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                       axis=0) \
                            if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                            else available_wps_.copy()
        self.flight_phases = flight_phases_.copy() + \
                             ['innactive' for _ in
                              range(self.n_agent - self.envs_n_agent[self.env_number])] \
                            if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                            else flight_phases_.copy()
        self.finished_FL_change = np.append(finished_FL_change_.copy(),
                                            [False for _ in
                                             range(self.n_agent - self.envs_n_agent[self.env_number])],
                                            axis=0) \
                                if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                else finished_FL_change_.copy()
        self.finished_direct_to = np.append(finished_direct_to_.copy(),
                                            [False for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                            axis=0) \
                                    if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                    else finished_direct_to_.copy()
        self.finished_resume_to_fplan = np.append(finished_resume_to_fplan_.copy(),
                                                  [False for _ in
                                                   range(self.n_agent - self.envs_n_agent[self.env_number])],
                                                  axis=0) \
                                        if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                        else finished_resume_to_fplan_.copy()
        self.executing_FL_change = np.append(executing_FL_change_.copy(),
                                             [False for _ in
                                              range(self.n_agent - self.envs_n_agent[self.env_number])],
                                             axis=0) \
                                    if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                    else executing_FL_change_.copy()
        self.executing_direct_to = np.append(executing_direct_to_.copy(),
                                             [False for _ in
                                             range(self.n_agent - self.envs_n_agent[self.env_number])],
                                             axis=0) \
                                    if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                    else executing_direct_to_.copy()
        self.executing_resume_to_fplan = np.append(executing_resume_to_fplan_.copy(),
                                                   [False for _ in
                                                    range(self.n_agent - self.envs_n_agent[self.env_number])],
                                                   axis=0) \
                                        if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                        else executing_resume_to_fplan_.copy()

        self.active_flights_mask = np.append(self.env.active_flights_mask.copy(),
                                             [False for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                             axis=0) \
                                    if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                    else self.env.active_flights_mask.copy()

        # Get the first timestamp
        self.timestamp = self.env.timestamp

        # Normalize the flight features
        self.flight_array = self.get_flight_features(self.flight_array_, self.active_flights_mask, self.flight_phases)

        # Normalize the edges features, get the adjacency matrix and other useful information about the state
        self.adjacency_matrix_and_norm_edges_features(self.edges)

    def initialize_log_files(self):
        self.f = open('episodes_log.txt', 'w')

        if self.evaluation:
            self.dict_actions_attentions = {}
            self.dict_actions = {}

        if self.debug_:
            self.dict_actions_debug = {}

    def initialize_episode_vars(self):
        self.loss = 0
        self.total_episode_num_ATC_instr = []
        self.total_episode_score = []
        self.total_episode_losses_of_separation = []
        self.total_episode_alerts = []
        self.steps_per_scenario = []
        self.total_additional_nautical_miles = []
        self.total_confls_with_positive_tcpa = []
        self.total_additional_duration = []

    def initialize_vars_and_out_files_for_cur_env(self):
        self.flight_ids_to_one_hot()
        self.flight_ids_to_idx()

        if self.evaluation:
            self.dict_actions_attentions[self.i_episode - 1] = {}
            self.flights_positions = pd.DataFrame(columns=self.flights_positions_df_columns)
            self.confl_htmaps = pd.DataFrame(columns=['HitMapID', 'RTkey', 'ConflictID'] +
                                                     [self.flights_idxs_to_ids[iii] for iii in
                                                      range(self.envs_n_agent[self.env_number])])
            self.confl_resol_act = pd.DataFrame(columns=self.confl_resol_act_df_columns)
            self.htmp_by_agent = [pd.DataFrame(columns=self.htmp_by_agent_df_columns)
                                  for _ in range(self.envs_n_agent[self.env_number])]
            self.losses_file = pd.DataFrame(columns=['conflictID'])
            self.conflicts_file = pd.DataFrame(columns=['conflictID', 'warn_type'])
        if self.debug_:
            self.dict_actions_debug[self.i_episode - 1] = {}

        self.steps = 0
        self.obs = []
        self.score = []
        self.num_ATC_instr = 0
        self.add_nautical_miles = 0
        self.add_duration = 0
        self.confls_with_positive_tcpa = 0
        self.done = False
        self.history_loss_confl = []

        # The first part of the tuple is the duration and
        # the second is the timestamp when this duration was decided
        self.durations_of_actions = [(0, 0) for _ in range(self.n_agent)]
        self.previous_durations_of_actions = [(0, 0) for _ in range(self.n_agent)]

        # Keep the history of rewards to update q-values.
        # This is necessary because the q-value of an action depends on the discounted reward for the
        # executed maneuver, the duration of which is equal to the duration of the executed action.
        self.reward_history = [[] for _ in range(self.n_agent)]
        self.previous_end_of_maneuver = [True for _ in range(self.n_agent)]
        self.previous_fls_with_loss_of_separation = []
        self.previous_fls_with_conflicts = []
        self.previous_np_mask_climb_descend_res_fplan_with_confl_loss = [False for _ in range(self.n_agent)]
        self.previous_np_mask_res_fplan_after_maneuv = [False for _ in range(self.n_agent)]
        self.previous_duration_dir_to_wp_ch_FL_res_fplan = [0 for _ in range(self.n_agent)]

    def adjacency_matrix_and_norm_edges_features(self, edges_features, next_state=False):

        """
        Computes the adjacency matrix of each agent and the edge features for each of its neighbors.
        Also, this function calculates useful information about the current state, that is the number of flights
        in conflict/alert/loss of separation and the corresponding flight IDs.

        :param edges_features: numpy array as returned by the environment.
        :param next_state: Boolean, it specifies if 'edges_features' are referred to the next or the current state.
        """

        flight_id_and_time_to_cpa_with_positive_time = [[] for _ in range(self.n_agent)]
        flight_id_and_time_to_cpa_with_negative_time = [[] for _ in range(self.n_agent)]
        flight_id_and_dis_between_flights = [[] for _ in range(self.n_agent)]
        flight_id_and_dis_to_cpa_pos_time_without_confl = [[] for _ in range(self.n_agent)]
        flight_id_and_dis_to_cpa_neg_time_without_confl = [[] for _ in range(self.n_agent)]
        flight_id_and_time_to_cpa = [[] for _ in range(self.n_agent)]  # For alerts
        flights_with_conflicts = []
        conflicts_ids_with_negative_tcpa = []
        conflicts_ids_with_positive_tcpa = []
        flights_with_alerts = []
        alerts_ids = []
        flights_with_loss_of_separation = []
        loss_of_separation_ids = []
        count_flights_with_conflicts = [0 for _ in range(self.n_agent)]
        count_flights_with_alerts = [0 for _ in range(self.n_agent)]
        count_flights_with_loss_of_separation = [0 for _ in range(self.n_agent)]
        count_total_conflicts_not_alerts_per_flight = [0 for _ in range(self.n_agent)]
        count_total_alerts_per_flight = [0 for _ in range(self.n_agent)]
        agents_edges_features = [{} for _ in range(self.n_agent)]
        MOC = np.zeros(self.n_agent, dtype=float)
        RoC = np.zeros(self.n_agent, dtype=float)
        rateOfClosureH = np.zeros(self.n_agent, dtype=float)
        rateOfClosureV = np.zeros(self.n_agent, dtype=float)

        # If edges_features is an empty list then this means that there are not edges with conflict/loss of separation
        if isinstance(edges_features, np.ndarray):
            for j_ in range(edges_features.shape[0]):

                # If edge is not referred to a flight and itself
                if edges_features[j_, 11] == 1:

                    # Count flights with loss of separation and/or conflicts
                    if edges_features[j_, 30] == 1:
                        count_flights_with_loss_of_separation[self.env.flight_index[edges_features[j_, 0]]['idx']] += 1
                        if self.env.flight_index[edges_features[j_, 0]]['idx'] not in flights_with_loss_of_separation:
                            flights_with_loss_of_separation.append(self.env.flight_index[edges_features[j_, 0]]['idx'])

                    if edges_features[j_, 31] == 1 and \
                            (-self.t_CPA_threshold <= edges_features[j_, 2] <= self.t_CPA_threshold):
                        count_flights_with_conflicts[self.env.flight_index[edges_features[j_, 0]]['idx']] += 1
                        if self.env.flight_index[edges_features[j_, 0]]['idx'] not in flights_with_conflicts:
                            flights_with_conflicts.append(self.env.flight_index[edges_features[j_, 0]]['idx'])

                    # As alert, we consider a conflict with 0 <= tcpa < 2 minutes.
                    if edges_features[j_, 31] == 1 and 0 <= edges_features[j_, 2] < 120.0:
                        count_flights_with_alerts[self.env.flight_index[edges_features[j_, 0]]['idx']] += 1
                        if self.env.flight_index[edges_features[j_, 0]]['idx'] not in flights_with_alerts:
                            flights_with_alerts.append(self.env.flight_index[edges_features[j_, 0]]['idx'])

                    if edges_features[j_, 32] > 0 and \
                            ((120 <= edges_features[j_, 2] <= self.t_CPA_threshold) or
                             (-self.t_CPA_threshold <= edges_features[j_, 2] < 0)):
                        count_total_conflicts_not_alerts_per_flight[
                            self.env.flight_index[edges_features[j_, 0]]['idx']] += edges_features[j_, 14]

                    if edges_features[j_, 33] > 0:
                        count_total_alerts_per_flight[self.env.flight_index[edges_features[j_, 0]]['idx']] += \
                            edges_features[j_, 33]

                    # If t_first_cp and t_closest_cp are not NaN
                    # (which means that there is a conflict) and there is a conflict or loss,
                    # and 0 <= t_closest_cp <= 600 secs,
                    # we should measure MOC and RoC
                    if not np.isnan(edges_features[j_, 12]) and not np.isnan(edges_features[j_, 27]) and \
                            (edges_features[j_, 31] == 1 or edges_features[j_, 30] == 1) and \
                            0 <= edges_features[j_, 27] <= self.t_CPA_threshold:

                        # Compute MOC based on vertical and horizontal distance of flights at CPA
                        hor_separ_perc = edges_features[j_, 3] / self.horizontal_minimum
                        vert_separ_perc = np.abs(edges_features[j_, 6]) / self.vertical_minimum
                        perc_MOC = hor_separ_perc if hor_separ_perc >= vert_separ_perc else vert_separ_perc
                        rateMOC = 0 if perc_MOC >= 1.0 \
                            else (1 if 1.0 > perc_MOC > 0.75
                                  else (3 if 0.75 >= perc_MOC > 0.5
                                        else (7 if 0.5 >= perc_MOC > 0.25
                                              else 10)))
                        MOC[self.env.flight_index[edges_features[j_, 0]]['idx']] += rateMOC / 10

                        # Compute RoC if t_closest_cp > t_first_cp
                        if not np.isnan(edges_features[j_, 27]) and not np.isnan(edges_features[j_, 12]) and \
                                edges_features[j_, 27] > edges_features[j_, 12]:

                            RoC_value, \
                            rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']], \
                            rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']], \
                            relSpeedV, \
                            relSpeedH = compute_ROC(edges_features, self.max_rateOfClosureHV, j_)
                            RoC[self.env.flight_index[edges_features[j_, 0]]['idx']] += RoC_value

                    ### Store flights and their edges features which are ###
                    ### in conflict/loss of separation with the current flight ###

                    # If edge is referred to flights which are currently in loss of separation
                    if edges_features[j_, 30] == 1:
                        flight_id_and_dis_between_flights[self.env.flight_index[edges_features[j_, 0]]['idx']].\
                            append([np.sqrt(np.square(edges_features[j_, 9] / self.H_dij) +
                                            np.square(edges_features[j_, 10] / self.V_dij)),
                                    self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])
                        loss_of_separation_ids.append((self.env.flight_index[edges_features[j_, 0]]['idx'],
                                                       self.env.flight_index[edges_features[j_, 1]]['idx']))

                    # If edge is referred to flights which are in alert
                    elif edges_features[j_, 31] == 1 and 0 <= edges_features[j_, 2] <= 120:
                        flight_id_and_time_to_cpa[self.env.flight_index[edges_features[j_, 0]]['idx']]. \
                            append([edges_features[j_, 2], self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])
                        alerts_ids.append((self.env.flight_index[edges_features[j_, 0]]['idx'],
                                           self.env.flight_index[edges_features[j_, 1]]['idx']))

                    # If edge is referred to flights which are in conflict and t_cpa >= 120
                    elif edges_features[j_, 31] == 1 and self.t_CPA_threshold >= edges_features[j_, 2] >= 120:
                        flight_id_and_time_to_cpa_with_positive_time[
                            self.env.flight_index[edges_features[j_, 0]]['idx']].append(
                            [edges_features[j_, 2], self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])
                        conflicts_ids_with_positive_tcpa.append((self.env.flight_index[edges_features[j_, 0]]['idx'],
                                                                 self.env.flight_index[edges_features[j_, 1]]['idx']))

                    # If edge is referred to flights which are in conflict and t_cpa < 0
                    elif edges_features[j_, 31] == 1 and -self.t_CPA_threshold <= edges_features[j_, 2] < 0:
                        flight_id_and_time_to_cpa_with_negative_time[
                            self.env.flight_index[edges_features[j_, 0]]['idx']]. \
                            append([edges_features[j_, 2], self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])
                        conflicts_ids_with_negative_tcpa.append((self.env.flight_index[edges_features[j_, 0]]['idx'],
                                                                 self.env.flight_index[edges_features[j_, 1]]['idx']))

                    # If edge is referred to flights which are NOT in conflict and t_cpa >= 0
                    elif edges_features[j_, 30] == 0 and edges_features[j_, 31] == 0 and edges_features[j_, 2] >= 0:
                        flight_id_and_dis_to_cpa_pos_time_without_confl[
                            self.env.flight_index[edges_features[j_, 0]]['idx']].\
                            append([np.sqrt(np.square(edges_features[j_, 3] / self.D_cpa) +
                                            np.square(edges_features[j_, 6] / self.V_dcpa)),
                                    self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])

                    # If edge is referred to flights which are NOT in conflict and t_cpa < 0
                    elif edges_features[j_, 30] == 0 and edges_features[j_, 31] == 0 and edges_features[j_, 2] < 0:
                        flight_id_and_dis_to_cpa_neg_time_without_confl[
                            self.env.flight_index[edges_features[j_, 0]]['idx']].\
                            append([np.sqrt(np.square(edges_features[j_, 3] / self.D_cpa) +
                                            np.square(edges_features[j_, 6] / self.V_dcpa)),
                                    self.env.flight_index[edges_features[j_, 1]]['idx']])
                        agents_edges_features[self.env.flight_index[edges_features[j_, 0]]['idx']][
                            self.env.flight_index[edges_features[j_, 1]]['idx']] = \
                            np.append(edges_features[j_, 2:2 + self.num_edges_features],
                                      [rateOfClosureV[self.env.flight_index[edges_features[j_, 0]]['idx']],
                                       rateOfClosureH[self.env.flight_index[edges_features[j_, 0]]['idx']]])

        dummy_edges_feats = np.zeros(self.num_edges_features + self.num_edges_feats_for_ROC)
        dummy_edges_feats[:] = np.inf
        adj_edges_feat_ = [[] for _ in range(self.n_agent)]

        adj__ = [[] for _ in range(self.n_agent)]
        adj_matrix_ = [[] for _ in range(self.n_agent)]
        adj_one_hot_ = [[] for _ in range(self.n_agent)]

        for j_ in range(self.n_agent):

            flight_id_and_dis_between_flights[j_].sort(key=lambda x: x[0])
            flight_id_and_time_to_cpa_with_positive_time[j_].sort(key=lambda x: x[0])
            flight_id_and_time_to_cpa_with_negative_time[j_].sort(key=lambda x: x[0], reverse=True)
            flight_id_and_dis_to_cpa_pos_time_without_confl[j_].sort(key=lambda x: x[0])
            flight_id_and_dis_to_cpa_neg_time_without_confl[j_].sort(key=lambda x: x[0])
            flight_id_and_time_to_cpa[j_].sort(key=lambda x: x[0])

            count_neighbs_with_dis_between_flights = \
                len(flight_id_and_dis_between_flights[j_]) \
                    if len(flight_id_and_dis_between_flights[j_]) <= self.neighbors_observed \
                    else self.neighbors_observed
            adj__[j_] = [j_] + [flight_id_and_dis_between_flights[j_][i__][1]
                                for i__ in range(count_neighbs_with_dis_between_flights)]

            if len(adj__[j_]) < self.neighbors_observed + 1:
                count_neighbs_with_alerts = \
                    len(flight_id_and_time_to_cpa[j_]) \
                    if len(flight_id_and_time_to_cpa[j_]) + len(adj__[j_]) <= self.neighbors_observed + 1 \
                    else self.neighbors_observed + 1 - len(adj__[j_])
                adj__[j_].extend([flight_id_and_time_to_cpa[j_][i__][1] for i__ in range(count_neighbs_with_alerts)])

                if len(adj__[j_]) < self.neighbors_observed + 1:
                    count_neighbs_with_time_to_cpa_and_positive_tcpa = \
                        len(flight_id_and_time_to_cpa_with_positive_time[j_]) \
                        if len(flight_id_and_time_to_cpa_with_positive_time[j_]) + \
                           len(adj__[j_]) <= self.neighbors_observed + 1 \
                        else self.neighbors_observed + 1 - len(adj__[j_])
                    adj__[j_].extend([flight_id_and_time_to_cpa_with_positive_time[j_][i__][1]
                                      for i__ in range(count_neighbs_with_time_to_cpa_and_positive_tcpa)])

                    if len(adj__[j_]) < self.neighbors_observed + 1:
                        count_neighbs_with_time_to_cpa_and_negative_tcpa = \
                            len(flight_id_and_time_to_cpa_with_negative_time[j_]) \
                            if len(flight_id_and_time_to_cpa_with_negative_time[j_]) + \
                               len(adj__[j_]) <= self.neighbors_observed + 1 \
                            else self.neighbors_observed + 1 - len(adj__[j_])
                        adj__[j_].extend([flight_id_and_time_to_cpa_with_negative_time[j_][i__][1]
                                          for i__ in range(count_neighbs_with_time_to_cpa_and_negative_tcpa)])

                        if len(adj__[j_]) < self.neighbors_observed + 1:
                            count_dummy_neighbs = self.neighbors_observed + 1 - len(adj__[j_])
                            adj__[j_].extend([self.n_agent for _ in range(count_dummy_neighbs)])
                            agents_edges_features[j_][self.n_agent] = dummy_edges_feats

            adj_edges_feat_[j_] = [dummy_edges_feats] + [agents_edges_features[j_][j__] for j__ in adj__[j_][1:]]

            adj_matrix_[j_] = np.asarray(adj__[j_])
            adj_one_hot_[j_] = self.flight_ids_one_hot_np[np.asarray(adj__[j_])]

        norm_adj_edges_feat = \
            np.ones((self.n_agent, self.neighbors_observed + 1,
                     self.num_edges_features + 2 + self.num_edges_feats_for_ROC),
                    dtype=float)
        adj_edges_feat = np.asarray(adj_edges_feat_)

        norm_adj_edges_feat[:, 1:, 0] = \
            np.where(adj_edges_feat[:, 1:, 0] != np.inf, adj_edges_feat[:, 1:, 0] / self.T_cpa,
                     self.mask_edges_features[0])  # t_cpa
        norm_adj_edges_feat[:, 1:, 1] = \
            np.where(adj_edges_feat[:, 1:, 1] != np.inf, adj_edges_feat[:, 1:, 1] / self.D_cpa,
                     self.mask_edges_features[1])  # d_cpa
        with np.errstate(invalid='ignore'):
            norm_adj_edges_feat[:, 1:, 2] = \
                np.where(adj_edges_feat[:, 1:, 2] != np.inf, np.cos(adj_edges_feat[:, 1:, 2]),
                         self.mask_edges_features[2])  # aij
            norm_adj_edges_feat[:, 1:, 3] = \
                np.where(adj_edges_feat[:, 1:, 2] != np.inf, np.sin(adj_edges_feat[:, 1:, 2]),
                         self.mask_edges_features[3])  # aij
            norm_adj_edges_feat[:, 1:, 4] = \
                np.where(adj_edges_feat[:, 1:, 3] != np.inf, np.cos(adj_edges_feat[:, 1:, 3]),
                         self.mask_edges_features[4])  # bij
            norm_adj_edges_feat[:, 1:, 5] = \
                np.where(adj_edges_feat[:, 1:, 3] != np.inf, np.sin(adj_edges_feat[:, 1:, 3]),
                         self.mask_edges_features[5])  # bij
        norm_adj_edges_feat[:, 1:, 6] = \
            np.where(adj_edges_feat[:, 1:, 4] != np.inf, adj_edges_feat[:, 1:, 4] / self.V_dcpa,
                     self.mask_edges_features[6])  # v_dcpa
        norm_adj_edges_feat[:, 1:, 7] = \
            np.where(adj_edges_feat[:, 1:, 5] != np.inf, adj_edges_feat[:, 1:, 5] / self.D_cp,
                     self.mask_edges_features[7])  # d_cp
        norm_adj_edges_feat[:, 1:, 8] = \
            np.where(adj_edges_feat[:, 1:, 6] != np.inf, adj_edges_feat[:, 1:, 6] / self.T_cp,
                     self.mask_edges_features[8])  # t_cp
        norm_adj_edges_feat[:, 1:, 9] = \
            np.where(adj_edges_feat[:, 1:, 7] != np.inf, adj_edges_feat[:, 1:, 7] / self.H_dij,
                     self.mask_edges_features[9])  # h_dij
        norm_adj_edges_feat[:, 1:, 10] = \
            np.where(adj_edges_feat[:, 1:, 8] != np.inf, adj_edges_feat[:, 1:, 8] / self.V_dij,
                     self.mask_edges_features[10])  # v_dij
        norm_adj_edges_feat[:, 1:, 11] = \
            np.where(adj_edges_feat[:, 1:, 9] != np.inf, adj_edges_feat[:, 1:, 9] / self.max_rateOfClosureHV,
                     self.mask_edges_features[11])  # rateOfClosureV
        norm_adj_edges_feat[:, 1:, 12] = \
            np.where(adj_edges_feat[:, 1:, 10] != np.inf, adj_edges_feat[:, 1:, 10] / self.max_rateOfClosureHV,
                     self.mask_edges_features[12])  # rateOfClosureH

        # Set NaN values of tcp and dcp equal to 20
        norm_adj_edges_feat[:, 1:, 7] = \
            np.where(np.isnan(norm_adj_edges_feat[:, 1:, 7]), 20.0, norm_adj_edges_feat[:, 1:, 7])
        norm_adj_edges_feat[:, 1:, 8] = \
            np.where(np.isnan(norm_adj_edges_feat[:, 1:, 8]), 20.0, norm_adj_edges_feat[:, 1:, 8])

        # Block normalized tcp and dcp to be in range [-20, 20]
        norm_adj_edges_feat[:, 1:, 7] = \
            np.where(norm_adj_edges_feat[:, 1:, 7] > 20.0, 20.0, norm_adj_edges_feat[:, 1:, 7])
        norm_adj_edges_feat[:, 1:, 7] = \
            np.where(norm_adj_edges_feat[:, 1:, 7] < -20.0, -20.0, norm_adj_edges_feat[:, 1:, 7])
        norm_adj_edges_feat[:, 1:, 8] = \
            np.where(norm_adj_edges_feat[:, 1:, 8] > 20.0, 20.0, norm_adj_edges_feat[:, 1:, 8])
        norm_adj_edges_feat[:, 1:, 8] = \
            np.where(norm_adj_edges_feat[:, 1:, 8] < -20.0, -20.0, norm_adj_edges_feat[:, 1:, 8])

        if not next_state:
            self.adjacency_one_hot = adj_one_hot_
            self.adjacency_matrix = adj_matrix_
            self.norm_edges_feats = norm_adj_edges_feat
            self.fls_with_loss_of_separation = flights_with_loss_of_separation
            self.fls_with_conflicts = flights_with_conflicts
            self.fls_with_alerts = flights_with_alerts
            self.count_fls_with_loss_of_separation = count_flights_with_loss_of_separation
            self.count_fls_with_conflicts = count_flights_with_conflicts
            self.count_fls_with_alerts = count_flights_with_alerts
            self.count_total_conflicts_not_alerts_per_flt = count_total_conflicts_not_alerts_per_flight
            self.count_total_alerts_per_flt = count_total_alerts_per_flight
            self.loss_ids = loss_of_separation_ids
            self.alrts_ids = alerts_ids
            self.confls_ids_with_positive_tcpa = conflicts_ids_with_positive_tcpa
            self.confls_ids_with_negative_tcpa = conflicts_ids_with_negative_tcpa
            self.MOC = MOC
            self.RoC = RoC

        else:
            self.next_adjacency_one_hot = adj_one_hot_
            self.next_adjacency_matrix = adj_matrix_
            self.next_norm_edges_feats = norm_adj_edges_feat
            self.next_fls_with_loss_of_separation = flights_with_loss_of_separation
            self.next_fls_with_conflicts = flights_with_conflicts
            self.next_fls_with_alerts = flights_with_alerts
            self.count_next_fls_with_loss_of_separation = count_flights_with_loss_of_separation
            self.count_next_fls_with_conflicts = count_flights_with_conflicts
            self.count_next_fls_with_alerts = count_flights_with_alerts
            self.count_next_total_conflicts_not_alerts_per_flt = count_total_conflicts_not_alerts_per_flight
            self.count_next_total_alerts_per_flt = count_total_alerts_per_flight
            self.next_loss_ids = loss_of_separation_ids
            self.next_alrts_ids = alerts_ids
            self.next_confls_ids_with_positive_tcpa = conflicts_ids_with_positive_tcpa
            self.next_confls_ids_with_negative_tcpa = conflicts_ids_with_negative_tcpa
            self.next_MOC = MOC
            self.next_RoC = RoC

        """
        #Check if any edge-feature of any flight is out of the range [-20,20]

        for agg in range(self.n_agent):

            if self.active_flights_mask[agg] and ((norm_adj_edges_feat[agg] > 20).any() or (norm_adj_edges_feat[agg] < -20).any()):
                print('\nnorm_adj_edges_feat : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                print(norm_adj_edges_feat[agg])
                #print("\n alert list: {}".format(flight_id_and_time_to_cpa[agg]))
                #print("\n conflict with pos time: {}".format(flight_id_and_time_to_cpa_with_positive_time[agg]))
                #print("\n conflict with neg time: {}".format(flight_id_and_time_to_cpa_with_negative_time[agg]))
                #print('\nflight_id_and_dis_between_flights : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(flight_id_and_dis_between_flights[agg])
                #print('\nflight_id_and_dis_to_cpa_with_positive_time : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(flight_id_and_dis_to_cpa_with_positive_time[agg])
                #print('\nflight_id_and_dis_to_cpa_with_negative_time : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(flight_id_and_dis_to_cpa_with_negative_time[agg])
                #print('\nflight_id_and_dis_to_cpa_pos_time_without_confl : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(flight_id_and_dis_to_cpa_pos_time_without_confl[agg])
                #print('\nflight_id_and_dis_to_cpa_neg_time_without_confl : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(flight_id_and_dis_to_cpa_neg_time_without_confl[agg])
                #print('\nagents_edges_features : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(agents_edges_features[agg])
                #print('\nadj_matrix : ' + 'Agent ' + str(agg) + ' with id: ' + str(self.flights_idxs_to_ids[agg]))
                #print(adj_matrix_[agg])

        """

    def run_episode_for_current_env(self):

        while self.done is False:
            self.steps += 1

            # Monitor the flights which are in conflict/alert/loss of separation
            self.update_history_losses_conflicts()

            # Create the input of the model
            self.obs = [np.asarray([self.flight_array]), np.asarray([self.adjacency_one_hot]),
                        np.asarray([self.norm_edges_feats])]

            # Get the outputs of the model
            self.get_model_outputs()

            # Assign an action to each agent based on the value of randomness (alpha) and the outputs of the model
            self.actions = np.zeros(self.n_agent, dtype=np.int32)
            self.rand_choice = [False for _ in range(self.n_agent)]
            for j in range(self.n_agent):
                random_number = np.random.rand()
                if random_number < self.alpha:
                    self.rand_choice[j] = True
                    self.actions[j] = random.randrange(self.n_actions)
                else:
                    self.rand_choice[j] = False
                    self.actions[j] = np.argmax(self.acts[0][j])

            # Filter the actions
            self.get_valid_actions()

            # Convert the duration of actions according to the requirements of the environment
            self.fix_duration_to_be_passed_to_env()

            # Convert the actions to the form expected by the environment
            self.true_acts = self.get_true_actions(self.actions,
                                                   self.n_agent,
                                                   self.active_flights_mask,
                                                   dur_vals=self.actions_duration_to_be_passed_to_env)

            # Update the dataframes which are going to be written
            # to log files (when running in evaluation mode).
            # Also, update the resolution action IDs that will be passed to the environment
            # (when running in training mode).
            self.update_dataframes_and_actionsID()

            # Update files for debugging (only when running in debug_ mode)
            self.update_debugging_files()

            # Execute an environment step and get the returns
            self.env_step_and_get_returns()

            # Get the normalized reward
            self.compute_norm_reward()

            # Update statistics of the episode for the current environment, that is,
            # score, number of ATC instructions, additional nautical miles, number of conflicts with positive tcpa,
            # and additional duration.
            self.update_episode_stats_for_current_env()

            # Store samples for training (only in training mode)
            self.sampling()

            # Store statistics (only when 'done' is True) of the episode for the current environment, that is,
            # number of steps, score, number of ATC instructions, additional nautical miles,
            # number of conflicts with positive tcpa, number of alerts, number of losses of separation,
            # and additional duration.
            self.store_episode_stats_for_current_env()

            # Copy values of next state arrays/lists to current state arrays/lists
            self.update_arrays_and_lists_from_current_state_to_next()

            # Write and store the log files of the episode only for the current environment
            self.write_log_files_of_episode_for_current_env()

    def update_history_losses_conflicts(self):
        temp_history_loss = [ag__ for ag__ in self.fls_with_loss_of_separation if ag__ not in self.history_loss_confl]
        temp_history_conf = [ag__ for ag__ in self.fls_with_conflicts
                             if ag__ not in self.history_loss_confl and ag__ not in temp_history_loss]
        self.history_loss_confl += temp_history_loss + temp_history_conf

    def get_model_outputs(self):

        if not self.evaluation:
            self.unfixed_acts = np.array(self.DGN_m.model_predict(self.obs))

            # Get all outputs except the last two which are the "continue" action and "resume to fplan" action
            self.acts = self.unfixed_acts[:, :, :self.end_action_array]

        else:
            model_outputs = self.DGN_m.model_predict(self.obs)

            # unfixed_acts shape: [batch, agent, n_actions+plus_model]
            # att_all shape: [agent, layer, head, neighbors+1]
            self.unfixed_acts, self.att_all = model_outputs[0], model_outputs[1][0]

            # Get all outputs except the last two which are the "continue" action and "resume to fplan" action
            self.acts = np.array(self.unfixed_acts)[:, :, :self.end_action_array]

            # Duration of actions should be copied here because in store_resolution_actions function,
            # valid actions are recalculated, and therefore we should not pass the updated durations.
            self.durations_of_actions_for_evaluation_ = \
                [tuple(list(self.durations_of_actions[i_agent]).copy()) for i_agent in range(self.n_agent)]

    def get_valid_actions(self):

        self.dict_agents_actions = {}
        self.cur_num_ATC_instr = 0
        self.end_of_maneuver = [True for _ in range(self.actions.shape[0])]

        # Mask to monitor if a flight is in 'climbing'/'descending' phase and
        # is executing the 'resume to fplan' action
        # while it participates in conflicts/losses of separation.
        mask_climb_descend_res_fplan_with_confl_loss = [False for _ in range(self.actions.shape[0])]

        # Mask to monitor if a flight is executing the action 'direct to next way point'
        # after having executed a maneuver.
        mask_res_fplan_after_maneuv = [False for _ in range(self.actions.shape[0])]

        # List to keep the duration for the actions 'direct to (any) way point' and 'change FL' in order to be able
        # to update these actions as well as 'continue' action for the right interval.
        # In the case of deterministic action 'direct to next way point', we keep the duration only when
        # mask_climb_descend_res_fplan_with_confl_loss or mask_res_fplan_after_maneuv is True.
        self.duration_dir_to_wp_ch_FL_res_fplan = [0 for _ in range(self.actions.shape[0])]

        resolutionActionID_only_for_train_ = \
            [[str(int(self.timestamp)) + "_" + str(self.flights_idxs_to_ids[j_]), False]
             for j_ in range(self.actions.shape[0])]
        additional_nautical_miles = [0 for _ in range(self.actions.shape[0])]
        additional_duration = [0 for _ in range(self.actions.shape[0])]

        for agent in range(self.actions.shape[0]):
            if not self.active_flights_mask[agent]:

                # num_dc+num_ds+num_as+num_wp means "zero" action
                self.actions[agent] = self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': self.actions_list[self.actions[agent]],
                     'action_type': self.types_of_actions[self.actions[agent]]}
                self.durations_of_actions[agent] = (0, 0)  # Reset the duration
                resolutionActionID_only_for_train_[agent][0] += \
                    "_" + self.types_of_actions[self.actions[agent]] + \
                    "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                    "_" + str(self.interval_between_two_steps)
                continue

            elif not (agent in self.history_loss_confl):

                # Plus 3 means the default action (follow flight plan)
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 3

                # 'null' in this case means the default action (follow flight plan)
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent], 'true_action': "null",
                     "action_type": self.types_of_actions[self.actions[agent]]}

                self.durations_of_actions[agent] = (0, 0)  # Reset the duration
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                continue

            # We assume that the flight, which is in 'climbing' or 'descending' flight phase,
            # is already in 'history_loss_confl',
            # otherwise the previous condition would be true.
            # IF the phase of a flight is 'climbing'/'descending' and there is no action in progress
            # (based on (durations_of_actions[agent][0] == 0 and not executing_resume_to_fplan[agent])
            # for previous deterministic execution of the action 'resume to fplan',
            # and
            # (durations_of_actions[agent][0] == np.inf and not executing_direct_to[agent]
            # and not executing_FL_change[agent])
            # for previous non-deterministic execution of actions
            # 'direct to (any) way point'/'change FL'),
            # OR the selected action has just finished
            # (based on timestamp - durations_of_actions[agent][1] >= durations_of_actions[agent][0]),
            # THEN the action "resume to fplan" should be executed (EVEN IF THERE IS CONFLICT/LOSS OF SEPARATION).
            elif (self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending') and \
                    (((self.durations_of_actions[agent][0] == 0 or
                       self.timestamp - self.durations_of_actions[agent][1] >= self.durations_of_actions[agent][0])
                      and not self.executing_resume_to_fplan[agent])
                     or (self.durations_of_actions[agent][0] == np.inf and not self.executing_direct_to[agent]
                         and not self.executing_FL_change[agent])):

                # num_dc+num_ds+num_as+num_wp+2 means "resume to fplan" action
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2

                # 'null' in this case means # 'resume to fplan'.
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': "null",
                     'action_type': self.types_of_actions[self.actions[agent]]}

                self.durations_of_actions[agent] = (0, 0)  # Reset the duration
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                if agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts:
                    mask_climb_descend_res_fplan_with_confl_loss[agent] = True
                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = self.interval_between_two_steps

                # If at the previous step the maneuver had not stopped and the duration of its action was not 0 or inf,
                # then this means that the executed action was not the 'direct to next way point'
                # neither the 'resume fplan'.
                elif (agent in self.previous_fls_with_loss_of_separation or
                      agent in self.previous_fls_with_conflicts) or \
                        (not self.previous_end_of_maneuver[agent] and self.durations_of_actions[agent][0] != 0 and
                         self.durations_of_actions[agent][0] != np.inf):
                    mask_res_fplan_after_maneuv[agent] = True
                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = self.interval_between_two_steps

                continue

            # If the phase of a flight is 'climbing'/'descending' and the action which is in progress is
            # the 'resume_to_fplan',
            # (based on durations_of_actions[agent][0] == 0 and executing_resume_to_fplan[agent]),
            # then the 'continue' action should be selected.
            elif (self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending') and \
                    self.durations_of_actions[agent][0] == 0 and self.executing_resume_to_fplan[agent]:

                # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1

                # 'null' in this case is the 'continue' action.
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': "null",
                     'action_type': self.types_of_actions[self.actions[agent]]}

                self.end_of_maneuver[agent] = False  # This is the first case that a maneuver has not ended yet.

                # Reset the duration in order to be able to distinguish the continuation of
                # the deterministic action 'resume fplan' (assign zero duration)
                # from the non-deterministic actions (assign inf duration),
                # based on the duration value.
                self.durations_of_actions[agent] = (0, 0)
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                if self.previous_np_mask_res_fplan_after_maneuv[agent]:
                    mask_res_fplan_after_maneuv[agent] = True
                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = \
                        self.previous_duration_dir_to_wp_ch_FL_res_fplan[agent] + self.interval_between_two_steps
                if (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) or \
                        self.previous_np_mask_climb_descend_res_fplan_with_confl_loss[agent]:
                    mask_climb_descend_res_fplan_with_confl_loss[agent] = True
                    if not mask_res_fplan_after_maneuv[agent]:
                        self.duration_dir_to_wp_ch_FL_res_fplan[agent] = \
                            self.previous_duration_dir_to_wp_ch_FL_res_fplan[agent] + self.interval_between_two_steps
                if self.executing_resume_to_fplan[agent] and self.finished_resume_to_fplan[agent]:
                    print("executing_resume_to_fplan and finished_resume_to_fplan are not properly synchronized!!! "
                          "Case 1!!")
                    exit(0)
                continue

            # We assume that the phase is not 'climbing'/'descending', otherwise the previous condition would be True.
            # If a flight is executing the action 'resume to fplan', and it does not participate in a
            # conflict/loss of separation, then the action 'continue' should be executed.
            elif self.executing_resume_to_fplan[agent] and self.durations_of_actions[agent][0] == 0 and \
                    not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts):

                # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1

                # 'null' in this case is the 'continue' action.
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': "null",
                     'action_type': self.types_of_actions[self.actions[agent]]}
                self.end_of_maneuver[agent] = False  # This is the second case that a maneuver has not ended yet.

                # Reset the duration in order to be able to distinguish the case of executing the action
                # 'direct to next way point' deterministically (assign zero duration)
                # from the case that this action was selected by the agent (assign inf duration).
                self.durations_of_actions[agent] = (0, 0)
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                if self.previous_np_mask_climb_descend_res_fplan_with_confl_loss[agent]:
                    mask_climb_descend_res_fplan_with_confl_loss[agent] = True
                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = \
                        self.previous_duration_dir_to_wp_ch_FL_res_fplan[agent] + self.interval_between_two_steps
                if self.previous_np_mask_res_fplan_after_maneuv[agent]:
                    mask_res_fplan_after_maneuv[agent] = True
                    if not self.previous_np_mask_climb_descend_res_fplan_with_confl_loss[agent]:
                        self.duration_dir_to_wp_ch_FL_res_fplan[agent] = \
                            self.previous_duration_dir_to_wp_ch_FL_res_fplan[agent] + self.interval_between_two_steps
                if self.executing_resume_to_fplan[agent] and self.finished_resume_to_fplan[agent]:
                    print("executing_resume_to_fplan and finished_resume_to_fplan are not properly synchronized!!! "
                          "Case 1!!")
                    exit(0)
                continue

            # We assume that even if the phase is 'climbing' or 'descending', we should not interrupt the
            # executing action
            # (which is not selected deterministically but by the agent. The cases of
            # the deterministic selection are covered by the second and third previous conditions).
            # The executing actions that we care about in this condition is the "change flight level" and
            # "direct to (any) way point".
            elif (self.executing_FL_change[agent] or self.executing_direct_to[agent]) and \
                    self.durations_of_actions[agent][0] == np.inf:

                # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1

                # 'null' in this case means the 'continue' action.
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': "null",
                     'action_type': self.types_of_actions[self.actions[agent]]}
                self.end_of_maneuver[agent] = False  # This is the third case that a maneuver has not ended yet.
                self.duration_dir_to_wp_ch_FL_res_fplan[agent] = \
                    self.previous_duration_dir_to_wp_ch_FL_res_fplan[agent] + self.interval_between_two_steps
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                if (self.executing_direct_to[agent] and self.finished_direct_to[agent]) or \
                        (self.executing_FL_change[agent] and self.finished_FL_change[agent]):
                    print("self.executing_direct_to/self.executing_FL_change and "
                          "self.finished_direct_to/self.finished_FL_change are not properly synchronized!!!")
                    exit(0)

                continue

            # Conditions for checking if all cases have been taken into account
            elif (self.executing_FL_change[agent] or self.executing_direct_to[agent] or
                  self.executing_resume_to_fplan[agent]) \
                    and not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts):
                print("There are cases of executing_FL_change/executing_direct_to/executing_resume_to_fplan "
                      "while there is no conflict/loss of separation, that should be taken into account!!")
                print("executing_FL_change {}".format(self.executing_FL_change[agent]))
                print("executing_direct_to {}".format(self.executing_direct_to[agent]))
                print("executing_resume_to_fplan {}".format(self.executing_resume_to_fplan[agent]))
                exit(0)

            # We assume that there is no action "direct to (any) way point" or "change flight level" or
            # "resume to fplan" in progress,
            # otherwise the previous condition would be true.
            # If the maneuver has just finished and there is no conflict/loss of separation,
            # the default action "resume to fplan" should be executed.
            elif not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) and \
                    (agent in self.history_loss_confl) and \
                    ((self.timestamp - self.durations_of_actions[agent][1] >= self.durations_of_actions[agent][0]) or
                     (self.durations_of_actions[agent][0] == np.inf and not self.executing_FL_change[agent] and
                      not self.executing_direct_to[agent]) or
                     (self.durations_of_actions[agent][0] == 0 and not self.executing_resume_to_fplan[agent])):

                # num_dc+num_ds+num_as+num_wp+2 means the default action (resume to fplan)
                self.actions[agent] = \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': "null",
                     'action_type': self.types_of_actions[self.actions[agent]]}
                if self.timestamp - self.durations_of_actions[agent][1] > self.durations_of_actions[agent][0] and \
                        self.durations_of_actions[agent][1] != 0:
                    print("A problem with duration resetting might exist, case 1!!")
                    exit(0)

                # If at the previous step the maneuver had not stopped and the duration of its action was not 0 or inf,
                # then this means that the executed action was not the 'direct to next way point'
                # neither in the case deterministically selected nor in the case of being selected by the agent.
                if (agent in self.previous_fls_with_loss_of_separation or
                    agent in self.previous_fls_with_conflicts) or \
                        (not self.previous_end_of_maneuver[agent] and
                         self.durations_of_actions[agent][0] != 0 and
                         self.durations_of_actions[agent][0] != np.inf):
                    mask_res_fplan_after_maneuv[agent] = True
                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = self.interval_between_two_steps

                self.durations_of_actions[agent] = (0, 0)  # Reset the duration
                resolutionActionID_only_for_train_[agent][0] += "_" + self.types_of_actions[self.actions[agent]]
                continue

            # When the duration of the selected action
            # (not for the actions "direct to (any) way point" and "change flight level")
            # of a flight has not been reached,
            # the corresponding flight should execute the "zero" action,
            # even if this flight participates in conflict/loss of separation or not (but it is in history_loss_confl).
            elif (agent in self.history_loss_confl) and \
                    (self.timestamp - self.durations_of_actions[agent][1] < self.durations_of_actions[agent][0]) \
                    and self.durations_of_actions[agent][0] != np.inf \
                    and not (self.executing_FL_change[agent] or self.executing_direct_to[agent]):

                # num_dc+num_ds+num_as+num_wp means "zero" action
                self.actions[agent] = self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp
                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': self.actions_list[self.actions[agent]],
                     'action_type': self.types_of_actions[self.actions[agent]]}
                self.end_of_maneuver[agent] = False  # This is the forth case that a maneuver has not ended yet.
                resolutionActionID_only_for_train_[agent][0] += \
                    "_" + self.types_of_actions[self.actions[agent]] + \
                    "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                    "_" + str(self.interval_between_two_steps)
                continue

            # Conditions for checking if all cases have been taken into account
            elif not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) \
                    and not (agent in self.history_loss_confl):
                print("The action 'follow flight plan' should have been selected, but it has not!!")
                exit(0)
            elif self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending':
                print("The action 'direct to next way point' should have been selected!!!")
                exit(0)

            # IF a flight is in conflict/loss and:
            #   - there is no non-deterministic action (apart from 'direct to (any) way point'/'change flight level')
            #     in progress
            #     (based on the condition: timestamp - durations_of_actions[agent][1] >=
            #     durations_of_actions[agent][0]),
            #     and there is no action 'direct to (any) way point' or 'change flight level' in progress
            #     (based on the condition:
            #     not (self.executing_FL_change[agent] or self.executing_direct_to[agent])),
            #   OR
            #   - a flight is executing the deterministic action 'resume fplan'
            #     (based on the condition:
            #     (durations_of_actions[agent][0] == 0 and
            #     self.executing_resume_to_fplan[agent]) and
            #     (agent in flights_with_loss_of_separation or agent in flights_with_conflicts))
            # --> THEN a valid action should be chosen.
            # NOTE that we assume that the phase is not climbing/descending (due to the previous checking condition).
            elif (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) and \
                    (((self.timestamp - self.durations_of_actions[agent][1] >= self.durations_of_actions[agent][0]) and
                      not (self.executing_FL_change[agent] or self.executing_direct_to[agent])) or
                     (self.durations_of_actions[agent][0] == np.inf and not self.executing_FL_change[agent] and
                      not self.executing_direct_to[agent]) or
                     (self.durations_of_actions[agent][0] == 0 and self.executing_resume_to_fplan[agent])):

                # Reset the duration and its timestamp if the duration period has expired
                if (self.timestamp - self.durations_of_actions[agent][1] > self.durations_of_actions[agent][0]) and \
                        self.durations_of_actions[agent][1] != 0 and self.durations_of_actions[agent][0] != np.inf:
                    print("A problem with duration resetting might exist, case 2!!")
                    exit(0)
                self.durations_of_actions[agent] = (0, 0)

                validity_flag = True

                if self.actions[agent] < self.num_actions_dc:
                    self.durations_of_actions[agent] = \
                        (self.total_duration_values[self.actions[agent]], self.timestamp)
                    resolutionActionID_only_for_train_[agent][0] += \
                        "_" + self.types_of_actions[self.actions[agent]] + \
                        "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                        "_" + str(self.durations_of_actions[agent][0])
                    resolutionActionID_only_for_train_[agent][1] = True
                    temp_true_action = \
                        self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]],
                                              np.expand_dims(np.asarray([self.durations_of_actions[agent][0]]),
                                                             axis=1))[0]
                    temp_additional_nautical_miles_and_duration = \
                        self.env.compute_additional_nautical_miles_course_change(self.flights_idxs_to_ids[agent],
                                                                                 temp_true_action)
                    additional_nautical_miles[agent] = temp_additional_nautical_miles_and_duration[0]
                    additional_duration[agent] = temp_additional_nautical_miles_and_duration[1]

                elif self.num_actions_dc <= self.actions[agent] < self.num_actions_dc + self.num_actions_ds:
                    if (((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) + self.min_h_speed) +
                        self.actions_list[self.actions[agent]] < self.min_h_speed) or \
                            (((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) +
                              self.min_h_speed) + self.actions_list[self.actions[agent]] > self.max_h_speed):
                        validity_flag = False
                    else:
                        self.durations_of_actions[agent] = \
                            (self.total_duration_values[self.actions[agent]], self.timestamp)
                        resolutionActionID_only_for_train_[agent][0] += \
                            "_" + self.types_of_actions[self.actions[agent]] + \
                            "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                            "_" + str(self.durations_of_actions[agent][0])
                        resolutionActionID_only_for_train_[agent][1] = True
                        temp_true_action = \
                            self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]],
                                                  np.expand_dims(np.asarray([self.durations_of_actions[agent][0]]),
                                                                 axis=1))[0]
                        additional_duration[agent] = \
                            self.env.compute_additional_duration_speed_change(self.flights_idxs_to_ids[agent],
                                                                              temp_true_action)

                elif self.num_actions_dc + self.num_actions_ds <= self.actions[agent] < \
                        self.num_actions_dc + self.num_actions_ds + self.num_actions_as:

                    if (((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                         self.min_alt_speed) + self.actions_list[self.actions[agent]] < self.min_alt_speed) or \
                            (((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                              self.min_alt_speed) + self.actions_list[self.actions[agent]] > self.max_alt_speed):
                        validity_flag = False
                    else:

                        # The duration of the action 'change flight level' is not known,
                        # and it depends on the environment.
                        self.durations_of_actions[agent] = (np.inf, self.timestamp)

                        # Duration is 0 in resolutionActionID when duration is unknown
                        resolutionActionID_only_for_train_[agent][0] += \
                            "_" + self.types_of_actions[self.actions[agent]] + \
                            "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + "_0"
                        resolutionActionID_only_for_train_[agent][1] = True
                        temp_true_action = \
                            self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]])[0]
                        additional_duration[agent] = \
                            self.env.compute_additional_duration_FL_change(self.flights_idxs_to_ids[agent],
                                                                           temp_true_action)

                elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= self.actions[agent] < \
                        self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                    way_point_index = \
                        abs(((self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp) -
                             self.actions[agent]) - self.num_dir_wp)
                    if self.available_wps[agent, way_point_index] == 0:
                        validity_flag = False
                    else:

                        # The duration of the action 'direct to (any) way point' is not known,
                        # and it depends on the environment.
                        # Additionally, by assigning 'inf' duration value we will be able to distinguish when
                        # this kind of action is selected by the agent or deterministically.
                        self.durations_of_actions[agent] = (np.inf, self.timestamp)

                        # Duration is 0 in resolutionActionID when duration is unknown
                        resolutionActionID_only_for_train_[agent][0] += \
                            "_" + self.types_of_actions[self.actions[agent]] + \
                            "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + "_0"
                        resolutionActionID_only_for_train_[agent][1] = True
                        temp_true_action = \
                            self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]])[0]
                        temp_additional_nautical_miles_and_duration = \
                            self.env.compute_additional_nautical_miles_direct_to(self.flights_idxs_to_ids[agent],
                                                                                 temp_true_action)
                        additional_nautical_miles[agent] = temp_additional_nautical_miles_and_duration[0]
                        additional_duration[agent] = temp_additional_nautical_miles_and_duration[1]

                elif self.actions[agent] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + \
                        self.num_dir_wp:

                    # Zero action has duration equal to the step duration.
                    self.durations_of_actions[agent] = (self.interval_between_two_steps, self.timestamp)
                    resolutionActionID_only_for_train_[agent][0] += \
                        "_" + self.types_of_actions[self.actions[agent]] + \
                        "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                        "_" + str(self.durations_of_actions[agent][0])
                    resolutionActionID_only_for_train_[agent][1] = True

                else:
                    print("There is at least one case that has not taken into account in action selection, case 1!!!")
                    exit(0)

                if validity_flag is False:
                    count_tested_action = 1
                    while validity_flag is False:
                        if count_tested_action > len(self.actions_list) and self.rand_choice[agent] is False:
                            print("All possible actions were tried for agent {}, but all are invalid!!!".format(agent))
                            exit(0)
                        if self.rand_choice[agent]:
                            self.actions[agent] = random.randrange(self.n_actions)
                        else:
                            if count_tested_action == 1:
                                # Use -1 to inverse sorting (that is descending)
                                sorted_actions = np.argsort(-1 * self.acts[0][agent])
                            self.actions[agent] = sorted_actions[count_tested_action]

                        count_tested_action += 1
                        validity_flag = True

                        if self.actions[agent] < self.num_actions_dc:
                            self.durations_of_actions[agent] = \
                                (self.total_duration_values[self.actions[agent]], self.timestamp)
                            resolutionActionID_only_for_train_[agent][0] += \
                                "_" + self.types_of_actions[self.actions[agent]] + \
                                "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                                "_" + str(self.durations_of_actions[agent][0])
                            resolutionActionID_only_for_train_[agent][1] = True
                            temp_true_action = \
                                self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]],
                                                      np.expand_dims(np.asarray([self.durations_of_actions[agent][0]]),
                                                                     axis=1))[0]
                            temp_additional_nautical_miles_and_duration = \
                                self.env.compute_additional_nautical_miles_course_change(self.flights_idxs_to_ids[agent],
                                                                                         temp_true_action)
                            additional_nautical_miles[agent] = temp_additional_nautical_miles_and_duration[0]
                            additional_duration[agent] = temp_additional_nautical_miles_and_duration[1]

                        elif self.num_actions_dc <= self.actions[agent] < self.num_actions_dc + self.num_actions_ds:

                            if (((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) +
                                 self.min_h_speed) + self.actions_list[self.actions[agent]] < self.min_h_speed) or \
                                    (((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) +
                                      self.min_h_speed) + self.actions_list[self.actions[agent]] > self.max_h_speed):
                                validity_flag = False
                            else:
                                self.durations_of_actions[agent] = \
                                    (self.total_duration_values[self.actions[agent]], self.timestamp)
                                resolutionActionID_only_for_train_[agent][0] += \
                                    "_" + self.types_of_actions[self.actions[agent]] + \
                                    "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                                    "_" + str(self.durations_of_actions[agent][0])
                                resolutionActionID_only_for_train_[agent][1] = True
                                temp_true_action = \
                                    self.get_true_actions([self.actions[agent]], 1, [self.active_flights_mask[agent]],
                                                          np.expand_dims(np.asarray(
                                                              [self.durations_of_actions[agent][0]]), axis=1))[0]
                                additional_duration[agent] = \
                                    self.env.compute_additional_duration_speed_change(self.flights_idxs_to_ids[agent],
                                                                                      temp_true_action)

                        elif self.num_actions_dc + self.num_actions_ds <= self.actions[agent] < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as:

                            if (((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                                 self.min_alt_speed) + self.actions_list[self.actions[agent]] < self.min_alt_speed) or \
                                    (((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                                      self.min_alt_speed) +
                                     self.actions_list[self.actions[agent]] > self.max_alt_speed):
                                validity_flag = False
                            else:
                                self.durations_of_actions[agent] = (np.inf, self.timestamp)
                                # Duration is 0 in resolutionActionID when duration is unknown
                                resolutionActionID_only_for_train_[agent][0] += \
                                    "_" + self.types_of_actions[self.actions[agent]] + \
                                    "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + "_0"
                                resolutionActionID_only_for_train_[agent][1] = True
                                temp_true_action = \
                                    self.get_true_actions([self.actions[agent]], 1,
                                                          [self.active_flights_mask[agent]])[0]
                                additional_duration[agent] = \
                                    self.env.compute_additional_duration_FL_change(self.flights_idxs_to_ids[agent],
                                                                                   temp_true_action)

                        elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= self.actions[agent] < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                            way_point_index = \
                                abs(((self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp) -
                                     self.actions[agent]) - self.num_dir_wp)
                            if self.available_wps[agent, way_point_index] == 0:
                                validity_flag = False
                            else:
                                self.durations_of_actions[agent] = (np.inf, self.timestamp)
                                # Duration is 0 in resolutionActionID when duration is unknown
                                resolutionActionID_only_for_train_[agent][0] += \
                                    "_" + self.types_of_actions[self.actions[agent]] + \
                                    "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + "_0"
                                resolutionActionID_only_for_train_[agent][1] = True
                                temp_true_action = \
                                    self.get_true_actions([self.actions[agent]], 1,
                                                          [self.active_flights_mask[agent]])[0]
                                temp_additional_nautical_miles_and_duration = \
                                    self.env.compute_additional_nautical_miles_direct_to(self.flights_idxs_to_ids[agent],
                                                                                         temp_true_action)
                                additional_nautical_miles[agent] = temp_additional_nautical_miles_and_duration[0]
                                additional_duration[agent] = temp_additional_nautical_miles_and_duration[1]

                        elif self.actions[agent] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + \
                                self.num_dir_wp:

                            # Zero action has duration equal to the step duration.
                            self.durations_of_actions[agent] = \
                                (self.interval_between_two_steps, self.timestamp)
                            resolutionActionID_only_for_train_[agent][0] += \
                                "_" + self.types_of_actions[self.actions[agent]] + \
                                "_" + str(self.action_values_for_evaluation[self.actions[agent]]) + \
                                "_" + str(self.durations_of_actions[agent][0])
                            resolutionActionID_only_for_train_[agent][1] = True

                        else:
                            print("There is at least one case that has not taken into account "
                                  "in action selection, case 1!!!")
                            exit(0)

                # If the selected action is 'direct to (any) way point' or 'change FL' or 'resume fplan',
                # then we should store the duration.
                if (self.num_actions_dc + self.num_actions_ds <= self.actions[agent] <
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp) or \
                        (self.actions[agent] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as +
                         self.num_dir_wp + 2):

                    self.duration_dir_to_wp_ch_FL_res_fplan[agent] = self.interval_between_two_steps

                self.dict_agents_actions[self.flights_idxs_to_ids[agent]] = \
                    {'action_id': self.actions[agent],
                     'true_action': self.actions_list[self.actions[agent]],
                     'action_type': self.types_of_actions[self.actions[agent]]}

                # "Zero" action is not considered as an action of the controller.
                # The same is true for the default actions ('resume to fplan', 'continue' and 'follow plan' actions)
                # because they are deterministic actions which are applied when the corresponding flights
                # are not in conflicts/losses of separation.
                if self.actions[agent] != self.num_actions_dc + self.num_actions_ds + self.num_actions_as + \
                        self.num_dir_wp:
                    self.cur_num_ATC_instr += 1

            else:
                print("There is at least one case that has not been taken into account for the filtering of actions!!")
                exit(0)

        self.np_dur_of_actions = np.asarray(self.durations_of_actions).copy()
        self.np_mask_climb_descend_res_fplan_with_confl_loss = np.asarray(mask_climb_descend_res_fplan_with_confl_loss)
        self.np_mask_res_fplan_after_maneuv = np.asarray(mask_res_fplan_after_maneuv)
        self.np_resolutionActionID_only_for_train = np.asarray(resolutionActionID_only_for_train_)
        self.cur_additional_nautical_miles = np.asarray(additional_nautical_miles)
        self.cur_additional_duration = np.asarray(additional_duration)

    def fix_duration_to_be_passed_to_env(self):

        np_dur_of_actions_only_dur = self.np_dur_of_actions[:, 0]

        # num_dc+num_ds+num_actions_as <= actions < num_dc+num_ds+num_as+num_wp means 'direct to (any) way point' action
        # num_dc+num_ds <= actions < num_dc+num_ds+num_as means 'change FL' action
        dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL = \
            np.where((np_dur_of_actions_only_dur == np.inf) &
                     (((self.actions >= self.num_actions_dc + self.num_actions_ds + self.num_actions_as) &
                       (self.actions < self.num_actions_dc + self.num_actions_ds +
                        self.num_actions_as + self.num_dir_wp)) |
                      ((self.actions >= self.num_actions_dc + self.num_actions_ds) &
                       (self.actions < self.num_actions_dc + self.num_actions_ds + self.num_actions_as))),
                     0, np_dur_of_actions_only_dur)

        # num_dc+num_ds+num_as+num_wp+1 means 'continue' action
        dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL_and_without_inf_for_contin = np.where(
            (dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL == np.inf) &
            (self.actions == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1),
            0, dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL)

        # num_dc+num_ds+num_as+num_wp means 'zero' action
        dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL_and_without_inf_for_contin_with_min_dur_in_zero_action = \
            np.where(self.actions == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp,
                     self.interval_between_two_steps,
                     dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL_and_without_inf_for_contin)

        self.actions_duration_to_be_passed_to_env = \
            np.expand_dims(
                dur_of_acts_without_inf_for_dir_to_wp_or_ch_FL_and_without_inf_for_contin_with_min_dur_in_zero_action,
                axis=1)

    def get_true_actions(self, actions_, n_agents_, active_flights, dur_vals=None):
        """

        :param actions_: numpy array with shape [n_agents_,] or list with length=n_agents_. Each element 'i'
        declares the selected action (a number in the range
        [0, self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 3]) of agent 'i'.
        :param n_agents_: The total number of elements in array/list 'actions_'
        :param active_flights: numpy array with shape [n_agents_,] or list with length=n_agents_. Each element 'i'
        declares if agent 'i' is active or not.
        :param dur_vals: None or numpy array with shape [n_agents_, 1]. If not None, each element [i, 0] is the
        duration of the action of agent 'i'.

        :return: numpy array with shape [n_agents_, 11]. The first 10 elements indicate the type action
        to be executed and its value, and the last elements determines the duration of the action.
        """

        true_actions = np.zeros((n_agents_, self.num_types_of_actions), dtype=float)

        for i__ in range(n_agents_):
            if not active_flights[i__]:  # Zero action
                true_actions[i__] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif actions_[i__] < self.num_actions_dc:
                true_actions[i__] = np.asarray([self.actions_list[actions_[i__]], 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.num_actions_dc <= actions_[i__] < self.num_actions_dc + self.num_actions_ds:
                true_actions[i__] = np.asarray([0, self.actions_list[actions_[i__]], 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.num_actions_dc + self.num_actions_ds <= actions_[i__] < \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as:
                true_actions[i__] = np.asarray([0, 0, self.actions_list[actions_[i__]], 0, 0, 0, 0, 0, 0, 0])
            elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= actions_[i__] < \
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                wp_vector = []
                for way_p in range(self.num_dir_wp):
                    # The vector of 'actions' should have the indicator of 'follow flight plan' action
                    # in the position after the position of the indicator of the 'direct to next way point' action
                    if way_p == 1:
                        wp_vector.append(0)
                    if self.num_actions_dc + self.num_actions_ds + self.num_actions_as + way_p == actions_[i__]:
                        wp_vector.append(1)
                    else:
                        wp_vector.append(0)
                true_actions[i__] = np.asarray([0, 0, 0] + wp_vector + [0, 0])

            # Zero action
            elif actions_[i__] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                true_actions[i__] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            # Default action ('continue' action)
            elif actions_[i__] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1:
                true_actions[i__] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

            # Default action ('resume fplan')
            elif actions_[i__] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2:
                true_actions[i__] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

            # Default action ('follow flight plan' action)
            elif actions_[i__] == self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 3:
                true_actions[i__] = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            else:
                print("Action {} of agent {} out of range of action list!!".format(actions_[i__], i__))
                exit(0)

        # Append duration to the last column.
        # Duration values should be a numpy array with shape: (n_agents, 1) and not (n_agents,).
        if dur_vals is None:
            dur_vals = np.zeros((n_agents_, 1), dtype=float)
        true_actions_with_duration = np.append(np.append(true_actions[:, :9], dur_vals, axis=1),
                                               np.expand_dims(true_actions[:, 9], axis=1), axis=1)

        return true_actions_with_duration

    def update_dataframes_and_actionsID(self):

        if self.evaluation:

            self.dict_actions_attentions[self.i_episode - 1][self.timestamp] = \
                {'actions': copy.deepcopy(self.dict_agents_actions),
                 'attentions:': {'values': self.att_all.copy(),
                                 'labels': self.flights_idxs_to_ids[np.asarray(self.adjacency_matrix)].copy()}}

            # Update dataframes
            self.store_flights_position()
            self.store_conflicts_heat_maps()
            self.store_htmp_by_agent()
            self.store_losses()
            self.store_conflicts()
            self.store_resolution_actions()

            # Store resolution actions and measure their impact
            self.dict_actions[self.steps - 1] = copy.deepcopy(self.res_acts_dict_for_evaluation)
            self.env.actions_impact(copy.deepcopy(self.res_acts_dict_for_evaluation))

            ######Write to csv files######
            if not os.path.exists('./logs'): os.mkdir('./logs')
            self.confl_htmaps.to_csv("./logs/conflict_heatmaps_episode_{}.csv".format(self.i_episode), index=False)
            self.confl_resol_act.to_csv("./logs/resolution_actions_episode_{}.csv".format(self.i_episode), index=False)
        else:
            self.valid_acts_to_be_executed_ID = \
                self.np_resolutionActionID_only_for_train[:self.envs_n_agent[self.env_number], 0]

    def store_flights_position(self):

        len_df = len(self.flights_positions.index)
        counter_ = 0
        for idx, flight in enumerate(self.flight_array_):
            if self.active_flights_mask[idx]:
                self.flights_positions.loc[len_df + counter_] = \
                    [flight[1],  # 'lon'
                     flight[2],  # 'lat'
                     flight[0],  # 'flight'
                     self.timestamp,  # 'step'
                     (datetime.utcfromtimestamp(self.timestamp)).strftime("%Y-%m-%d %H:%M:%S"),  # 'utc_timestamp'
                     self.flight_phases[idx],  # 'flight phase'
                     flight[5],  # 'alt'
                     flight[8],  # 'x'
                     flight[9],  # 'speed_h_magnitude'
                     flight[10],  # 'speed_v'
                     flight[11],  # 'x-y'
                     flight[12],  # 'alt-exit_point_alt'
                     flight[13]]  # 'd'
                counter_ += 1
            else:
                continue

    def store_conflicts_heat_maps(self):

        adj_matrix_ids = self.flights_idxs_to_ids[np.asarray(self.adjacency_matrix)]
        current_scenario_n_agent = self.envs_n_agent[self.env_number]
        len_df = len(self.confl_htmaps.index)
        counter_ = 0
        all_confl = \
            self.loss_ids + self.alrts_ids + self.confls_ids_with_positive_tcpa + self.confls_ids_with_negative_tcpa
        for confl_ in range(len(all_confl)):

            # Since any conflict will be added twice we should keep only one
            # (the conflict of edge flight_i_flight_j or the edge flight_j_flight_i).
            # We keep the one with the minimum flight ID first.
            if ((all_confl[confl_][1], all_confl[confl_][0]) in all_confl) \
                    and (self.flights_idxs_to_ids[all_confl[confl_][1]] <
                         self.flights_idxs_to_ids[all_confl[confl_][0]]):
                continue
            else:
                self.confl_htmaps.loc[len_df + counter_] = \
                    [str(int(self.timestamp)) + "_" + str(self.flights_idxs_to_ids[all_confl[confl_][0]])] + \
                    [self.flights_idxs_to_ids[all_confl[confl_][0]]] + \
                    [str(int(self.timestamp)) + "_" + str(self.flights_idxs_to_ids[all_confl[confl_][0]]) + "_" +
                     str(self.flights_idxs_to_ids[all_confl[confl_][1]])] + \
                    [np.mean(self.att_all[all_confl[confl_][0], :, :,
                             np.where(adj_matrix_ids[all_confl[confl_][0]] == self.flights_idxs_to_ids[iii])[0][0]])
                     if self.flights_idxs_to_ids[iii] in adj_matrix_ids[all_confl[confl_][0]]
                     else "null" for iii in range(current_scenario_n_agent)]
                self.confl_htmaps.loc[len_df + counter_ + 1] = \
                    [str(int(self.timestamp)) + "_" + str(self.flights_idxs_to_ids[all_confl[confl_][1]])] + \
                    [self.flights_idxs_to_ids[all_confl[confl_][1]]] + \
                    [str(int(self.timestamp)) + "_" + str(self.flights_idxs_to_ids[all_confl[confl_][0]]) + "_" +
                     str(self.flights_idxs_to_ids[all_confl[confl_][1]])] + \
                    [np.mean(self.att_all[all_confl[confl_][1], :, :,
                             np.where(adj_matrix_ids[all_confl[confl_][1]] == self.flights_idxs_to_ids[iii])[0][0]])
                     if self.flights_idxs_to_ids[iii] in adj_matrix_ids[all_confl[confl_][1]]
                     else "null" for iii in range(current_scenario_n_agent)]

                counter_ += 2

    def store_htmp_by_agent(self):

        for agent in range(len(self.htmp_by_agent)):

            if self.active_flights_mask[agent]:
                len_df = len(self.htmp_by_agent[agent].index)
                counter_ = 0

                for neighbor in range(self.adjacency_matrix[agent].shape[0]):

                    # -1 means a dummy neighbor
                    if self.flights_idxs_to_ids[self.adjacency_matrix[agent][neighbor]] != -1:

                        if self.flight_array_[agent, 0] != \
                                self.flight_array_[self.adjacency_matrix[agent][neighbor]][0]:

                            if tuple((agent, self.adjacency_matrix[agent][neighbor])) in self.loss_ids:
                                warn_type = "loss"
                            elif tuple((agent, self.adjacency_matrix[agent][neighbor])) in self.alrts_ids:
                                warn_type = "alert"
                            elif tuple((agent, self.adjacency_matrix[agent][neighbor])) in \
                                    self.confls_ids_with_positive_tcpa:
                                warn_type = "confl_pos_tcpa"
                            elif tuple((agent, self.adjacency_matrix[agent][neighbor])) in \
                                    self.confls_ids_with_negative_tcpa:
                                warn_type = "confl_neg_tcpa"
                            else:
                                print("Current warning is not included in the possible warnings!!!")
                                exit(0)

                        else:
                            warn_type = "None"

                        current_flights_edges_ar = \
                            self.edges[np.where((self.edges[:, 0] == self.flight_array_[agent, 0]) *
                                                self.edges[:, 1] ==
                                                self.flight_array_[self.adjacency_matrix[agent][neighbor]][0])[0][0]] \
                                if self.flight_array_[agent, 0] != \
                                   self.flight_array_[self.adjacency_matrix[agent][neighbor]][0] \
                                else np.zeros((self.edges.shape[1]))

                        self.htmp_by_agent[agent].loc[len_df + counter_] = \
                            [self.flight_array_[agent, 0],  # 'Current_flight'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][1],  # 'lon'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][2],  # 'lat'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][0],  # 'flight'
                             self.timestamp,  # 'step'
                             (datetime.utcfromtimestamp(self.timestamp)).
                             strftime("%Y-%m-%d %H:%M:%S"),  # 'utc_timestamp'
                             np.mean(self.att_all[agent, :, :, neighbor]),  # 'mean_all_attention'
                             np.argsort(np.argsort(np.mean(np.mean(self.att_all[agent, :, :, :], axis=1), axis=0) * -1,
                                                   axis=0))[neighbor],  # 'mean_all_attention_ranking'
                             np.mean(self.att_all[agent, 0, :, neighbor]),  # 'mean_1st_conv'
                             np.argsort(np.argsort(np.mean(self.att_all[agent, 0, :, :], axis=0) * -1,
                                                   axis=0))[neighbor], # 'mean_1st_conv_ranking'
                             np.mean(self.att_all[agent, 1, :, neighbor]),  # 'mean_2nd_conv'
                             np.argsort(np.argsort(np.mean(self.att_all[agent, 1, :, :], axis=0) * -1,
                                                   axis=0))[neighbor], # 'mean_2nd_conv_ranking',
                             np.max(self.att_all[agent, :, :, neighbor]),  # 'max_all'
                             np.argsort(np.argsort(np.max(np.max(self.att_all[agent, :, :, :], axis=1), axis=0) * -1,
                                                   axis=0))[neighbor],  # 'max_all_ranking'
                             np.max(self.att_all[agent, 0, :, neighbor]),  # '1st_conv_max'
                             np.argsort(np.argsort(np.max(self.att_all[agent, 0, :, :], axis=0) * -1,
                                                   axis=0))[neighbor], # '1st_conv_max_ranking'
                             np.max(self.att_all[agent, 1, :, neighbor]),  # '2nd_conv_max'
                             np.argsort(np.argsort(np.max(self.att_all[agent, 1, :, :], axis=0) * -1,
                                                   axis=0))[neighbor], # '2nd_conv_max_ranking'
                             self.dict_agents_actions[self.flights_idxs_to_ids[self.adjacency_matrix[agent][
                                 neighbor]]]["action_type"], # 'action_type'
                             self.dict_agents_actions[self.flights_idxs_to_ids[self.adjacency_matrix[agent][
                                 neighbor]]]['true_action'],  # 'action'
                             self.durations_of_actions[self.adjacency_matrix[agent][neighbor]][0],  # 'duration'
                             self.flight_phases[self.adjacency_matrix[agent][neighbor]],  # flight phase
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][5],  # 'alt'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][8],  # 'x'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][9],  # 'speed_h_magnitude'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][10],  # 'speed_v'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][11],  # 'x-y'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][12],  # 'alt-exit_point_alt'
                             self.flight_array_[self.adjacency_matrix[agent][neighbor]][13],  # 'd'
                             current_flights_edges_ar[2],  # 'tcpa',
                             current_flights_edges_ar[3],  # 'dcpa'
                             current_flights_edges_ar[4],  # 'aij'
                             current_flights_edges_ar[5],  # 'bij'
                             current_flights_edges_ar[6],  # 'vdcpa'
                             current_flights_edges_ar[7],  # 'dcp'
                             current_flights_edges_ar[8],  # 'tcp'
                             current_flights_edges_ar[9],  # 'hdij'
                             current_flights_edges_ar[10],  # 'vdij'
                             warn_type]  # 'warn_type'

                        counter_ += 1

    def store_losses(self):

        len_df = len(self.losses_file.index)
        counter_ = 0
        for loss_ in range(len(self.loss_ids)):

            if ((self.loss_ids[loss_][1], self.loss_ids[loss_][0]) in self.loss_ids) and \
                    (self.flights_idxs_to_ids[self.loss_ids[loss_][1]] <
                     self.flights_idxs_to_ids[self.loss_ids[loss_][0]]):
                continue
            else:
                self.losses_file.loc[len_df + counter_] = \
                    [(self.timestamp,
                      self.flights_idxs_to_ids[self.loss_ids[loss_][0]],
                      self.flights_idxs_to_ids[self.loss_ids[loss_][1]])]

                counter_ += 1

    def store_conflicts(self):

        len_df = len(self.conflicts_file.index)
        counter_ = 0
        conflicts = self.alrts_ids + self.confls_ids_with_positive_tcpa + self.confls_ids_with_negative_tcpa

        for conflict_ in range(len(conflicts)):

            if ((conflicts[conflict_][1], conflicts[conflict_][0]) in conflicts) and \
                    (self.flights_idxs_to_ids[conflicts[conflict_][1]] <
                     self.flights_idxs_to_ids[conflicts[conflict_][0]]):
                continue

            else:

                if tuple((conflicts[conflict_][0], conflicts[conflict_][1])) in self.alrts_ids:
                    warn_type = "alert"
                elif tuple((conflicts[conflict_][0], conflicts[conflict_][1])) in self.confls_ids_with_positive_tcpa:
                    warn_type = "confl_pos_tcpa"
                elif tuple((conflicts[conflict_][0], conflicts[conflict_][1])) in self.confls_ids_with_negative_tcpa:
                    warn_type = "confl_neg_tcpa"
                else:
                    print("Current warning is not included in the possible warnings!!!")
                    exit(0)

                self.conflicts_file.loc[len_df + counter_] = [
                    (self.timestamp,
                     self.flights_idxs_to_ids[conflicts[conflict_][0]],
                     self.flights_idxs_to_ids[conflicts[conflict_][1]]),
                    warn_type]

                counter_ += 1

    def store_resolution_actions(self):

        len_df = len(self.confl_resol_act.index)
        counter_ = 0
        all_confl = \
            self.loss_ids + self.alrts_ids + self.confls_ids_with_positive_tcpa + self.confls_ids_with_negative_tcpa
        resolution_actions_ids = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        resolution_actions_ids_to_be_passed_to_env = \
            np.zeros((self.n_agent, self.max_poss_actions_for_evaluation_to_be_passed_to_env), dtype=object)
        self.valid_acts_to_be_executed_ID = np.zeros(self.n_agent, dtype=object)
        filtered_out_mask = \
            np.asarray([[False for _ in range(self.max_poss_actions_for_evaluation_to_be_passed_to_env)]
                               for _ in range(self.n_agent)])
        dummy_possible_actions_mask_to_be_passed_to_env = \
            np.asarray([[False for _ in range(self.max_poss_actions_for_evaluation_to_be_passed_to_env)]
                               for _ in range(self.n_agent)])
        additional_nautical_miles = \
            np.array([["0"] * (self.max_poss_actions_for_evaluation + 1)] * self.n_agent, dtype=object)
        additional_duration = \
            np.array([["0"] * (self.max_poss_actions_for_evaluation + 1)] * self.n_agent, dtype=object)
        VSpeedChange = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        HSpeedChange = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        CourseChange = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        HShiftFromExitPoint = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        VShiftFromExitPoint = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)
        Bearing = np.zeros((self.n_agent, self.max_poss_actions_for_evaluation + 1), dtype=object)

        # Filter the actions
        self.get_valid_actions_for_evaluation()

        true_actions_for_evaluation_to_be_passed_to_env = \
            np.asarray([[[0] * self.true_actions_for_evaluation.shape[2]
                         for _ in range(self.max_poss_actions_for_evaluation_to_be_passed_to_env)]
                         for _ in range(self.n_agent)])

        # Mask to monitor the agents for which the actions are not stored to the dictionary
        # through the loop of conflicts
        true_actions_for_evaluation_to_be_passed_to_env_mask = [False for _ in range(self.n_agent)]
        counter_poss_actions_to_be_passed_to_env = [0 for _ in range(self.n_agent)]

        for confl_ in range(len(all_confl)):

            # Because any conflict will be added twice we should keep only the one
            # (the conflict of edge flight_i_flight_j or the edge flight_j_flight_i).
            # We keep the one with the minimum flight ID first.
            if ((all_confl[confl_][1], all_confl[confl_][0]) in all_confl) \
                    and (self.flights_idxs_to_ids[all_confl[confl_][1]] <
                         self.flights_idxs_to_ids[all_confl[confl_][0]]):
                continue
            else:

                # Range is 2 because there are only two flights in a conflict/alert/loss of separation
                for agent in range(2):
                    actionRank = 0
                    for poss_action in range(self.max_poss_actions_for_evaluation + 1):

                        # We need to examine max_poss_actions_for_evaluation+1 actions in case of
                        # self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0],
                        # because the extra deterministic action ('continue'/'resume fplan')
                        # should not replace another possible action.
                        # Also, it might be the rare case that the q-value of the deterministic action is the minimum,
                        # and then it will not be reported to the file 'resolution_action_{}.csv'.
                        # However, in all other cases we need to break the loop in
                        # max_poss_actions_for_evaluation-1 actions (because the counting starts from zero).
                        if not self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0] and \
                                poss_action >= self.max_poss_actions_for_evaluation:
                            break

                        # Compute additional nautical miles and additional duration
                        if self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'S2':
                            temp_additional_nautical_miles_and_duration = \
                                self.env.compute_additional_nautical_miles_course_change(
                                    self.flights_idxs_to_ids[all_confl[confl_][agent]],
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action])
                            additional_nautical_miles[all_confl[confl_][agent], poss_action] = \
                                temp_additional_nautical_miles_and_duration[0]
                            additional_duration[all_confl[confl_][agent], poss_action] = \
                                temp_additional_nautical_miles_and_duration[1]
                        elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A3':
                            if self.available_wps[
                                all_confl[confl_][agent],
                                abs(((self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp) -
                                     self.indices_possible_actions[all_confl[confl_][agent]][poss_action])
                                    - self.num_dir_wp)] == 1:

                                temp_additional_nautical_miles_and_duration = \
                                    self.env.compute_additional_nautical_miles_direct_to(
                                        self.flights_idxs_to_ids[all_confl[confl_][agent]],
                                        self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action])
                                additional_nautical_miles[all_confl[confl_][agent], poss_action] = \
                                    temp_additional_nautical_miles_and_duration[0]
                                additional_duration[all_confl[confl_][agent], poss_action] = \
                                    temp_additional_nautical_miles_and_duration[1]
                            else:
                                additional_nautical_miles[all_confl[confl_][agent], poss_action] = "null"
                                additional_duration[all_confl[confl_][agent], poss_action] = "null"
                        elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A1':
                            additional_duration[all_confl[confl_][agent], poss_action] = \
                                self.env.compute_additional_duration_FL_change(
                                    self.flights_idxs_to_ids[all_confl[confl_][agent]],
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action])
                        elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A2':
                            additional_duration[all_confl[confl_][agent], poss_action] = \
                                self.env.compute_additional_duration_speed_change(
                                    self.flights_idxs_to_ids[all_confl[confl_][agent]],
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action])

                        # Compute VSpeedChange, HSpeedChange, HShiftFromExitPoint, VShiftFromExitPoint, Bearing
                        # A1 is FL_change
                        # A2 is (horizontal) speed change
                        # S2 is course change
                        # A4 is "zero" action
                        # A3 is direct_to_way_point
                        # The condition:
                        # (poss_action == 0 and
                        # self.dummy_possible_actions_mask[all_confl[confl_][agent], poss_action+1] == True)
                        # has the below explanation:
                        # If the case of dummy actions, which are referred to "zero"/"continue" actions.
                        # Specifically, the cases are:
                        # 1) when a selected action is in progress
                        #    (not for the action 'direct to (any) way point'/'change FL') and
                        #    'zero' action should be executed.
                        # 2) when the action 'direct to (any) way point'/'change FL' is selected and
                        #    'continue' action should be executed
                        if self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'CA' or \
                                self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'RFP' or \
                                (poss_action == 0 and
                                 self.dummy_possible_actions_mask[all_confl[confl_][agent], poss_action + 1]) or \
                                (self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A3' and
                                 self.available_wps[
                                     all_confl[confl_][agent],
                                     self.indices_possible_actions[all_confl[confl_][agent]][poss_action] -
                                     (self.num_actions_dc + self.num_actions_ds + self.num_actions_as)] == 0):

                            VSpeedChange[all_confl[confl_][agent], poss_action] = 'null'
                            HSpeedChange[all_confl[confl_][agent], poss_action] = 'null'
                            CourseChange[all_confl[confl_][agent], poss_action] = 'null'
                            HShiftFromExitPoint[all_confl[confl_][agent], poss_action] = 'null'
                            VShiftFromExitPoint[all_confl[confl_][agent], poss_action] = 'null'
                            Bearing[all_confl[confl_][agent], poss_action] = 'null'
                        else:
                            HShiftFromExitPoint[all_confl[confl_][agent], poss_action], \
                            VShiftFromExitPoint[all_confl[confl_][agent], poss_action], \
                            Bearing[all_confl[confl_][agent], poss_action] = \
                                self.env.compute_shift_from_exit_point(
                                    self.flights_idxs_to_ids[all_confl[confl_][agent]],
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action])
                            if self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A1':
                                VSpeedChange[all_confl[confl_][agent], poss_action] = \
                                    self.action_values_for_evaluation[
                                        self.indices_possible_actions[all_confl[confl_][agent]][poss_action]]
                            elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A2':
                                HSpeedChange[all_confl[confl_][agent], poss_action] = \
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action][1]
                            elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'S2':
                                CourseChange[all_confl[confl_][agent], poss_action] = \
                                    self.flight_array_[all_confl[confl_][agent], 8] + \
                                    self.true_actions_for_evaluation[all_confl[confl_][agent]][poss_action][0]
                            elif self.possible_actions_type[all_confl[confl_][agent]][poss_action] == 'A3':
                                CourseChange[all_confl[confl_][agent], poss_action] = \
                                    self.flight_array_[
                                        all_confl[confl_][agent],
                                        14 + (self.indices_possible_actions[all_confl[confl_][agent]][poss_action] -
                                              (self.num_actions_dc + self.num_actions_ds + self.num_actions_as))]

                        # Assign resolution action ID
                        # If the executed action is:
                        #   - "zero"/"continue" due to another resolution action in progress, or
                        #   - "resume_fplan" (or "continue" because "resume_fplan" is already in progress),
                        #      and self.not_action_due_to_phase_mask[all_confl[confl_][agent]] is True
                        # --> then the resolution action id should be: "timestamp_flightID_no_resolution"
                        resolution_actions_ids[all_confl[confl_][agent], poss_action] = \
                            str(int(self.timestamp)) + "_" + \
                            str(self.flights_idxs_to_ids[all_confl[confl_][agent]]) + "_" + "no_resolution" \
                                if ((poss_action == 0 and
                                     self.dummy_possible_actions_mask[all_confl[confl_][agent], poss_action + 1]) or
                                    (self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0] and
                                     (self.indices_possible_actions[all_confl[confl_][agent]][poss_action] ==
                                      self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp + 2 or
                                      self.indices_possible_actions[all_confl[confl_][agent]][poss_action] ==
                                      self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp + 1))) \
                                else str(int(self.timestamp)) + "_" + \
                                     str(self.flights_idxs_to_ids[all_confl[confl_][agent]]) + "_" + \
                                     str(self.possible_actions_type[all_confl[confl_][agent]][poss_action]) + "_" + \
                                     str(self.possible_actions[all_confl[confl_][agent]][poss_action]) + "_" + \
                                     str(self.durations_of_actions_for_evaluation[
                                             all_confl[confl_][agent]][poss_action][0])

                        # Assign actionRank
                        temp_actionRank = actionRank
                        if self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0]:
                            if not (self.indices_possible_actions[all_confl[confl_][agent]][poss_action] ==
                                    self.num_actions_dc + self.num_actions_ds +
                                    self.num_actions_as + self.num_dir_wp + 2 or
                                    self.indices_possible_actions[all_confl[confl_][agent]][poss_action] ==
                                    self.num_actions_dc + self.num_actions_ds +
                                    self.num_actions_as + self.num_dir_wp + 1):

                                temp_actionRank = 100

                            else:

                                temp_actionRank = 0

                        elif self.filtered_out_reason[all_confl[confl_][agent]][poss_action] != 'null':

                            temp_actionRank = 100

                        else:

                            actionRank += 1

                        # Store info to dataframe
                        self.confl_resol_act.loc[len_df + counter_] = \
                            [resolution_actions_ids[all_confl[confl_][agent], poss_action],  # ResolutionID
                             str(self.flights_idxs_to_ids[all_confl[confl_][agent]]),  # RTKey
                             str(int(self.timestamp)) + "_" +
                             str(self.flights_idxs_to_ids[all_confl[confl_][0]]) + "_" +
                             str(self.flights_idxs_to_ids[all_confl[confl_][1]]),  # ConflictID
                             self.possible_actions_type[all_confl[confl_][agent]][poss_action],  # ResolutionActionType
                             self.possible_actions[all_confl[confl_][agent]][poss_action],  # ResolutionAction
                             self.q_values_of_poss_actions[all_confl[confl_][agent]][poss_action],  # Q-values
                             temp_actionRank,  # ActionRank
                             additional_nautical_miles[all_confl[confl_][agent],
                                                       poss_action], # AdditionalNauticalMiles
                             additional_duration[all_confl[confl_][agent], poss_action],  # AdditionalDuration
                             self.durations_of_actions_for_evaluation[
                                 all_confl[confl_][agent]][poss_action][0],  # Duration
                             self.filtered_out_reason[all_confl[confl_][agent]][poss_action],  # FilteredOut
                             self.action_in_progress[all_confl[confl_][agent]],  # ActionInProgress
                             str(int(self.timestamp)) + "_" +
                             str(self.flights_idxs_to_ids[all_confl[confl_][agent]]),  # Prioritization
                             VSpeedChange[all_confl[confl_][agent], poss_action],  # VSpeedChange
                             HSpeedChange[all_confl[confl_][agent], poss_action],  # HSpeedChange
                             CourseChange[all_confl[confl_][agent], poss_action],  # CourseChange
                             HShiftFromExitPoint[all_confl[confl_][agent], poss_action],  # HShiftFromExitPoint
                             VShiftFromExitPoint[all_confl[confl_][agent], poss_action],  # VShiftFromExitPoint
                             Bearing[all_confl[confl_][agent], poss_action]]  # Bearing

                        if self.indices_possible_actions[all_confl[confl_][agent]][poss_action] == \
                                self.actions[all_confl[confl_][agent]]:

                            # resolution action IDs of actions which will be executed
                            self.valid_acts_to_be_executed_ID[all_confl[confl_][agent]] = \
                                resolution_actions_ids[all_confl[confl_][agent], poss_action]

                        counter_ += 1

                        # If the case of dummy actions, which is only "zero"/"continue" actions.
                        # Specifically, the cases are:
                        # 1) when a selected action is in progress
                        #    (not for the action 'direct to (any) way point'/'change FL')
                        #    and 'zero' action should be executed.
                        # 2) when the action 'direct to (any) way point'/'change FL' is selected
                        #    and 'continue' action should be executed
                        if poss_action == 0 and \
                                self.dummy_possible_actions_mask[all_confl[confl_][agent], poss_action + 1]:

                            for zero_or_continue_action_ in \
                                    range(self.max_poss_actions_for_evaluation_to_be_passed_to_env):

                                resolution_actions_ids_to_be_passed_to_env[all_confl[confl_][agent],
                                                                           zero_or_continue_action_] = \
                                    resolution_actions_ids[all_confl[confl_][agent], poss_action]

                                true_actions_for_evaluation_to_be_passed_to_env[all_confl[confl_][agent],
                                                                                zero_or_continue_action_] = \
                                    self.true_actions_for_evaluation[all_confl[confl_][agent],
                                                                     poss_action + zero_or_continue_action_]

                                filtered_out_mask[all_confl[confl_][agent], zero_or_continue_action_] = True

                                dummy_possible_actions_mask_to_be_passed_to_env[all_confl[confl_][agent],
                                                                                zero_or_continue_action_] = \
                                    self.dummy_possible_actions_mask[
                                        all_confl[confl_][agent]][poss_action + zero_or_continue_action_]

                            counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]] = \
                                self.max_poss_actions_for_evaluation_to_be_passed_to_env
                            break

                        # If the flight phase is 'climbing'/'descending' and:
                        #  - the current flight is in conflict/loss,
                        #  --> then the action 'resume fplan' should be executed,
                        #  OR
                        #  - if the action 'resume fplan' was already in progress and
                        #    the current flight is in conflict/loss,
                        #    --> then the action 'continue' should be executed.
                        # In these cases, all the actions are reported to file 'resolution_actions_episode_',
                        # but the action passed to environment is the action
                        # 'resume fplan'/'continue' max_poss_actions_for_evaluation_to_be_passed_to_env times,
                        # annotated with False in 'filtered_out_mask' and
                        # 'dummy_possible_actions_mask_to_be_passed_to_env' arrays only in the first position.
                        # Note that the condition is poss_action==max_poss_actions_for_evaluation,
                        # because in case of self.not_action_due_to_phase_mask[agent][0]==True the loop
                        # of actions ends at max_poss_actions_for_evaluation (starting from zero).
                        elif self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0] and \
                                poss_action == self.max_poss_actions_for_evaluation:

                            for res_fplan_or_continue_action_ in \
                                    range(self.max_poss_actions_for_evaluation_to_be_passed_to_env):

                                resolution_actions_ids_to_be_passed_to_env[all_confl[confl_][agent],
                                                                           res_fplan_or_continue_action_] = \
                                    resolution_actions_ids[all_confl[confl_][agent],
                                                           self.not_action_due_to_phase_mask[
                                                               all_confl[confl_][agent]][2]]

                                true_actions_for_evaluation_to_be_passed_to_env[
                                    all_confl[confl_][agent], res_fplan_or_continue_action_] = \
                                    self.true_actions_for_evaluation[
                                        all_confl[confl_][agent],
                                        self.not_action_due_to_phase_mask[all_confl[confl_][agent]][2]]

                                filtered_out_mask[all_confl[confl_][agent], res_fplan_or_continue_action_] = \
                                    False if res_fplan_or_continue_action_ == 0 \
                                          else True

                                dummy_possible_actions_mask_to_be_passed_to_env[all_confl[confl_][agent],
                                                                                res_fplan_or_continue_action_] = \
                                    False if res_fplan_or_continue_action_ == 0 \
                                          else True

                            counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]] = \
                                self.max_poss_actions_for_evaluation_to_be_passed_to_env

                        # Else if maximum number of actions to be passed to environment has not been reached
                        # and the current action is not filtered out
                        elif counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]] < \
                                self.max_poss_actions_for_evaluation_to_be_passed_to_env and \
                                self.filtered_out_reason[all_confl[confl_][agent]][poss_action] == "null" and \
                                not self.not_action_due_to_phase_mask[all_confl[confl_][agent]][0]:

                            resolution_actions_ids_to_be_passed_to_env[
                                all_confl[confl_][agent],
                                counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]]] = \
                                resolution_actions_ids[all_confl[confl_][agent], poss_action]

                            true_actions_for_evaluation_to_be_passed_to_env[
                                all_confl[confl_][agent],
                                counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]]] = \
                                self.true_actions_for_evaluation[all_confl[confl_][agent], poss_action]

                            filtered_out_mask[all_confl[confl_][agent],
                                              counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]]] = \
                                False if self.filtered_out_reason[all_confl[confl_][agent]][poss_action] == "null" \
                                      else True

                            dummy_possible_actions_mask_to_be_passed_to_env[
                                all_confl[confl_][agent],
                                counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]]] = \
                                self.dummy_possible_actions_mask[all_confl[confl_][agent]][poss_action]

                            counter_poss_actions_to_be_passed_to_env[all_confl[confl_][agent]] += 1

                    true_actions_for_evaluation_to_be_passed_to_env_mask[all_confl[confl_][agent]] = True

        # Store the actions which were not stored to the dictionary through the loop of conflicts
        # because not all the agents participate in a conflict/loss of separation.
        for agent___ in range(self.n_agent):
            if not true_actions_for_evaluation_to_be_passed_to_env_mask[agent___]:
                true_actions_for_evaluation_to_be_passed_to_env[agent___] = \
                    self.true_actions_for_evaluation[
                    agent___, :self.max_poss_actions_for_evaluation_to_be_passed_to_env]

        self.res_acts_dict_for_evaluation = \
            {'res_acts': true_actions_for_evaluation_to_be_passed_to_env[:self.envs_n_agent[self.env_number]],
             'res_acts_ID': resolution_actions_ids_to_be_passed_to_env[:self.envs_n_agent[self.env_number]],
             'filt_out_mask': filtered_out_mask[:self.envs_n_agent[self.env_number]],
             'dummy_poss_acts_mask':
                 dummy_possible_actions_mask_to_be_passed_to_env[:self.envs_n_agent[self.env_number]]}

    def get_valid_actions_for_evaluation(self):

        self.indices_possible_actions = [[] for _ in range(self.n_agent)]
        self.possible_actions = [[] for _ in range(self.n_agent)]
        self.dummy_possible_actions_mask = \
            [[False for _ in range(self.max_poss_actions_for_evaluation)] for _ in range(self.n_agent)]
        self.possible_actions_type = [[] for _ in range(self.n_agent)]
        self.q_values_of_poss_actions = [[] for _ in range(self.n_agent)]
        self.filtered_out_reason = [[] for _ in range(self.n_agent)]  # Null if not filtered out
        self.durations_of_actions_for_evaluation = [[] for _ in range(self.n_agent)]

        # Null except for the case that a flight executes an action with duration and this duration has not
        # been reached while a conflict/loss of separation has been detected for this flight at the current timestep.
        self.action_in_progress = ["Null" for _ in range(self.n_agent)]

        # Mask to annotate the cases where there is conflict/loss and there is no action in progress, but only the
        # action "resume fplan"/'continue' can be executed due to the flight phase ('climbing'/'descending').
        # The second item of the tuple is the index of action which should be executed ('resume fplan'/'continue').
        # The third item of the tuple is index of the action (which should be executed) of array with the actions
        # sorted according to q-values.
        self.not_action_due_to_phase_mask = [(False, -1, -1) for _ in range(self.n_agent)]

        for agent in range(self.n_agent):

            # When an agent is not active, or it doesn't participate in a
            # conflict/alert/loss of separation we don't care about its actions
            if not self.active_flights_mask[agent]:

                # num_dc+num_ds+num_as+num_wp means "zero" action
                self.indices_possible_actions[agent].\
                    extend([self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp
                            for _ in range(self.max_poss_actions_for_evaluation)])
                self.possible_actions[agent].extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.filtered_out_reason[agent].\
                    extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.durations_of_actions_for_evaluation[agent].\
                    extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            elif not (agent in self.history_loss_confl):
                # For flights which are not detected to participate in a conflict / loss of separation until
                # now in this episode.
                # Plus 3 means the default action (follow flight plan)
                self.indices_possible_actions[agent].\
                    extend([self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 3
                            for _ in range(self.max_poss_actions_for_evaluation)])
                self.possible_actions[agent].extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.filtered_out_reason[agent].\
                    extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.durations_of_actions_for_evaluation[agent].\
                    extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            # We assume that the flight, which is in 'climbing' or 'descending' flight phase,
            # is already in 'history_loss_confl', otherwise the previous condition would be true.
            # IF the phase of a flight is 'climbing'/'descending' and there is no action in progress
            # (based on (self.durations_of_actions_for_evaluation_[agent][0] == 0 and
            # not self.executing_resume_to_fplan[agent])
            # for previous deterministic execution of the action 'self.executing_resume_to_fplan',
            # and (self.durations_of_actions_for_evaluation_[agent][0] == np.inf and not self.executing_direct_to[agent]
            # and not self.executing_FL_change[agent]) for previous non-deterministic execution of actions
            # 'direct to (any) way point'/'change FL'),
            # OR the selected action has just finished
            # (based on
            # self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] >=
            # self.durations_of_actions_for_evaluation_[agent][0]),
            # THEN the action "direct to way point" should be executed (EVEN IF THERE IS CONFLICT/LOSS OF SEPARATION).
            elif (self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending') and \
                    (((self.durations_of_actions_for_evaluation_[agent][0] == 0 or
                       self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] >=
                       self.durations_of_actions_for_evaluation_[agent][0]) and
                      not self.executing_resume_to_fplan[agent])
                     or (self.durations_of_actions_for_evaluation_[agent][0] == np.inf and
                         not self.executing_direct_to[agent] and not self.executing_FL_change[agent])):

                if agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts:
                    index_of_res_fplan_in_sorted_array = -1
                    sorted_actions = np.argsort(-1 * self.unfixed_acts[0][agent])

                    # plus 2 because of the 2 deterministic actions, ('continue action', 'resume fplan').
                    # We should take into account only 'resume fplan'. For 'continue' action, we just continue the loop.
                    act_index_not_changed_from_continue = -1
                    for act__ in range(self.max_poss_actions_for_evaluation + 2):
                        act_index_not_changed_from_continue += 1
                        temp_act_dur = None
                        if sorted_actions[act__] < self.num_actions_dc + self.num_actions_ds:
                            temp_act_dur = self.total_duration_values[sorted_actions[act__]]
                        elif self.num_actions_dc + self.num_actions_ds <= sorted_actions[act__] < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                            temp_act_dur = 0  # The duration is zero when the duration is unknown.
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                            temp_act_dur = self.interval_between_two_steps

                        # num_dc+num_ds+num_as+num_wp+1 is the deterministic action 'continue'.
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1:
                            act_index_not_changed_from_continue -= 1
                            continue
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2:
                            temp_act_dur = 0  # The duration is zero when the duration is unknown.
                            index_of_res_fplan_in_sorted_array = act_index_not_changed_from_continue
                        else:
                            print("Not a valid action, case 1!! Function get_valid_actions_for_evaluation!!")
                            exit(0)
                        self.durations_of_actions_for_evaluation[agent].append((temp_act_dur, self.timestamp))
                        self.indices_possible_actions[agent].append(sorted_actions[act__])
                        self.possible_actions[agent].append(self.action_values_for_evaluation[sorted_actions[act__]])
                        self.possible_actions_type[agent].append(self.types_of_actions_for_evaluation[sorted_actions[act__]])
                        self.q_values_of_poss_actions[agent].append(self.unfixed_acts[0][agent][sorted_actions[act__]])
                        self.filtered_out_reason[agent].append("Fight phase is: " + str(self.flight_phases[agent])
                                                               if sorted_actions[act__] !=
                                                                  self.num_actions_dc + self.num_actions_ds +
                                                                  self.num_actions_as + self.num_dir_wp + 2
                                                               else "null")
                    self.not_action_due_to_phase_mask[agent] = \
                        (True, self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2,
                         index_of_res_fplan_in_sorted_array)
                else:

                    # num_dc+num_ds+num_as+num_wp+2 means "resume fplan" action
                    self.indices_possible_actions[agent].\
                        extend([self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2
                                for _ in range(self.max_poss_actions_for_evaluation)])
                    self.possible_actions[agent].extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                    self.filtered_out_reason[agent].\
                        extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                    self.durations_of_actions_for_evaluation[agent].\
                        extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            # If the phase of a flight is 'climbing'/'descending' and
            # the action which is in progress is the 'resume fplan'
            # (based on self.durations_of_actions_for_evaluation_[agent][0] == 0 and
            # self.executing_resume_to_fplan[agent]),
            # then the 'continue' action should be selected.
            elif (self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending') and \
                    self.durations_of_actions_for_evaluation_[agent][0] == 0 and self.executing_resume_to_fplan[agent]:

                if agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts:
                    index_of_res_fplan_in_sorted_array = -1
                    sorted_actions = np.argsort(-1 * self.unfixed_acts[0][agent])

                    # plus 2 because of the 2 deterministic actions, ('continue action', 'resume fplan').
                    # We should take into account only 'continue action'. For 'resume fplan', we just continue the loop.
                    act_index_not_changed_from_continue = -1
                    for act__ in range(self.max_poss_actions_for_evaluation + 2):
                        act_index_not_changed_from_continue += 1
                        temp_act_dur = None
                        if sorted_actions[act__] < self.num_actions_dc + self.num_actions_ds:
                            temp_act_dur = self.total_duration_values[sorted_actions[act__]]
                        elif self.num_actions_dc + self.num_actions_ds <= sorted_actions[act__] < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                            temp_act_dur = 0  # The duration is zero when the duration is unknown.
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:
                            temp_act_dur = self.interval_between_two_steps
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1:

                            # Assign zero duration just like in case of "resume fplan" because in testing
                            # we do not separate the actions:
                            # a)"resume fplan"
                            # and
                            # b) "continue" when the flight phase is 'climbing'/'descending' and "resume fplan"
                            #    is going to be executed or is already in progress.
                            temp_act_dur = 0
                            index_of_res_fplan_in_sorted_array = act_index_not_changed_from_continue
                        elif sorted_actions[act__] == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 2:
                            act_index_not_changed_from_continue -= 1
                            continue
                        else:
                            print("Not a valid action, case 2!! Function get_valid_actions_for_evaluation!!")
                            exit(0)

                        self.durations_of_actions_for_evaluation[agent].append((temp_act_dur, self.timestamp))
                        self.indices_possible_actions[agent].append(sorted_actions[act__])
                        self.possible_actions[agent].append(self.action_values_for_evaluation[sorted_actions[act__]])
                        self.possible_actions_type[agent].\
                            append(self.types_of_actions_for_evaluation[sorted_actions[act__]]
                                    if (not sorted_actions[act__] ==
                                            self.num_actions_dc + self.num_actions_ds +
                                            self.num_actions_as + self.num_dir_wp + 1
                                        and not sorted_actions[act__] ==
                                                self.num_actions_dc + self.num_actions_ds +
                                                self.num_actions_as + self.num_dir_wp + 2) else "RFP")
                        self.q_values_of_poss_actions[agent].append(self.unfixed_acts[0][agent][sorted_actions[act__]])
                        self.filtered_out_reason[agent].\
                            append("Fight phase is: " + str(self.flight_phases[agent]) +
                                   " and the action 'RFP' is already in progress."
                                   if sorted_actions[act__] !=
                                      self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp + 1
                                   else "null")
                    self.not_action_due_to_phase_mask[agent] = \
                        (True, self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1,
                         index_of_res_fplan_in_sorted_array)
                else:

                    # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                    self.indices_possible_actions[agent].\
                        extend([self.num_actions_dc + self.num_actions_ds +
                                self.num_actions_as + self.num_dir_wp + 1
                                for _ in range(self.max_poss_actions_for_evaluation)])
                    self.possible_actions[agent].\
                        extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                    self.filtered_out_reason[agent].\
                        extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                    self.durations_of_actions_for_evaluation[agent].\
                        extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            # We assume that the phase is not 'climbing'/'descending', otherwise the previous condition would be True.
            # If a flight is executing the action 'resume fplan', and it does not participate in a
            # conflict/loss of separation, then the action 'continue' should be executed.
            elif self.executing_resume_to_fplan[agent] and \
                    self.durations_of_actions_for_evaluation_[agent][0] == 0 and \
                    not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts):

                # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                self.indices_possible_actions[agent].\
                    extend([self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1
                            for _ in range(self.max_poss_actions_for_evaluation)])
                self.possible_actions[agent].extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.filtered_out_reason[agent].\
                    extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.durations_of_actions_for_evaluation[agent].\
                    extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            # We assume that even if the phase is 'climbing' or 'descending',
            # we should not interrupt the executing action
            # (which is not selected deterministically but by the agent. The cases of
            # the deterministic selection are covered by the second and third previous conditions).
            # The executing actions that we care about in this condition are: "change flight level" and
            # "direct to (any) way point".
            elif (self.executing_FL_change[agent] or self.executing_direct_to[agent]) and \
                    self.durations_of_actions_for_evaluation_[agent][0] == np.inf:

                self.possible_actions[agent].extend([
                    self.action_values_for_evaluation[
                        self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1]
                    for _ in range(self.max_poss_actions_for_evaluation)])  # num_dc+num_ds+num_as+num_wp+1 means

                # "continue" action
                self.indices_possible_actions[agent].extend([
                    self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1
                    for _ in range(self.max_poss_actions_for_evaluation)])

                self.possible_actions_type[agent].extend([
                    self.types_of_actions_for_evaluation[
                        self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1]
                    for _ in range(self.max_poss_actions_for_evaluation)])

                self.q_values_of_poss_actions[agent].extend(
                    [self.unfixed_acts[0][agent][
                         self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1]
                     for _ in range(self.max_poss_actions_for_evaluation)])

                self.filtered_out_reason[agent].extend(["null" for _ in range(self.max_poss_actions_for_evaluation)])
                self.durations_of_actions_for_evaluation[agent].extend(
                    (self.interval_between_two_steps, self.timestamp)
                    for _ in range(self.max_poss_actions_for_evaluation))

                # We need to annotate all possible actions of the current flight
                # (which are the "continue" actions), except the first,
                # as dummy actions in order to be able to distinguish them from the rest of different possible actions.
                self.dummy_possible_actions_mask[agent] = \
                    [False] + [True for _ in range(self.max_poss_actions_for_evaluation - 1)]
                self.action_in_progress[agent] = \
                    self.search_resolution_action(self.confl_resol_act, self.flights_idxs_to_ids[agent],
                                                  self.max_poss_actions_for_evaluation)
                continue

            # Conditions for checking if all cases have been taken into account
            elif (self.executing_FL_change[agent] or self.executing_direct_to[agent]) and \
                    not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts):
                print("There are cases of self.executing_FL_change/self.executing_direct_to "
                      "while there is no conflict/loss of separation that should be taken into account!!")
                exit(0)

            # We assume that there is no action "direct to (any) way point" or "change flight level" in progress,
            # otherwise the previous condition would be true.
            # If the maneuver has just finished and there is no conflict/loss of separation,
            # the default action "resume fplan" should be executed.
            elif not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) and \
                    (agent in self.history_loss_confl) and \
                    ((self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] >=
                      self.durations_of_actions_for_evaluation_[agent][0]) or
                     (self.durations_of_actions_for_evaluation_[agent][0] == np.inf and
                      not self.executing_FL_change[agent] and not self.executing_direct_to[agent])):

                # num_dc+num_ds+num_as+num_wp+2 means the default action (resume fplan)
                self.indices_possible_actions[agent].extend(
                    [self.num_actions_dc + self.num_actions_ds + self.num_actions_as +
                     self.num_dir_wp + 2 for _ in range(self.max_poss_actions_for_evaluation)])
                self.possible_actions[agent].extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.filtered_out_reason[agent].\
                    extend(["-1" for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                self.durations_of_actions_for_evaluation[agent].\
                    extend([-1 for _ in range(self.max_poss_actions_for_evaluation)])  # -1
                continue

            # When the duration of the selected action
            # (not for the actions "direct to (any) way point" and "change flight level")
            # of a flight has not been reached,
            # the corresponding flight should execute the "zero" action,
            # even if this flight participates in conflict/loss of separation or not
            # (but it is in self.history_loss_confl).
            # However, in testing we should report it as "continue" action.
            elif (agent in self.history_loss_confl) and \
                    (self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] <
                     self.durations_of_actions_for_evaluation_[agent][0]) \
                    and self.durations_of_actions_for_evaluation_[agent][0] != np.inf and \
                    not (self.executing_FL_change[agent] or self.executing_direct_to[agent]):

                # num_dc+num_ds+num_as+num_wp+1 means "continue" action
                self.possible_actions[agent].extend(
                    [self.action_values_for_evaluation[
                         self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1]
                     for _ in range(self.max_poss_actions_for_evaluation)])

                self.indices_possible_actions[agent].extend(
                    [self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1
                     for _ in range(self.max_poss_actions_for_evaluation)])

                self.possible_actions_type[agent].extend(
                    [self.types_of_actions_for_evaluation[
                         self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp + 1]
                     for _ in range(self.max_poss_actions_for_evaluation)])

                self.q_values_of_poss_actions[agent].extend(
                    [self.unfixed_acts[0][agent][self.num_actions_dc + self.num_actions_ds +
                                                 self.num_actions_as + self.num_dir_wp + 1]
                     for _ in range(self.max_poss_actions_for_evaluation)])

                self.filtered_out_reason[agent].extend(["null" for _ in range(self.max_poss_actions_for_evaluation)])
                self.durations_of_actions_for_evaluation[agent].\
                    extend((self.interval_between_two_steps, self.timestamp)
                           for _ in range(self.max_poss_actions_for_evaluation))

                # We need to annotate all possible actions of the current flight (which are the "zero" actions),
                # except the first, as dummy actions in order to be able to distinguish them from
                # the rest of different possible actions.
                self.dummy_possible_actions_mask[agent] = \
                    [False] + [True for _ in range(self.max_poss_actions_for_evaluation - 1)]
                self.action_in_progress[agent] = \
                    self.search_resolution_action(self.confl_resol_act, self.flights_idxs_to_ids[agent],
                                                  self.max_poss_actions_for_evaluation)
                continue

            # Conditions for checking if all cases have been taken into account
            elif not (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) and \
                    not (agent in self.history_loss_confl):
                print("The action 'follow flight plan' should have been selected, but it has not!! "
                      "Function get_valid_actions_for_evaluation!!")
                exit(0)
            elif self.flight_phases[agent] == 'climbing' or self.flight_phases[agent] == 'descending':
                print("The action 'direct to next way point' should have been selected!!! "
                      "Function get_valid_actions_for_evaluation!!")
                exit(0)

            # IF a flight is in conflict/loss and:
            #   - there is no non-deterministic action
            #     (apart from 'direct to (any) way point'/'change flight level') in progress
            #     (based on the condition:
            #     self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] >=
            #     self.durations_of_actions_for_evaluation_[agent][0] ),
            #     and there is no action 'direct to (any) way point' or 'change flight level' in progress
            #     (based on the condition:
            #     not (self.executing_FL_change[agent] or self.executing_direct_to[agent]) ),
            #   OR
            #   - a flight is executing the deterministic action 'resume fplan'
            #     (based on the condition:
            #     (self.durations_of_actions_for_evaluation_[agent][0] == 0 and
            #     self.executing_resume_to_fplan[agent]) and
            #     (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts))
            # --> THEN a valid action should be chosen.
            # Note that we assume that the phase is not climbing/descending (due to the previous checking condition).
            elif (agent in self.fls_with_loss_of_separation or agent in self.fls_with_conflicts) and \
                    (((self.timestamp - self.durations_of_actions_for_evaluation_[agent][1] >=
                       self.durations_of_actions_for_evaluation_[agent][0]) and
                      not (self.executing_FL_change[agent] or self.executing_direct_to[agent])) or
                     (self.durations_of_actions_for_evaluation_[agent][0] == np.inf and
                      not self.executing_FL_change[agent] and not self.executing_direct_to[agent]) or
                     (self.durations_of_actions_for_evaluation_[agent][0] == 0 and
                      self.executing_resume_to_fplan[agent])):

                sorted_actions = np.argsort(-1 * self.acts[0][agent])  # Use -1 to inverse sorting (that is descending)
                for poss_action in range(self.max_poss_actions_for_evaluation):

                    # Store filtered out reason and duration of actions
                    if sorted_actions[poss_action] < self.num_actions_dc:
                        self.filtered_out_reason[agent].append("null")
                        self.durations_of_actions_for_evaluation[agent].\
                            append((self.total_duration_values[sorted_actions[poss_action]], self.timestamp))

                    elif self.num_actions_dc <= sorted_actions[poss_action] < \
                            self.num_actions_dc + self.num_actions_ds:

                        if ((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) +
                             self.min_h_speed) + self.actions_list[sorted_actions[poss_action]] < self.min_h_speed:

                            self.filtered_out_reason[agent].\
                                append("Horizontal speed cannot be decreased because it will fall below 178.67 m/s")

                        elif ((self.flight_array[agent, 3] * (self.max_h_speed - self.min_h_speed)) +
                              self.min_h_speed) + self.actions_list[sorted_actions[poss_action]] > self.max_h_speed:

                            self.filtered_out_reason[agent].\
                                append("Horizontal speed cannot be increased because it will exceed 291 m/s")

                        else:
                            self.filtered_out_reason[agent].append("null")

                        self.durations_of_actions_for_evaluation[agent].\
                            append((self.total_duration_values[sorted_actions[poss_action]], self.timestamp))

                    elif self.num_actions_dc + self.num_actions_ds <= sorted_actions[poss_action] < \
                            self.num_actions_dc + self.num_actions_ds + self.num_actions_as:

                        if ((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                             self.min_alt_speed) + self.actions_list[sorted_actions[poss_action]] < self.min_alt_speed:

                            self.filtered_out_reason[agent].\
                                append("Vertical speed cannot be decreased because it will fall below -80.0 feet/s")

                        elif ((self.flight_array[agent, 4] * (self.max_alt_speed - self.min_alt_speed)) +
                              self.min_alt_speed) + self.actions_list[sorted_actions[poss_action]] > self.max_alt_speed:

                            self.filtered_out_reason[agent].\
                                append("Vertical speed cannot be increased because it will exceed 60.0 feet/s")

                        else:
                            self.filtered_out_reason[agent].append("null")

                        self.durations_of_actions_for_evaluation[agent].\
                            append((0, self.timestamp))  # The duration is zero when the duration is unknown.

                    elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= \
                            sorted_actions[poss_action] < \
                            self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                        way_point_index = \
                            abs(((self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp) -
                                 sorted_actions[poss_action]) - self.num_dir_wp)

                        if self.available_wps[agent, way_point_index] == 0:
                            self.filtered_out_reason[agent].\
                                append("The way point " + str(way_point_index + 1) + " is not available")

                        else:
                            self.filtered_out_reason[agent].append("null")

                        self.durations_of_actions_for_evaluation[agent].\
                            append((0, self.timestamp))  # The duration is zero when the duration is unknown.

                    elif sorted_actions[poss_action] == \
                            self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                        self.filtered_out_reason[agent].append("null")

                        # Minimum duration is assigned for "zero" action
                        self.durations_of_actions_for_evaluation[agent].\
                            append((self.interval_between_two_steps, self.timestamp))

                    else:
                        print("There are cases in valid_actions_for_evaluation "
                              "that have not been taken into consideration!!")
                        exit(0)

                    # Store the action itself
                    self.possible_actions[agent].append(self.action_values_for_evaluation[sorted_actions[poss_action]])

                    # Store the index of the action
                    self.indices_possible_actions[agent].append(sorted_actions[poss_action])

                    # Store action type
                    self.possible_actions_type[agent].\
                        append(self.types_of_actions_for_evaluation[sorted_actions[poss_action]])

                    # Store q-value
                    self.q_values_of_poss_actions[agent].append(self.acts[0][agent][sorted_actions[poss_action]])

            else:
                print("There is at least one case that has not been "
                      "taken into account for the filtering of actions in function "
                      "'get_valid_actions_for_evaluation'!!")
                exit(0)

        # Append a dummy action ("zero") to a flight when self.not_action_due_to_phase_mask[agent][0] is not true.
        fixed_indices_possible_actions = \
            [[self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp
                if not self.not_action_due_to_phase_mask[agent][0] and act__ == self.max_poss_actions_for_evaluation
                else self.indices_possible_actions[agent][act__]
              for act__ in range(self.max_poss_actions_for_evaluation + 1)]
             for agent in range(self.n_agent)]

        indices_actions_for_evaluation_ = np.asarray(fixed_indices_possible_actions)
        fixed_durations_of_actions_for_evaluation = \
            [[0 if not self.not_action_due_to_phase_mask[agent][0] and act__ == self.max_poss_actions_for_evaluation
                else
              self.durations_of_actions_for_evaluation[agent][act__]
              for act__ in range(self.max_poss_actions_for_evaluation + 1)]
             for agent in range(self.n_agent)]

        true_actions_for_evaluation_with_first_dim_action_sort_by_rank = \
            [self.get_true_actions(indices_actions_for_evaluation_[:, poss_action],
                                   self.n_agent,
                                   self.active_flights_mask,
                                   dur_vals=
                                   np.expand_dims(np.array([0 if not isinstance(agent[poss_action], tuple)
                                                              else
                                                            fixed_durations_of_actions_for_evaluation[
                                                                agent_indx][poss_action][0]
                                                            for agent_indx, agent in
                                                            enumerate(fixed_durations_of_actions_for_evaluation)]),
                                                  axis=1))
             for poss_action in range(self.max_poss_actions_for_evaluation + 1)]

        self.true_actions_for_evaluation = \
            np.transpose(np.asarray(true_actions_for_evaluation_with_first_dim_action_sort_by_rank), (1, 0, 2))

        self.dummy_possible_actions_mask = np.asarray(self.dummy_possible_actions_mask)

    @staticmethod
    def search_resolution_action(res_action_dataframe, flight_RTKey, max_poss_actions):
        """
        Search in dataframe (in reverse) for the last record of the specific flight.
        Specifically, we want the last record which has not 'no_resolution' in resolutionID,
        and it is the first non-filtered out action in the rank.

        :param res_action_dataframe: pandas dataframe containing the resolution actions.
        :param flight_RTKey: RTKey of the flight for which the last record is requested.
        :param max_poss_actions: number of possible actions to search in for the last record.
        :return: the requested resolution action
        """

        len_df = len(res_action_dataframe.index)
        demanded_res_act_ID = False
        break_flag = False

        for dataframe_row in range(len_df - 1, -1, -1):
            if res_action_dataframe.loc[dataframe_row]['RTKey'] == str(flight_RTKey) and \
                    res_action_dataframe.loc[dataframe_row]['ResolutionID'].split('_')[-1] != 'resolution' and \
                    res_action_dataframe.loc[dataframe_row]['ActionRank'] == 0:
                for poss_act_dataframe_row in range(dataframe_row, dataframe_row + max_poss_actions, 1):
                    if res_action_dataframe.loc[poss_act_dataframe_row]['FilteredOut'] == 'null':
                        demanded_res_act_ID = res_action_dataframe.loc[poss_act_dataframe_row]['ResolutionID']
                        break_flag = True
                        break
                if break_flag:
                    break
        return demanded_res_act_ID

    def update_debugging_files(self):
        if self.debug_:
            print("Episode: {}, step: {}".format(self.i_episode - 1, self.steps - 1))
            self.dict_actions_debug[self.i_episode - 1][self.steps - 1] = \
                {'res_acts': self.true_acts[:self.envs_n_agent[self.env_number]].copy(),
                 'res_acts_ID': self.valid_acts_to_be_executed_ID.copy(),
                 'flight_array': self.flight_array__.copy(),
                 'norm_flight_array': self.flight_array[:self.envs_n_agent[self.env_number]].copy()}
            dict_actions_debug_file = open("./dict_actions_debug.pkl", "wb")
            pickle.dump(self.dict_actions_debug, dict_actions_debug_file)
            dict_actions_debug_file.close()

    def env_step_and_get_returns(self):

        #######Environment step########
        env_step_returns = \
            self.env.step(self.true_acts[:self.envs_n_agent[self.env_number]].copy(),
                          self.valid_acts_to_be_executed_ID.copy())

        self.next_flight_array__ = env_step_returns[0]
        next_edges_ = env_step_returns[1]
        reward_ = env_step_returns[2]
        reward_per_factor_ = env_step_returns[3]
        self.done = env_step_returns[4]
        clipped_actions_ = env_step_returns[5]
        next_available_wps_ = env_step_returns[6]
        next_flight_phases_ = env_step_returns[7]
        next_finished_FL_change_ = env_step_returns[8]
        next_finished_direct_to_ = env_step_returns[9]
        next_finished_resume_to_fplan_ = env_step_returns[10]
        next_executing_FL_change_ = env_step_returns[11]
        next_executing_direct_to_ = env_step_returns[12]
        next_executing_resume_to_fplan_ = env_step_returns[13]

        # Copy arrays and lists to avoid changing of the mutable objects
        self.next_flight_array_ = np.append(self.next_flight_array__.copy(),
                                            [[0] * next_flight_array__.shape[1]
                                             for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                            axis=0) \
                                            if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                            else self.next_flight_array__.copy()
        self.next_edges = next_edges_.copy()
        self.reward = np.append(reward_.copy(),
                                [0 for _ in range(self.n_agent - self.envs_n_agent[self.env_number])], axis=0) \
                                if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                else reward_.copy()
        self.reward_per_factor = np.append(reward_per_factor_.copy(),
                                           [[0] * reward_per_factor_.shape[1]
                                            for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                           axis=0) \
                                           if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                           else reward_per_factor_.copy()

        self.next_available_wps = np.append(next_available_wps_.copy(),
                                            [[0] * next_available_wps_.shape[1]
                                             for _ in range(self.n_agent - self.envs_n_agent[self.env_number])],
                                            axis=0) \
                                            if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                            else next_available_wps_.copy()
        self.next_flight_phases = next_flight_phases_.copy() + \
                                  ['innactive' for _ in range(self.n_agent - self.envs_n_agent[self.env_number])] \
                                  if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                  else next_flight_phases_.copy()
        self.next_finished_FL_change = np.append(next_finished_FL_change_.copy(),
                                                 [False for _ in range(self.n_agent -
                                                                       self.envs_n_agent[self.env_number])],
                                                 axis=0) \
                                                 if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                 else next_finished_FL_change_.copy()
        self.next_finished_direct_to = np.append(next_finished_direct_to_.copy(),
                                                 [False for _ in range(self.n_agent -
                                                                       self.envs_n_agent[self.env_number])],
                                                 axis=0) \
                                                 if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                 else next_finished_direct_to_.copy()
        self.next_finished_resume_to_fplan = np.append(next_finished_resume_to_fplan_.copy(),
                                                       [False for _ in range(self.n_agent -
                                                                             self.envs_n_agent[self.env_number])],
                                                       axis=0) \
                                                       if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                       else next_finished_resume_to_fplan_.copy()
        self.next_executing_FL_change = np.append(next_executing_FL_change_.copy(),
                                                  [False for _ in range(self.n_agent -
                                                                        self.envs_n_agent[self.env_number])],
                                                  axis=0) \
                                                 if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                 else next_executing_FL_change_.copy()
        self.next_executing_direct_to = np.append(next_executing_direct_to_.copy(),
                                                  [False for _ in range(self.n_agent -
                                                                        self.envs_n_agent[self.env_number])],
                                                  axis=0) \
                                                 if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                 else next_executing_direct_to_.copy()
        self.next_executing_resume_to_fplan = np.append(next_executing_resume_to_fplan_.copy(),
                                                        [False for _ in range(self.n_agent -
                                                                              self.envs_n_agent[self.env_number])],
                                                        axis=0) \
                                                        if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                        else next_executing_resume_to_fplan_.copy()

        self.next_active_flights_mask = np.append(self.env.active_flights_mask.copy(),
                                                  [False for _ in range(self.n_agent -
                                                                        self.envs_n_agent[selfenv_number])],
                                                  axis=0) \
                                                  if self.n_agent - self.envs_n_agent[self.env_number] > 0 \
                                                  else self.env.active_flights_mask.copy()

        # Get the timestamp of the next state
        self.next_timestamp = self.env.timestamp

        # Get the normalized features of each flight
        self.next_flight_array = self.get_flight_features(self.next_flight_array_,
                                                          self.next_active_flights_mask,
                                                          self.next_flight_phases)

        # Normalize the edges features, get the adjacency matrix and other useful information about the state
        self.adjacency_matrix_and_norm_edges_features(self.next_edges, next_state=True)

    def compute_norm_reward(self):
        """
        Computes the normalized reward.
        reward_per_factor numpy array: [delta_course, delta_speed, delta_alt_speed, drift_from_exit_point,
                                        altitude_drift_from_exit_point, number_of_losses, number_of_alerts]
        """

        computed_reward = \
            -np.abs((self.next_flight_array_[:, 8] - self.next_flight_array_[:, 8]) /
                    max(self.actions_delta_course)) + \
            self.reward_per_factor[:, 1] + \
            -np.abs((self.next_flight_array_[:, 10] - self.next_flight_array_[:, 10]) /
                    max(self.actions_delta_altitude_speed)) + \
            self.reward_per_factor[:, 3] + \
            self.reward_per_factor[:, 4] / self.altitude_drift_from_exit_point + \
            self.reward_per_factor[:, 5] + \
            self.reward_per_factor[:, 6] + \
            (-(0 if not self.with_RoC_term_reward else self.next_RoC) * self.RoC_term_weight)

        self.normalized_reward = computed_reward / self.r_norm

    def update_episode_stats_for_current_env(self):

        # n_actions+plus_model_action is the action "follow flight plan"
        # num_dc+num_ds+num_as+num_dir_wp+2 is the action "resume to fplan"
        # n_actions is the action "continue"
        condition_for_reward_to_be_computed = \
            self.active_flights_mask & (((self.actions < (self.n_actions + self.plus_model_action)) &
                                         (self.actions != self.num_actions_dc + self.num_actions_ds +
                                          self.num_actions_as + self.num_dir_wp + 2) &
                                         (self.actions != self.n_actions)) |
                                        (((self.actions == self.num_actions_dc + self.num_actions_ds +
                                                           self.num_actions_as + self.num_dir_wp + 2) |
                                          (self.actions == self.n_actions)) &
                                         ((self.np_dur_of_actions[:, 0] == np.inf) |
                                          ((self.np_dur_of_actions[:, 0] == 0) &
                                           (self.np_mask_climb_descend_res_fplan_with_confl_loss |
                                            self.np_mask_res_fplan_after_maneuv)))))

        reward_no_resume_fplan_no_continue_except_some_cases_no_follow_flight_plan_no_non_active_flights = \
            np.where(condition_for_reward_to_be_computed, self.reward, 0)

        self.score.append(
            sum(reward_no_resume_fplan_no_continue_except_some_cases_no_follow_flight_plan_no_non_active_flights) /
            np.count_nonzero(condition_for_reward_to_be_computed)
            if np.count_nonzero(condition_for_reward_to_be_computed) > 0
            else 0)

        self.num_ATC_instr += self.cur_num_ATC_instr
        self.add_nautical_miles += np.sum(self.cur_additional_nautical_miles)
        self.confls_with_positive_tcpa += (len(self.confls_ids_with_positive_tcpa) / 2)
        self.add_duration += np.sum(self.cur_additional_duration)

    def sampling(self):

        if not self.evaluation:

            # Update history of rewards
            self.store_rewards()

            # Get the normalized edges features of each agent's neighbors
            # of the next state based on the adjacency matrix of the previous state
            self.get_norm_edges_feats_based_on_previous_adj_matrix()

            # Add samples to a temporary buffer
            self.buff.add_in_temp_episode_buff(self.flight_array.copy(),
                                               self.actions.copy(),
                                               self.next_flight_array.copy(),
                                               self.normalized_reward.copy(),
                                               self.done,
                                               self.adjacency_one_hot.copy(),
                                               self.active_flights_mask.copy(),
                                               self.norm_edges_feats.copy(),
                                               self.next_norm_edges_feats_based_on_previous_adj_matrix.copy(),
                                               self.fls_with_loss_of_separation.copy(),
                                               self.fls_with_conflicts.copy(),
                                               self.next_fls_with_loss_of_separation.copy(),
                                               self.next_fls_with_conflicts.copy(),
                                               self.history_loss_confl.copy(),
                                               self.next_active_flights_mask.copy(),
                                               self.reward_history.copy(),
                                               copy.deepcopy(self.durations_of_actions),
                                               self.next_timestamp,
                                               self.timestamp,
                                               self.unfixed_acts[0].copy(),
                                               self.np_mask_climb_descend_res_fplan_with_confl_loss.copy(),
                                               self.np_mask_res_fplan_after_maneuv.copy(),
                                               self.duration_dir_to_wp_ch_FL_res_fplan.copy(),
                                               self.next_executing_FL_change.copy(),
                                               self.next_executing_direct_to.copy(),
                                               self.next_flight_phases.copy(),
                                               self.executing_FL_change.copy(),
                                               self.executing_direct_to.copy(),
                                               self.next_available_wps.copy(),
                                               self.next_executing_resume_to_fplan.copy())

            if self.done:

                ########At the end of an episode store the temporarily samples to replay buffer######
                if not self.prioritized_replay_buffer:
                    self.buff.store_episode_samples()

                elif self.prioritized_replay_buffer:

                    self.new_states = []
                    self.dones = []
                    self.active_flights_m = []
                    self.next_fls_with_loss_of_separation_m = []
                    self.next_fls_with_conflicts_m = []
                    self.history_loss_confl_m = []
                    self.next_active_flights_m = []
                    self.durations_of_actions_b = []
                    self.next_timestamp_b = []
                    self.data_needed_for_delayed_update_b = []
                    unfixed_acts_b = []
                    self.np_mask_climb_descend_res_fplan_with_confl_loss_b = []
                    self.np_mask_res_fplan_after_maneuv_b = []
                    self.next_executing_FL_change_b = []
                    self.next_executing_direct_to_b = []
                    self.next_flight_phases_b = []
                    self.executing_FL_change_b = []
                    self.executing_direct_to_b = []
                    self.next_available_wps_b = []
                    self.next_executing_resume_fplan_b = []

                    self.num_of_different_matrices_needed = 3
                    for i_ in range(num_of_different_matrices_needed):
                        self.new_states.append([])

                    self.list_temp_episode_samples = self.buff.get_temp_episode_samples()
                    self.buff.reset_temp_episode_buff()

                    for sample in self.list_temp_episode_samples:

                        # Compute q-values of valid actions and valid-max-next-target-q-values
                        # which are needed for the prioritized replay buffer
                        self.new_states[0].append(sample[2])
                        self.new_states[1].append(sample[5])
                        self.new_states[2].append(sample[8])

                        self.dones.append(sample[4])
                        self.active_flights_m.append(sample[6])
                        self.next_fls_with_loss_of_separation_m.append(sample[11])
                        self.next_fls_with_conflicts_m.append(sample[12])
                        self.history_loss_confl_m.append(sample[13])
                        self.next_active_flights_m.append(sample[14])
                        self.durations_of_actions_b.append(sample[16])
                        self.next_timestamp_b.append(sample[17])
                        self.data_needed_for_delayed_update_b.append(sample[19])
                        unfixed_acts_b.append(sample[20])
                        self.next_executing_direct_to_b.append(sample[21])
                        self.next_executing_FL_change_b.append(sample[22])
                        self.np_mask_res_fplan_after_maneuv_b.append(sample[23])
                        self.np_mask_climb_descend_res_fplan_with_confl_loss_b.append(sample[24])
                        self.next_flight_phases_b.append(sample[25])
                        self.executing_FL_change_b.append(sample[26])
                        self.executing_direct_to_b.append(sample[27])
                        self.next_available_wps_b.append(sample[28])
                        self.next_executing_resume_fplan_b.append(sample[29])

                    self.dones = np.asarray(self.dones)
                    self.active_flights_m = np.asarray(self.active_flights_m)
                    self.next_active_flights_m = self.np.asarray(self.next_active_flights_m)
                    unfixed_acts_b = np.asarray(unfixed_acts_b)

                    for i_ in range(3):
                        self.new_states[i_] = np.asarray(self.new_states[i_])

                    # Get predictions for the next states of the collected samples using the target network
                    self.predict_q_values_for_next_states_using_target_net(len(self.list_temp_episode_samples))

                    # Filter maxq
                    self.get_maxq_valid_actions(self.list_temp_episode_samples)

                    # Store the samples to PER
                    self.priorit_buff.observe(self.list_temp_episode_samples.copy(),
                                              self.maxq_actions.copy(),
                                              self.n_agent,
                                              self.GAMMA,
                                              unfixed_acts_b.copy(),
                                              self.different_target_q_values_mask.copy(),
                                              self.default_action_q_value_mask.copy(),
                                              self.np_mask_res_fplan_after_maneuv_b.copy(),
                                              self.np_mask_climb_descend_res_fplan_with_confl_loss_b.copy(),
                                              self.not_use_max_next_q_mask.copy(),
                                              self.next_flight_phases_b.copy())

                    if self.i_episode > self.episode_before_train and self.env_number == len(self.env_list) - 1:
                        self.priorit_buff.update_beta()

    def store_rewards(self):

        for agent in range(self.n_agent):
            if self.end_of_maneuver[agent]:
                self.reward_history[agent] = [self.normalized_reward[agent]]
            elif not self.end_of_maneuver[agent] and self.durations_of_actions[agent][0] != 0:
                self.reward_history[agent].append(self.normalized_reward[agent])
            elif not self.end_of_maneuver[agent] and self.durations_of_actions[agent][0] == 0 and \
                    (self.np_mask_climb_descend_res_fplan_with_confl_loss[agent] or
                     self.np_mask_res_fplan_after_maneuv[agent]):
                self.reward_history[agent].append(self.normalized_reward[agent])
            elif not self.end_of_maneuver[agent] and self.durations_of_actions[agent][0] == 0 and \
                    not (self.np_mask_climb_descend_res_fplan_with_confl_loss[agent] or
                         self.np_mask_res_fplan_after_maneuv[agent]):
                self.reward_history[agent] = [self.normalized_reward[agent]]
            else:
                print("There is at least one case that has not been taken into account in store_rewards function!!")
                exit(0)

    def get_norm_edges_feats_based_on_previous_adj_matrix(self):

        agents_edges_features = [{} for _ in range(self.n_agent)]
        RoC = np.zeros(self.n_agent, dtype=float)
        rateOfClosureH = np.zeros(self.n_agent, dtype=float)
        rateOfClosureV = np.zeros(self.n_agent, dtype=float)

        # If self.next_edges is an empty list then this means that there are not edges with loss of separation
        if isinstance(self.next_edges, np.ndarray):
            for j_ in range(self.next_edges.shape[0]):
                if self.next_edges[j_, 11] == 1:  # If edge is not referred to a flight and itself

                    # If t_first_cp and t_closest_cp are not NaN
                    # (which means that there is a conflict)
                    # and there is a conflict or loss, and 0 <= t_closest_cp <= 600 secs,
                    # we should compute RoC
                    if not np.isnan(self.next_edges[j_, 12]) and not np.isnan(self.next_edges[j_, 27]) and \
                            (self.next_edges[j_, 31] == 1 or self.next_edges[j_, 30] == 1) and \
                            0 <= self.next_edges[j_, 27] <= self.t_CPA_threshold and \
                            self.next_edges[j_, 27] > self.next_edges[j_, 12]:

                        # Compute RoC if t_closest_cp > t_first_cp
                        RoC_value, \
                        rateOfClosureH[self.env.flight_index[self.next_edges[j_, 0]]['idx']], \
                        rateOfClosureV[self.env.flight_index[self.next_edges[j_, 0]]['idx']], \
                        relSpeedV, \
                        relSpeedH = compute_ROC(self.next_edges, self.max_rateOfClosureHV, j_)
                        RoC[self.env.flight_index[self.next_edges[j_, 0]]['idx']] += RoC_value

                    agents_edges_features[self.env.flight_index[self.next_edges[j_, 0]]['idx']][
                        self.env.flight_index[self.next_edges[j_, 1]]['idx']] = \
                        np.append(self.next_edges[j_, 2:2 + self.num_edges_features],
                                  [rateOfClosureV[self.env.flight_index[self.next_edges[j_, 0]]['idx']],
                                   rateOfClosureH[self.env.flight_index[self.next_edges[j_, 0]]['idx']]])

        dummy_edges_feats = np.zeros(self.num_edges_features + self.num_edges_feats_for_ROC)
        dummy_edges_feats[:] = np.inf

        adj_edges_feat = [[dummy_edges_feats] +
                          [agents_edges_features[j_][j__]
                           if j__ in agents_edges_features[j_]
                           else dummy_edges_feats
                           for j__ in self.adjacency_matrix[j_][1:]]
                          for j_ in range(self.n_agent)]

        adj_edges_feat = np.asarray(adj_edges_feat)

        self.next_norm_edges_feats_based_on_previous_adj_matrix = \
            np.ones((self.n_agent, self.neighbors_observed + 1,
                     self.num_edges_features + 2 + self.num_edges_feats_for_ROC))
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 0] = \
            np.where(adj_edges_feat[:, 1:, 0] != np.inf, adj_edges_feat[:, 1:, 0] / self.T_cpa,
                     self.mask_edges_features[0])  # t_cpa
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 1] = \
            np.where(adj_edges_feat[:, 1:, 1] != np.inf, adj_edges_feat[:, 1:, 1] / self.D_cpa,
                     self.mask_edges_features[1])  # d_cpa
        with np.errstate(invalid='ignore'):
            self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 2] = \
                np.where(adj_edges_feat[:, 1:, 2] != np.inf,
                         np.cos(adj_edges_feat[:, 1:, 2]), self.mask_edges_features[2])  # aij
            self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 3] = \
                np.where(adj_edges_feat[:, 1:, 2] != np.inf,
                         np.sin(adj_edges_feat[:, 1:, 2]), self.mask_edges_features[3])  # aij
            self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 4] = \
                np.where(adj_edges_feat[:, 1:, 3] != np.inf,
                         np.cos(adj_edges_feat[:, 1:, 3]), self.mask_edges_features[4])  # bij
            self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 5] = \
                np.where(adj_edges_feat[:, 1:, 3] != np.inf,
                         np.sin(adj_edges_feat[:, 1:, 3]), self.mask_edges_features[5])  # bij
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 6] = \
            np.where(adj_edges_feat[:, 1:, 4] != np.inf,
                     adj_edges_feat[:, 1:, 4] / self.V_dcpa, self.mask_edges_features[6])  # v_dcpa
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] = \
            np.where(adj_edges_feat[:, 1:, 5] != np.inf,
                     adj_edges_feat[:, 1:, 5] / self.D_cp, self.mask_edges_features[7])  # d_cp
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] = \
            np.where(adj_edges_feat[:, 1:, 6] != np.inf,
                     adj_edges_feat[:, 1:, 6] / self.T_cp,self.mask_edges_features[8])  # t_cp
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 9] = \
            np.where(adj_edges_feat[:, 1:, 7] != np.inf,
                     adj_edges_feat[:, 1:, 7] / self.H_dij, self.mask_edges_features[9])  # h_dij
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 10] = \
            np.where(adj_edges_feat[:, 1:, 8] != np.inf,
                     adj_edges_feat[:, 1:, 8] / self.V_dij, self.mask_edges_features[10])  # v_dij
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 11] = \
            np.where(adj_edges_feat[:, 1:, 9] != np.inf,
                     adj_edges_feat[:, 1:, 9] / self.max_rateOfClosureHV,
                     self.mask_edges_features[11])  # rateOfClosureV
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 12] = \
            np.where(adj_edges_feat[:, 1:, 10] != np.inf,
                     adj_edges_feat[:, 1:, 10] / self.max_rateOfClosureHV,
                     self.mask_edges_features[12])  # rateOfClosureH

        # Set NaN values of tcp and dcp equal to 20 or -20 according to the previous edges values
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] = \
            np.where(np.isnan(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7]) *
                     (self.norm_edges_feats[:, 1:, 7] >= 0), 20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] = \
            np.where(np.isnan(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7]) *
                     (self.norm_edges_feats[:, 1:, 7] < 0), -20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] = \
            np.where(np.isnan(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8]) *
                     (self.norm_edges_feats[:, 1:, 8] >= 0), 20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] = \
            np.where(np.isnan(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8]) *
                     (self.norm_edges_feats[:, 1:, 8] < 0), -20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8])

        # Block normalized tcp and dcp to be in range [-20, 20]
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] = \
            np.where(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] > 20.0, 20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] = \
            np.where(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7] < -20.0, -20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 7])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] = \
            np.where(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] > 20.0, 20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8])
        self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] = \
            np.where(self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8] < -20.0, -20.0,
                     self.next_norm_edges_feats_based_on_previous_adj_matrix[:, 1:, 8])

        """
        #Check if any edge-feature of any flight is out of the range [-20,20]

        for agg in range(self.n_agent):
            if next_active_flights_mask[agg] and 
                ((self.next_norm_edges_feats_based_on_previous_adj_matrix[agg] > 20).any() or 
                (self.next_norm_edges_feats_based_on_previous_adj_matrix[agg] < -20).any()):
                
                print("\nAgent self.next_norm_edges_feats_based_on_previous_adj_matrix: " + str(agg))
                print(self.next_norm_edges_feats_based_on_previous_adj_matrix[agg])
        """

    def predict_q_values_for_next_states_using_target_net(self, batch_size_):

        self.target_q_values = self.DGN_m.model_t_predict(self.new_states)

        # Mask to monitor which agents have target_q_values for different timestep rather than the next.
        self.different_target_q_values_mask = [[False for _ in range(self.n_agent)] for _ in range(batch_size_)]

        # For the agents that their selected action has duration greater than the interval between two steps **,
        # we should predict q_values for the state right after the end of the duration
        # using the corresponding data stored ("self.data_needed_for_delayed_update_b")
        # except for those that did not reach the end of the duration because they became
        # inactive or because the episode ended.
        #
        # ** included non-deterministic actions 'direct to (any) way point' and 'change FL',
        # and deterministic actions 'resume fplan' and 'continue'.
        #
        # For the deterministic actions this is True only when
        # self.np_mask_climb_descend_res_fplan_with_confl_loss_b[sample_][agent] or
        # self.np_mask_res_fplan_after_maneuv_b[sample_][agent] is True
        # and at the next state they are still in progress, and when these actions end the corresponding agent
        # should be in conflict/loss of separation.
        # Note that in the case of self.np_mask_res_fplan_after_maneuv_b, the action could end prematurely
        # if a conflict/loss of separation is occurred,
        # and in this case we should get the max_next_q, but only if this is not the next state.
        # Also, if the case is self.np_mask_climb_descend_res_fplan_with_confl_loss_b,
        # we should check if at the next state the phase is the same,
        # or if it is not, we should check if there is not any conflict/loss of separation for this agent,
        # otherwise we should not get the max_next_q based on "self.data_needed_for_delayed_update_b".
        for sample_ in range(batch_size_):
            for agent in range(self.n_agent):

                # Indices 3, 4, 5, 6 of "self.data_needed_for_delayed_update_b" are the
                # "next_fls_with_loss_of_separation", "next_fls_with_conflicts", "next_active_flights_mask",
                # and "done" , respectively.
                if ((self.durations_of_actions_b[sample_][agent][1] != 0 and
                     self.durations_of_actions_b[sample_][agent][0] != np.inf and
                     self.next_timestamp_b[sample_] - self.durations_of_actions_b[sample_][agent][1] <
                     self.durations_of_actions_b[sample_][agent][0]) or
                    (self.durations_of_actions_b[sample_][agent][0] == np.inf and
                     ((self.next_executing_direct_to_b[sample_][agent] and
                       (agent in self.data_needed_for_delayed_update_b[sample_][agent][3] or
                        agent in self.data_needed_for_delayed_update_b[sample_][agent][4])) or
                      self.next_executing_FL_change_b[sample_][agent])) or
                    (self.durations_of_actions_b[sample_][agent][0] == 0 and
                    (((self.np_mask_res_fplan_after_maneuv_b[sample_][agent]
                       and not (agent in self.next_fls_with_conflicts_m[sample_] or
                                agent in self.next_fls_with_loss_of_separation_m[sample_])) or
                      (self.np_mask_climb_descend_res_fplan_with_confl_loss_b[sample_][agent] and
                       ((self.next_flight_phases_b[sample_][agent] == 'climbing' or
                         self.next_flight_phases_b[sample_][agent] == 'descending') or
                        (not (self.next_flight_phases_b[sample_][agent] == 'climbing' or
                              self.next_flight_phases_b[sample_][agent] == 'descending') and
                         not (agent in self.next_fls_with_conflicts_m[sample_] or
                              agent in self.next_fls_with_loss_of_separation_m[sample_]))))) and
                     self.next_executing_resume_fplan_b[sample_][agent] and
                     (agent in self.data_needed_for_delayed_update_b[sample_][agent][3] or
                      agent in self.data_needed_for_delayed_update_b[sample_][agent][4])))) and \
                        self.data_needed_for_delayed_update_b[sample_][agent][5][agent] and \
                        not self.data_needed_for_delayed_update_b[sample_][agent][6]:

                    temp_new_state_ = []
                    for matrix_needed in range(self.num_of_different_matrices_needed):
                        temp_new_state_.\
                            append(np.asarray([self.data_needed_for_delayed_update_b[sample_][agent][matrix_needed]]))

                    temp_q_values_ = self.DGN_m.model_t_predict(temp_new_state_)
                    self.target_q_values[sample_, agent] = temp_q_values_[0][agent].copy()
                    self.different_target_q_values_mask[sample_][agent] = True

    def get_maxq_valid_actions(self, btch):

        self.target_q_values = np.array(self.target_q_values)

        # default action "resume fplan"
        default_action_q_value = \
            self.target_q_values[:, :, self.num_actions_dc + self.num_actions_ds +
                                       self.num_actions_as + self.num_dir_wp + 2].copy()

        # The following is for checking if the default action ("resume fplan")
        # for max_next_q is assigned to the correct cases.
        self.default_action_q_value_mask = \
            [[False for _ in range(self.target_q_values[0].shape[0])] for _ in range(self.target_q_values.shape[0])]

        # Mask to keep track of which flights we should use the max_next_q in function 'update_q_values'.
        self.not_use_max_next_q_mask = \
            [[False for _ in range(self.target_q_values[0].shape[0])] for _ in range(self.target_q_values.shape[0])]

        # All but the default actions' q-value (which are "continue" and "resume fplan" actions)
        self.target_q_values = self.target_q_values[:, :, :-self.plus_model_action]

        self.maxq_actions = np.zeros((self.target_q_values.shape[0], self.target_q_values[0].shape[0]), dtype=float)
        for agent in range(self.target_q_values[0].shape[0]):
            for batch_ in range(self.target_q_values.shape[0]):

                # IF:
                # - the episode has just finished, OR
                # - the agent has just become inactive, OR
                # - it was inactive before the next state,
                # --> THEN we should not compute the max_next_q.
                # The first two cases include the case of any non-deterministic action that ends at the next state.
                #
                # Additionally, IF:
                # - an agent has not been found so far to participate in conflict/loss of separation,
                # --> THEN default action 'follow flight plan' will be executed. This action is not evaluated.
                #
                # Also, WHEN:
                # - a flight has participated in a conflict/loss of separation at some point in the current episode,
                #   and now the duration of its selected action
                #   (NOT for the actions 'direct to (any) way point'/'change FL')
                #   has not been reached,
                #   and it is anticipated not to be done because:
                #       - the flight will become inactive, OR
                #       - the episode will end,
                # --> its max_next_q should not be calculated and used.
                #
                # Also, WHEN:
                # - the flight will become inactive at any timepoint before the end of
                #   the action being in progress at the next state, OR
                # - the episode will end at any timepoint before the end of the action being progress at the next state,
                # - AND:
                #       - the selected action is 'direct to (any) way point'/'change FL' and is still in progress.
                #       - This is also True when:
                #           - self.np_mask_res_fplan_after_maneuv_b[batch_][agent] is True and at the next state
                #             the deterministic action 'resume fplan' will be still in progress and:
                #               - there is no conflict/loss, or
                #               - there is conflict/loss but the fight phase is 'climbing'/'descending' .
                #               NOTE: we only check the next state because if at any subsequent state there is
                #                     conflict/loss but the fight phase is not 'climbing'/'descending',
                #                     the Replay buffer would stop store rewards in that reward history.
                #        - Also, it is True:
                #           - IF self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] is True and
                #             at the next state the deterministic action 'resume fplan' is
                #             still in progress and the phase has not changed, OR
                #           - IF it has changed and there is no conflict/loss of separation.
                #           NOTE: we only check the next state because if at any subsequent state there is
                #                 conflict/loss but the fight phase has changed,
                #                 the Replay buffer would stop store rewards in that reward history.
                # --> its max_next_q should not be calculated and used
                #
                # - Furthermore, IF:
                #   - self.np_mask_res_fplan_after_maneuv_b[batch_][agent] is True,
                #     and the deterministic action 'resume fplan' has just finished
                #     and it does not exist any conflict/loss of separation, OR
                #       - the deterministic action has not finished yet, but at the end of its execution
                #         it will not exist any conflict/loss of separation
                #         (this is true even in the case that a conflict/loss of separation will exist
                #          before the end of the execution but the flight phase will be 'climbing'/'descending')
                #   - OR self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] is True,
                #     and the deterministic action 'resume fplan' has just finished and
                #     it does not exist any conflict/loss of separation, OR
                #       - the deterministic action has not finished yet, but at the next state its phase will
                #         have not changed and at the end of its execution it will not
                #         exist any conflict/loss of separation, OR
                #       - at the next state its phase will have changed, but it will not exist any
                #         conflict/loss of separation,
                #         and at the end of its execution (of the deterministic action being in progress)
                #         it will not exist any conflict/loss of separation
                #       NOTE: we only check the next state because if at any subsequent state there is conflict/loss
                #             but the fight phase has changed,
                #             the Replay buffer would stop store rewards in that reward history.
                #   - OR the deterministic action 'resume fplan'/'continue' is in progress and
                #     neither self.np_mask_res_fplan_after_maneuv_b[batch_][agent]
                #     nor self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] is True
                # --> we should not compute the max_next_q
                #
                if self.dones[batch_] or not self.active_flights_m[batch_, agent] or \
                        not self.next_active_flights_m[batch_, agent] or \
                        (not (agent in self.history_loss_confl_m[batch_])) or \
                        (self.durations_of_actions_b[batch_][agent][0] == 0 and
                         not (self.np_mask_res_fplan_after_maneuv_b[batch_][agent] or
                              self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent])) or \
                        ((agent in self.history_loss_confl_m[batch_]) and
                         ((self.next_timestamp_b[batch_] - self.durations_of_actions_b[batch_][agent][1] <
                           self.durations_of_actions_b[batch_][agent][0] != np.inf) or
                          (self.durations_of_actions_b[batch_][agent][0] == np.inf and
                           (self.next_executing_direct_to_b[batch_][agent] or
                            self.next_executing_FL_change_b[batch_][agent])) or
                          (self.durations_of_actions_b[batch_][agent][0] == 0 and
                           ((self.np_mask_res_fplan_after_maneuv_b[batch_][agent] and
                             self.next_executing_resume_fplan_b[batch_][agent] and
                             (not (agent in self.next_fls_with_conflicts_m[batch_] or
                                   agent in self.next_fls_with_loss_of_separation_m[batch_]) or
                              ((agent in self.next_fls_with_conflicts_m[batch_] or
                                agent in self.next_fls_with_loss_of_separation_m[batch_]) and
                               (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                self.next_flight_phases_b[batch_][agent] == 'descending')))) or
                            (self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] and
                             self.next_executing_resume_fplan_b[batch_][agent] and
                             ((self.next_flight_phases_b[batch_][agent] == 'climbing' or
                               self.next_flight_phases_b[batch_][agent] == 'descending') or
                              (not (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                    self.next_flight_phases_b[batch_][agent] == 'descending') and
                               not (agent in self.next_fls_with_conflicts_m[batch_] or
                                    agent in self.next_fls_with_loss_of_separation_m[batch_]))))))) and
                         (not self.data_needed_for_delayed_update_b[batch_][agent][5][agent] or
                          self.data_needed_for_delayed_update_b[batch_][agent][6])) or \
                        (self.durations_of_actions_b[batch_][agent][0] == 0 and
                         ((self.np_mask_res_fplan_after_maneuv_b[batch_][agent] and
                           ((not self.next_executing_resume_fplan_b[batch_][agent] and
                             not (agent in self.next_fls_with_conflicts_m[batch_] or
                                  agent in self.next_fls_with_loss_of_separation_m[batch_])) or
                            (self.next_executing_resume_fplan_b[batch_][agent] and
                             (not (agent in self.next_fls_with_conflicts_m[batch_] or
                                   agent in self.next_fls_with_loss_of_separation_m[batch_]) or
                              ((agent in self.next_fls_with_conflicts_m[batch_] or
                                agent in self.next_fls_with_loss_of_separation_m[batch_]) and
                               (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                self.next_flight_phases_b[batch_][agent] == 'descending'))) and not
                             (agent in self.data_needed_for_delayed_update_b[batch_][agent][3] or
                              agent in self.data_needed_for_delayed_update_b[batch_][agent][4])))) or
                          (self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] and
                           ((not self.next_executing_resume_fplan_b[batch_][agent] and
                             not (agent in self.next_fls_with_conflicts_m[batch_] or agent in
                                  self.next_fls_with_loss_of_separation_m[batch_])) or
                            (self.next_executing_resume_fplan_b[batch_][agent] and
                             ((self.next_flight_phases_b[batch_][agent] == 'climbing' or
                               self.next_flight_phases_b[batch_][agent] == 'descending') or
                              (not (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                    self.next_flight_phases_b[batch_][agent] == 'descending') and
                               not (agent in self.next_fls_with_conflicts_m[batch_] or
                                    agent in self.next_fls_with_loss_of_separation_m[batch_]))) and
                             not (agent in self.data_needed_for_delayed_update_b[batch_][agent][3] or
                                  agent in self.data_needed_for_delayed_update_b[batch_][agent][4])))))):

                    self.not_use_max_next_q_mask[batch_][agent] = True
                    continue

                # Default action ("resume fplan") at t+1 timestep if the flight is not in conflict/loss of separation
                # (or its phase is 'climbing/'descending')
                # but at some previous timestep t-k it was
                # (where k is in the range [0, duration_length])
                # and the duration of the selected
                # action has just been reached
                # (apart from the actions with duration, this is also true for both the actions
                # 'direct to (any) way point' and 'change FL').
                #
                # Additionally, when a flight has participated in a conflict/loss of separation
                # at some point in the current episode,
                # and now the duration of its selected action has not been reached, but it is anticipated to be done,
                # and at the end of this duration, this flight will not participate in a conflict/loss of separation
                # (or its phase will be 'climbing/'descending'),
                # its max_next_q should be the value of the default action ("resume fplan") for the state
                # right after the end of the duration.
                # Furthermore, if an agent is executing the deterministic action 'resume fplan'/'continue' and
                # the self.np_mask_res_fplan_after_maneuv_b or
                # self.np_mask_climb_descend_res_fplan_with_confl_loss_b is True,
                # we should take into consideration two cases:
                #   -IF at the next state, the deterministic action will be over,
                #    the flight will be in conflict/loss of separation and its phase will be 'climbing'/'descending',
                #    THEN the max_next_q of the action 'resume fplan' should be selected, OR
                #   -IF at the next state, the deterministic action will not be over, and at some next steps later,
                #    when this action will be over, the flight will be in conflict/loss and its phase will be
                #    'climbing'/'descending',
                #    THEN the max_next_q of the action 'resume fplan' should be selected.
                #
                # At these conditions we assume that the corresponding flight is/will be active at the
                # current/next timestep, and it is listed in history_loss_confl because of the previous condition
                # where all of these are checked.
                elif ((not (agent in self.next_fls_with_loss_of_separation_m[batch_] or
                            agent in self.next_fls_with_conflicts_m[batch_]) or
                       (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                        self.next_flight_phases_b[batch_][agent] == 'descending')) and
                      (self.next_timestamp_b[batch_] - self.durations_of_actions_b[batch_][agent][1] >=
                       self.durations_of_actions_b[batch_][agent][0]) and
                      self.durations_of_actions_b[batch_][agent][0] != np.inf and
                      self.durations_of_actions_b[batch_][agent][0] != 0) or \
                        ((self.next_timestamp_b[batch_] - self.durations_of_actions_b[batch_][agent][1] <
                          self.durations_of_actions_b[batch_][agent][0]) and
                         self.durations_of_actions_b[batch_][agent][0] != np.inf and
                         self.durations_of_actions_b[batch_][agent][0] != 0 and
                         self.data_needed_for_delayed_update_b[batch_][agent][5][agent] and
                         not self.data_needed_for_delayed_update_b[batch_][agent][6] and
                         ((agent not in self.data_needed_for_delayed_update_b[batch_][agent][3] and
                           agent not in self.data_needed_for_delayed_update_b[batch_][agent][4]) or
                          (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                           self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending'))) or \
                        (self.durations_of_actions_b[batch_][agent][0] == np.inf and
                         (((not self.next_executing_FL_change_b[batch_][agent] and
                            (self.executing_FL_change_b[batch_][agent] or
                             (self.num_actions_dc + self.num_actions_ds <= btch[batch_][1][agent] <
                              self.num_actions_dc + self.num_actions_ds + self.num_actions_as)) and
                            (not (agent in self.next_fls_with_loss_of_separation_m[batch_] or
                                  agent in self.next_fls_with_conflicts_m[batch_]) or
                             (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                              self.next_flight_phases_b[batch_][agent] == 'descending'))) or
                           (self.next_executing_FL_change_b[batch_][agent] and
                            (self.executing_FL_change_b[batch_][agent] or
                            (self.num_actions_dc + self.num_actions_ds <= btch[batch_][1][agent] <
                             self.num_actions_dc + self.num_actions_ds + self.num_actions_as)) and
                            ((agent not in self.data_needed_for_delayed_update_b[batch_][agent][3] and
                              agent not in self.data_needed_for_delayed_update_b[batch_][agent][4]) or
                             (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                              self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending')) and
                            self.data_needed_for_delayed_update_b[batch_][agent][5][agent] and not
                            self.data_needed_for_delayed_update_b[batch_][agent][6])) or
                          ((not self.next_executing_direct_to_b[batch_][agent] and
                            (self.executing_direct_to_b[batch_][agent] or
                             (self.num_actions_dc + self.num_actions_ds + self.num_actions_as <=
                              btch[batch_][1][agent] <
                              self.num_actions_dc + self.num_actions_ds + self.num_actions_as + num_dir_wp)) and
                            (not (agent in self.next_fls_with_loss_of_separation_m[batch_] or
                                  agent in self.next_fls_with_conflicts_m[batch_]) or
                             (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                              self.next_flight_phases_b[batch_][agent] == 'descending'))) or
                           (self.next_executing_direct_to_b[batch_][agent] and
                            (self.executing_direct_to_b[batch_][agent] or
                             (self.num_actions_dc + self.num_actions_ds + self.num_actions_as <=
                              btch[batch_][1][agent] <
                              self.num_actions_dc + self.num_actions_ds + self.num_actions_as + num_dir_wp)) and
                            ((agent not in self.data_needed_for_delayed_update_b[batch_][agent][3] and
                              agent not in self.data_needed_for_delayed_update_b[batch_][agent][4]) or
                             (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                              self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending')) and
                            self.data_needed_for_delayed_update_b[batch_][agent][5][agent] and
                            not self.data_needed_for_delayed_update_b[batch_][agent][6])))) or \
                        (self.durations_of_actions_b[batch_][agent][0] == 0 and
                         (self.np_mask_res_fplan_after_maneuv_b[batch_][agent] and
                          ((not self.next_executing_resume_fplan_b[batch_][agent] and
                            ((agent in self.next_fls_with_conflicts_m[batch_] or
                              agent in self.next_fls_with_loss_of_separation_m[batch_]) and
                             (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                              self.next_flight_phases_b[batch_][agent] == 'descending'))) or
                           (self.next_executing_resume_fplan_b[batch_][agent] and
                           (not (agent in self.next_fls_with_conflicts_m[batch_] or
                                 agent in self.next_fls_with_loss_of_separation_m[batch_]) or
                            ((agent in self.next_fls_with_conflicts_m[batch_] or
                              agent in self.next_fls_with_loss_of_separation_m[batch_]) and
                             (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                              self.next_flight_phases_b[batch_][agent] == 'descending'))) and
                            (agent in self.data_needed_for_delayed_update_b[batch_][agent][3] or
                             agent in self.data_needed_for_delayed_update_b[batch_][agent][4]) and
                            (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                             self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending')))) or
                         (self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] and
                          ((not self.next_executing_resume_fplan_b[batch_][agent] and
                            (agent in self.next_fls_with_conflicts_m[batch_] or
                             agent in self.next_fls_with_loss_of_separation_m[batch_]) and
                            (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                             self.next_flight_phases_b[batch_][agent] == 'descending')) or
                           (self.next_executing_resume_fplan_b[batch_][agent] and
                            ((self.next_flight_phases_b[batch_][agent] == 'climbing' or
                              self.next_flight_phases_b[batch_][agent] == 'descending') or
                             (not (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                   self.next_flight_phases_b[batch_][agent] == 'descending') and
                              not (agent in self.next_fls_with_conflicts_m[batch_] or
                                   agent in self.next_fls_with_loss_of_separation_m[batch_]))) and
                            (agent in self.data_needed_for_delayed_update_b[batch_][agent][3] or
                             agent in self.data_needed_for_delayed_update_b[batch_][agent][4]) and
                            (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                             self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending'))))):

                    self.maxq_actions[batch_, agent] = default_action_q_value[batch_][agent]
                    self.default_action_q_value_mask[batch_][agent] = True
                    continue

                # - IF:
                #   - the duration of a non-deterministic action has just been reached, OR
                #   - the duration was equal to the interval between two steps, OR
                #   - self.np_mask_res_fplan_after_maneuv_b[batch_][agent] or
                #     self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] is True
                #
                #    and now the corresponding flight is in conflict/loss of separation and the flight phase is
                #    not 'climbing'/'descending',
                #
                #    --> THEN a valid max_next_q should be selected (condition 1).
                #
                # - OR IF:
                #       - the duration of a non-deterministic action has not been reached yet,
                #         and the corresponding flight is anticipated to complete its
                #         action (i.e. the duration will not be interrupted), OR
                #       - self.np_mask_res_fplan_after_maneuv_b[batch_][agent] or
                #         self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent] is True
                #         and at the next state the flight will not be in conflict/loss, OR
                #         - the flight will be in conflict/loss and the phase will be 'climbing'/'descending'
                #
                #       and this flight is anticipated to be in a conflict/loss of separation after the end of the
                #       duration and its flight phase will not be 'climbing'/'descending',
                #
                #       --> THEN a valid max_next_q (for the state after the end of the duration)
                #           should be selected (condition 2).
                #
                # At these conditions we assume that the corresponding flight is/will be active
                # at the current/next timestep, and it is listed in history_loss_confl
                # because of the first condition where all of these cases are checked.
                else:

                    case_1 = ((agent in self.next_fls_with_loss_of_separation_m[batch_] or
                               agent in self.next_fls_with_conflicts_m[batch_]) and
                              not (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                   self.next_flight_phases_b[batch_][agent] == 'descending') and
                              (((self.next_timestamp_b[batch_] - self.durations_of_actions_b[batch_][agent][1] >=
                                 self.durations_of_actions_b[batch_][agent][0]) and
                                self.durations_of_actions_b[batch_][agent][0] != np.inf and
                                self.durations_of_actions_b[batch_][agent][0] != 0) or
                               (self.durations_of_actions_b[batch_][agent][0] == np.inf and
                                not (self.next_executing_FL_change_b[batch_][agent] and
                                     self.next_executing_direct_to_b[batch_][agent])) or
                               (self.durations_of_actions_b[batch_][agent][0] == 0 and
                                ((self.np_mask_res_fplan_after_maneuv_b[batch_][agent] or
                                  self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent]) and
                                 (not self.next_executing_resume_fplan_b[batch_][agent] or
                                  self.next_executing_resume_fplan_b[batch_][agent])))))

                    case_2 = (((self.next_timestamp_b[batch_] - self.durations_of_actions_b[batch_][agent][1] <
                                self.durations_of_actions_b[batch_][agent][0]) and
                               self.durations_of_actions_b[batch_][agent][0] != np.inf and
                               self.durations_of_actions_b[batch_][agent][0] != 0) or
                              (self.durations_of_actions_b[batch_][agent][0] == np.inf and
                               (self.next_executing_FL_change_b[batch_][agent] or
                                self.next_executing_direct_to_b[batch_][agent])) or
                              (self.durations_of_actions_b[batch_][agent][0] == 0 and
                               ((self.np_mask_res_fplan_after_maneuv_b[batch_][agent] or
                                 self.np_mask_climb_descend_res_fplan_with_confl_loss_b[batch_][agent])
                                and self.next_executing_resume_fplan_b[batch_][agent] and
                                (not (agent in self.next_fls_with_loss_of_separation_m[batch_] or
                                      agent in self.next_fls_with_conflicts_m[batch_]) or
                                 ((agent in self.next_fls_with_loss_of_separation_m[batch_] or
                                   agent in self.next_fls_with_conflicts_m[batch_]) and
                                  (self.next_flight_phases_b[batch_][agent] == 'climbing' or
                                   self.next_flight_phases_b[batch_][agent] == 'descending')))))) and \
                             (self.data_needed_for_delayed_update_b[batch_][agent][5][agent] and
                              not self.data_needed_for_delayed_update_b[batch_][agent][6] and
                              (agent in self.data_needed_for_delayed_update_b[batch_][agent][3] or
                               agent in self.data_needed_for_delayed_update_b[batch_][agent][4]) and
                              not (self.data_needed_for_delayed_update_b[batch_][agent][7] == 'climbing' or
                                   self.data_needed_for_delayed_update_b[batch_][agent][7] == 'descending'))

                    if case_1 or case_2:

                        temp_agent_state_features = None
                        temp_agent_next_available_wps = None

                        # If condition 1 is True, then the features/available_wps which should be used
                        # will be referred to the next state.
                        if case_1:
                            temp_agent_state_features = btch[batch_][2][agent].copy()
                            temp_agent_next_available_wps = self.next_available_wps_b[batch_][agent].copy()

                        # Else if condition 2 is True, then the features/available_wps which should be used
                        # will be referred to the state exactly after the end of the duration of the selected action.
                        elif case_2:
                            temp_agent_state_features = \
                                self.data_needed_for_delayed_update_b[batch_][agent][0][agent].copy()
                            temp_agent_next_available_wps = \
                                self.data_needed_for_delayed_update_b[batch_][agent][8][agent].copy()

                        else:
                            print("A case that was not considered is happening!!")
                            exit(0)

                        validity_flag = True

                        if self.num_actions_dc > np.argmax(self.target_q_values[batch_][agent]):
                            self.maxq_actions[batch_, agent] = np.max(self.target_q_values[batch_][agent])

                        elif self.num_actions_dc <= np.argmax(self.target_q_values[batch_][agent]) < \
                                self.num_actions_dc + self.num_actions_ds:

                            if (((temp_agent_state_features[3] * (self.max_h_speed - self.min_h_speed)) +
                                 self.min_h_speed) + self.actions_list[np.argmax(self.target_q_values[batch_][agent])] <
                                self.min_h_speed) or \
                                (((temp_agent_state_features[3] * (self.max_h_speed - self.min_h_speed)) +
                                  self.min_h_speed) +
                                 self.actions_list[np.argmax(self.target_q_values[batch_][agent])] > self.max_h_speed):

                                validity_flag = False

                            else:
                                self.maxq_actions[batch_, agent] = np.max(self.target_q_values[batch_][agent])

                        elif self.num_actions_dc + self.num_actions_ds <= \
                                np.argmax(self.target_q_values[batch_][agent]) < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as:

                            if (((temp_agent_state_features[4] * (self.max_alt_speed - self.min_alt_speed)) +
                                 self.min_alt_speed) + self.actions_list[np.argmax(self.target_q_values[batch_][agent])]
                                < self.min_alt_speed) or \
                                (((temp_agent_state_features[4] * (self.max_alt_speed - self.min_alt_speed)) +
                                  self.min_alt_speed) +
                                 self.actions_list[np.argmax(self.target_q_values[batch_][agent])] >
                                 self.max_alt_speed):

                                validity_flag = False

                            else:
                                self.maxq_actions[batch_, agent] = np.max(self.target_q_values[batch_][agent])

                        elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= \
                                np.argmax(self.target_q_values[batch_][agent]) < \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                            way_point_index = \
                                abs(((self.num_actions_dc + self.num_actions_ds +
                                      self.num_actions_as + self.num_dir_wp) -
                                     np.argmax(self.target_q_values[batch_][agent])) - self.num_dir_wp)

                            if temp_agent_next_available_wps[way_point_index] == 0:
                                validity_flag = False

                            else:
                                self.maxq_actions[batch_, agent] = np.max(self.target_q_values[batch_][agent])

                        elif np.argmax(self.target_q_values[batch_][agent]) == \
                                self.num_actions_dc + self.num_actions_ds + self.num_actions_as + self.num_dir_wp:

                            self.maxq_actions[batch_, agent] = np.max(self.target_q_values[batch_][agent])

                        else:
                            print("None of the possible actions was chosen!")
                            exit(0)

                        if validity_flag is False:
                            count_tested_action = 1
                            while validity_flag is False:
                                if count_tested_action > len(self.actions_list):
                                    print("All possible actions were tried for agent {}, but all are invalid!!!".
                                          format(agent))
                                    exit(0)
                                if count_tested_action == 1:

                                    # Use -1 to inverse sorting (that is descending)
                                    sorted_actions = np.argsort(-1 * self.target_q_values[batch_][agent])

                                validity_flag = True

                                if self.num_actions_dc > sorted_actions[count_tested_action]:
                                    self.maxq_actions[batch_, agent] = \
                                        self.target_q_values[batch_][agent][sorted_actions[count_tested_action]]

                                elif self.num_actions_dc <= sorted_actions[count_tested_action] < \
                                        self.num_actions_dc + self.num_actions_ds:

                                    if (((temp_agent_state_features[3] * (self.max_h_speed - self.min_h_speed)) +
                                         self.min_h_speed) + self.actions_list[sorted_actions[count_tested_action]] <
                                        self.min_h_speed) or \
                                        (((temp_agent_state_features[3] * (self.max_h_speed - self.min_h_speed)) +
                                          self.min_h_speed) + self.actions_list[sorted_actions[count_tested_action]] >
                                         self.max_h_speed):

                                        validity_flag = False

                                    else:
                                        self.maxq_actions[batch_, agent] = \
                                            self.target_q_values[batch_][agent][sorted_actions[count_tested_action]]

                                elif self.num_actions_dc + self.num_actions_ds <= sorted_actions[count_tested_action] < \
                                        self.num_actions_dc + self.num_actions_ds + self.num_actions_as:

                                    if (((temp_agent_state_features[4] * (self.max_alt_speed - self.min_alt_speed)) +
                                         self.min_alt_speed) + self.actions_list[sorted_actions[count_tested_action]] <
                                        self.min_alt_speed) or \
                                        (((temp_agent_state_features[4] * (self.max_alt_speed - self.min_alt_speed)) +
                                          self.min_alt_speed) + self.actions_list[sorted_actions[count_tested_action]] >
                                         self.max_alt_speed):

                                        validity_flag = False

                                    else:
                                        self.maxq_actions[batch_, agent] = \
                                            self.target_q_values[batch_][agent][sorted_actions[count_tested_action]]

                                elif self.num_actions_dc + self.num_actions_ds + self.num_actions_as <= \
                                        sorted_actions[count_tested_action] < \
                                        self.num_actions_dc + self.num_actions_ds + \
                                        self.num_actions_as + self.num_dir_wp:

                                    way_point_index = \
                                        abs(((self.num_actions_dc + self.num_actions_ds +
                                              self.num_actions_as + self.num_dir_wp) -
                                             sorted_actions[count_tested_action]) - self.num_dir_wp)

                                    if temp_agent_next_available_wps[way_point_index] == 0:
                                        validity_flag = False
                                    else:
                                        self.maxq_actions[batch_, agent] = \
                                            self.target_q_values[batch_][agent][sorted_actions[count_tested_action]]

                                elif sorted_actions[count_tested_action] == \
                                        self.num_actions_dc + self.num_actions_ds + \
                                        self.num_actions_as + self.num_dir_wp:

                                    self.maxq_actions[batch_, agent] = \
                                        self.target_q_values[batch_][agent][sorted_actions[count_tested_action]]

                                else:
                                    print("None of the possible actions was chosen!")
                                    exit(0)

                                count_tested_action += 1

                    else:
                        print("There is at least one case that has not been taken into account for max_next_q!!")
                        exit(0)

    def store_episode_stats_for_current_env(self):

        if self.done:
            self.total_episode_score.append(sum(self.score) / np.count_nonzero(self.score)
                                            if np.count_nonzero(self.score) > 0
                                            else 0)
            self.total_episode_num_ATC_instr.append(self.num_ATC_instr)
            self.total_episode_alerts.append(self.env.total_alerts)
            self.total_episode_losses_of_separation.append(self.env.total_losses_of_separation)
            self.steps_per_scenario.append(self.steps)
            self.total_additional_nautical_miles.append(self.add_nautical_miles)
            self.total_confls_with_positive_tcpa.append(self.confls_with_positive_tcpa)
            self.total_additional_duration.append(self.add_duration)

    def update_arrays_and_lists_from_current_state_to_next(self):

        self.previous_durations_of_actions = \
            [tuple(list(self.durations_of_actions[i_agent]).copy()) for i_agent in range(self.n_agent)]
        self.previous_duration_dir_to_wp_ch_FL_res_fplan = self.duration_dir_to_wp_ch_FL_res_fplan.copy()
        self.previous_np_mask_res_fplan_after_maneuv = self.np_mask_res_fplan_after_maneuv.copy()
        self.previous_np_mask_climb_descend_res_fplan_with_confl_loss = \
            self.np_mask_climb_descend_res_fplan_with_confl_loss.copy()
        self.available_wps = self.next_available_wps.copy()
        self.flight_phases = self.next_flight_phases.copy()
        self.finished_FL_change = self.next_finished_FL_change.copy()
        self.finished_direct_to = self.next_finished_direct_to.copy()
        self.finished_resume_to_fplan = self.next_finished_resume_to_fplan.copy()
        self.executing_FL_change = self.next_executing_FL_change.copy()
        self.executing_direct_to = self.next_executing_direct_to.copy()
        self.executing_resume_to_fplan = self.next_executing_resume_to_fplan.copy()
        self.previous_end_of_maneuver = self.end_of_maneuver.copy()
        self.previous_flight_array_ = self.flight_array_.copy()
        self.flight_array_ = self.next_flight_array_.copy()
        self.flight_array = self.next_flight_array.copy()
        self.edges = self.next_edges.copy()
        self.adjacency_one_hot = self.next_adjacency_one_hot.copy()
        self.norm_edges_feats = self.next_norm_edges_feats.copy()
        self.adjacency_matrix = self.next_adjacency_matrix.copy()
        self.previous_fls_with_loss_of_separation = self.fls_with_loss_of_separation.copy()
        self.fls_with_loss_of_separation = self.next_fls_with_loss_of_separation.copy()
        self.previous_fls_with_conflicts = self.fls_with_conflicts.copy()
        self.fls_with_conflicts = self.next_fls_with_conflicts.copy()
        self.fls_with_alerts = self.next_fls_with_alerts.copy()
        self.count_fls_with_loss_of_separation = self.count_next_fls_with_loss_of_separation.copy()
        self.count_fls_with_conflicts = self.count_next_fls_with_conflicts.copy()
        self.count_fls_with_alerts = self.count_next_fls_with_alerts.copy()
        self.count_total_conflicts_not_alerts_per_flt = self.count_next_total_conflicts_not_alerts_per_flt.copy()
        self.count_total_alerts_per_flt = self.count_next_total_alerts_per_flt.copy()
        self.active_flights_mask = self.next_active_flights_mask.copy()
        self.timestamp = self.next_timestamp
        self.loss_ids = self.next_loss_ids.copy()
        self.alrts_ids = self.next_alrts_ids.copy()
        self.confls_ids_with_positive_tcpa = self.next_confls_ids_with_positive_tcpa.copy()
        self.confls_ids_with_negative_tcpa = self.next_confls_ids_with_negative_tcpa.copy()
        self.MOC = self.next_MOC.copy()
        self.RoC = self.next_RoC.copy()

    def write_log_files_of_episode_for_current_env(self):

        ############In evaluation mode write the appropriate files#########
        if self.evaluation and self.done:
            # Write dataframes to csv files
            if not os.path.exists('./logs'): os.mkdir('./logs')
            self.flights_positions.to_csv("./logs/flights_position_episode_{}.csv".format(self.i_episode), index=False)
            self.losses_file.to_csv("./logs/losses_episode_{}.csv".format(self.i_episode), index=False)
            self.conflicts_file.to_csv("./logs/conflicts_episode_{}.csv".format(self.i_episode), index=False)

            if not os.path.exists('./heatmaps_by_agent'): os.mkdir('./heatmaps_by_agent')
            for iii in range(len(self.htmp_by_agent)):
                self.htmp_by_agent[iii].to_csv('./heatmaps_by_agent/htmp_agent_{}_flight_{}_episode_{}.csv'.
                                               format(iii, self.flights_idxs_to_ids[iii], self.i_episode), index=False)

            # Print and store info about solved conflicts, total conflicts, different flight pairs in conflict,
            # total losses, different flight pairs in loss and losses without conflict/loss before.
            self.total_conflicts_solved()

            solved_confl_and_further_info_to_be_printed = \
                'Conflicts solved: {} \n'.format(self.solved_conflicts) + \
                'Total conflicts: {} \n'.format(self.total_conflicts) + \
                'Different conflicting flight pairs: {} \n'.format(self.number_conflicting_flight_pairs) + \
                'Total losses: {} \n'.format(self.total_loss) + \
                'Different flight pairs in loss: {} \n'.format(self.number_loss_flight_pairs) + \
                'Losses without loss/conflict before: {} \n'.\
                    format(self.number_loss_flight_pairs_without_conflict_or_loss_before) + \
                'Conflicts in groups solved: {} \n'.format(self.solved_conflicts_in_groups) + \
                'Total conflict resolution duration: {} \n'.format(self.conflict_resolution_duration_in_groups)

            print(solved_confl_and_further_info_to_be_printed)
            f_solved_confl_and_further_info = open('./logs/solved_confl_and_further_info.txt', 'w')
            f_solved_confl_and_further_info.write(solved_confl_and_further_info_to_be_printed)

    def total_conflicts_solved(self):

        self.total_conflicts = len(self.conflicts_file.index)
        conflicting_flight_pairs = []
        self.total_loss = len(self.losses_file.index)
        loss_flight_pairs = []
        self.solved_conflicts = 0
        self.solved_conflicts_in_groups = 0
        self.conflict_resolution_duration_in_groups = 0
        flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration = []

        ####Compute solved_conflicts, solved_conflicts_in_groups, conflicting_flight_pairs,####
        ####loss_flight_pairs (the last is not computed only here)####
        for dataframe_row in range(self.total_conflicts):
            conflicting_flights = tuple((self.conflicts_file.loc[dataframe_row]['conflictID'][1],
                                         self.conflicts_file.loc[dataframe_row]['conflictID'][2]))
            timestamp = self.conflicts_file.loc[dataframe_row]['conflictID'][0]

            # Store current conflicting_flights if it has not been already stored,
            # in order to compute the number of different conflicting flight pairs
            if conflicting_flights not in conflicting_flight_pairs:
                conflicting_flight_pairs.append(conflicting_flights)

            ###Compute solved_conflicts###
            # Check if there is any other conflict for the same pair of flights.
            conflict_check = True
            if dataframe_row < self.total_conflicts - 1:
                for next_dataframe_row in range(dataframe_row + 1, self.total_conflicts):
                    if self.conflicts_file.loc[next_dataframe_row]['conflictID'][1] == conflicting_flights[0] and \
                            self.conflicts_file.loc[next_dataframe_row]['conflictID'][2] == conflicting_flights[1]:

                        conflict_check = False
                        break

            # Check if there is any loss with higher timestamp for the same pair of flights.
            # Also compute loss_flight_pairs.
            loss_check = True
            if conflict_check:
                for loss_dataframe_row in range(self.total_loss):
                    flight_pair_in_loss = \
                        tuple((self.losses_file.loc[loss_dataframe_row]['conflictID'][1],
                               self.losses_file.loc[loss_dataframe_row]['conflictID'][2]))
                    if flight_pair_in_loss not in loss_flight_pairs:
                        loss_flight_pairs.append(flight_pair_in_loss)
                    if self.losses_file.loc[loss_dataframe_row]['conflictID'][1] == conflicting_flights[0] and \
                            self.losses_file.loc[loss_dataframe_row]['conflictID'][2] == conflicting_flights[1] and \
                            self.losses_file.loc[loss_dataframe_row]['conflictID'][0] > timestamp:

                        loss_check = False
                        break

            if conflict_check and loss_check:
                self.solved_conflicts += 1
            ###End of computing solved_conflicts###

            ###Compute solved_conflicts_in_groups###
            # Check if there is any other conflict for each one of the current flights (with another flight)
            # after the current timestamp.
            # Note that if there is any such flight, its action should not be 'CA'/'RFP',
            # otherwise there will be no impact of the intermediate conflict in the current conflict resolution.
            conflict_in_groups_check = True
            if dataframe_row < self.total_conflicts - 1:
                for next_dataframe_row in range(dataframe_row + 1, self.total_conflicts):

                    conflicting_flight_1_action = self.confl_resol_act[
                        (self.confl_resol_act['ConflictID'] ==
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][0])) + '_' +
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][1])) + '_' +
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][2]))) &
                        (self.confl_resol_act['RTKey'] ==
                         str(self.conflicts_file.loc[next_dataframe_row]['conflictID'][1])) &
                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                                        ]

                    conflicting_flight_2_action = self.confl_resol_act[
                        (self.confl_resol_act['ConflictID'] ==
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][0])) + '_' +
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][1])) + '_' +
                         str(int(self.conflicts_file.loc[next_dataframe_row]['conflictID'][2]))) &
                        (self.confl_resol_act['RTKey'] ==
                         str(self.conflicts_file.loc[next_dataframe_row]['conflictID'][2])) &
                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                                        ]

                    if not (self.conflicts_file.loc[next_dataframe_row]['conflictID'][1] == conflicting_flights[0] and
                            self.conflicts_file.loc[next_dataframe_row]['conflictID'][2] ==
                            conflicting_flights[1]) and \
                            (((self.conflicts_file.loc[next_dataframe_row]['conflictID'][1] ==
                               conflicting_flights[0] or
                               self.conflicts_file.loc[next_dataframe_row]['conflictID'][1] ==
                               conflicting_flights[1]) and
                              not conflicting_flight_1_action.iloc[0]['ResolutionActionType'] == 'CA' and
                              not conflicting_flight_1_action.iloc[0]['ResolutionActionType'] == 'RFP') or
                             ((self.conflicts_file.loc[next_dataframe_row]['conflictID'][2] == conflicting_flights[0] or
                               self.conflicts_file.loc[next_dataframe_row]['conflictID'][2] ==
                               conflicting_flights[1]) and
                              not conflicting_flight_2_action.iloc[0]['ResolutionActionType'] == 'CA' and
                              not conflicting_flight_2_action.iloc[0]['ResolutionActionType'] == 'RFP')) and \
                            self.conflicts_file.loc[next_dataframe_row]['conflictID'][0] > timestamp:

                        conflict_in_groups_check = False
                        conflict_in_group_timestamp = self.conflicts_file.loc[next_dataframe_row]['conflictID'][0]
                        break

            # Check if there is any loss with higher timestamp for the same pair of flights.
            # In this check, we do not care if the conflict_check is True because we check the conflicts in groups.
            loss_check_new = True
            for loss_dataframe_row in range(self.total_loss):
                if self.losses_file.loc[loss_dataframe_row]['conflictID'][1] == conflicting_flights[0] and \
                        self.losses_file.loc[loss_dataframe_row]['conflictID'][2] == conflicting_flights[1] and \
                        self.losses_file.loc[loss_dataframe_row]['conflictID'][0] > timestamp:

                    loss_check_new = False
                    loss_timestamp = self.losses_file.loc[loss_dataframe_row]['conflictID'][0]
                    break

            # Check if there is any loss for each one of the current flights (with another flight)
            # after the current timestamp.
            # Note that if there is any such flight, its action should not be 'CA'/'RFP',
            # otherwise there will be no impact of the intermediate conflict in the current conflict resolution.
            loss_check_for_conflicts_in_groups = True
            for loss_dataframe_row in range(self.total_loss):

                conflicting_flight_1_action = self.confl_resol_act[
                    (self.confl_resol_act['ConflictID'] ==
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][0])) + '_' +
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][1])) + '_' +
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][2]))) &
                    (self.confl_resol_act['RTKey'] == str(self.losses_file.loc[loss_dataframe_row]['conflictID'][1])) &
                    (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                                    ]

                conflicting_flight_2_action = self.confl_resol_act[
                    (self.confl_resol_act['ConflictID'] ==
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][0])) + '_' +
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][1])) + '_' +
                     str(int(self.losses_file.loc[loss_dataframe_row]['conflictID'][2]))) &
                    (self.confl_resol_act['RTKey'] == str(self.losses_file.loc[loss_dataframe_row]['conflictID'][2])) &
                    (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                                    ]
                if not (self.losses_file.loc[loss_dataframe_row]['conflictID'][1] == conflicting_flights[0] and
                        self.losses_file.loc[loss_dataframe_row]['conflictID'][2] == conflicting_flights[1]) and \
                        (((self.losses_file.loc[loss_dataframe_row]['conflictID'][1] == conflicting_flights[0] or
                           self.losses_file.loc[loss_dataframe_row]['conflictID'][1] == conflicting_flights[1]) and
                          not conflicting_flight_1_action.iloc[0]['ResolutionActionType'] == 'CA' and
                          not conflicting_flight_1_action.iloc[0]['ResolutionActionType'] == 'RFP') or
                         ((self.losses_file.loc[loss_dataframe_row]['conflictID'][2] == conflicting_flights[0] or
                           self.losses_file.loc[loss_dataframe_row]['conflictID'][2] == conflicting_flights[1]) and
                          not conflicting_flight_2_action.iloc[0]['ResolutionActionType'] == 'CA' and
                          not conflicting_flight_2_action.iloc[0]['ResolutionActionType'] == 'RFP')) \
                        and self.losses_file.loc[loss_dataframe_row]['conflictID'][0] > timestamp:

                    loss_check_for_conflicts_in_groups = False
                    loss_for_conflicts_in_groups_timestamp = self.losses_file.loc[loss_dataframe_row]['conflictID'][0]
                    break

            solved_conflicts_in_groups_check = False
            if loss_check_new or \
                    (not loss_check_new and loss_check_for_conflicts_in_groups and not conflict_in_groups_check and
                     conflict_in_group_timestamp < loss_timestamp) or \
                    (not loss_check_new and conflict_in_groups_check and not loss_check_for_conflicts_in_groups and
                     loss_for_conflicts_in_groups_timestamp < loss_timestamp) or \
                    (not loss_check_new and not loss_check_for_conflicts_in_groups and not conflict_in_groups_check and
                     loss_for_conflicts_in_groups_timestamp < loss_timestamp and
                     conflict_in_group_timestamp < loss_timestamp):

                self.solved_conflicts_in_groups += 1
                solved_conflicts_in_groups_check = True

            ###End of computing solved_conflicts_in_groups###

            ###Compute conflict_resolution_duration_in_groups###
            if tuple((timestamp, conflicting_flights[0], conflicting_flights[1])) not in \
                    flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration:

                flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration. \
                    append(tuple((timestamp, conflicting_flights[0], conflicting_flights[1])))

                if solved_conflicts_in_groups_check:

                    list_flight_action_duration = []
                    list_conflicts_between_current_timestep_target_timestep = []

                    for conflicting_flight_i in range(2):

                        flight_htmp = self.htmp_by_agent[
                            self.env.flight_index[int(conflicting_flights[conflicting_flight_i])]['idx']]
                        flight_last_active_timestep = \
                            flight_htmp[(flight_htmp['flight'] == float(conflicting_flights[conflicting_flight_i])) &
                                        (flight_htmp['step'] >= float(timestamp))]['step'].iloc[-1]

                        other_conflicting_flight_htmp = \
                            self.htmp_by_agent[self.env.flight_index[
                                int(conflicting_flights[1 if conflicting_flight_i == 0 else 0])]['idx']
                                                                    ]
                        other_conflicting_flight_last_active_timestep = \
                            other_conflicting_flight_htmp[
                                (other_conflicting_flight_htmp['flight'] ==
                                 float(conflicting_flights[1 if conflicting_flight_i == 0 else 0])) &
                                (other_conflicting_flight_htmp['step'] >= float(timestamp))
                                                        ]['step'].iloc[-1]

                        # If there is a single conflict for the current pair of flights
                        if conflict_check:

                            flight_action = \
                                self.confl_resol_act[
                                    (self.confl_resol_act['ConflictID'] == str(int(timestamp)) + '_' +
                                     str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                    (self.confl_resol_act['RTKey'] == str(conflicting_flights[conflicting_flight_i])) &
                                    (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                    ].iloc[0]

                            flight_action_duration = flight_action['Duration']
                            flight_action_type = flight_action['ResolutionActionType']

                            # If the selected action at the current timestep is not 'A1'/'A3/'CA'/RFP and
                            # the end of this action (based on the specified duration)
                            # the flight is inactive, assign as duration the difference between the last timestep
                            # that the flight is active and the current timestep.
                            # The same is true if the other conflicting flight is inactive before the end of
                            # the action being executed by the current flight.
                            if int(flight_action_duration) != 0 and \
                                    flight_action_type != 'CA' and \
                                    (flight_last_active_timestep < timestamp + int(flight_action_duration) or
                                     other_conflicting_flight_last_active_timestep <
                                     timestamp + int(flight_action_duration)):

                                flight_action_duration = \
                                    (flight_last_active_timestep
                                     if flight_last_active_timestep < other_conflicting_flight_last_active_timestep
                                     else
                                     other_conflicting_flight_last_active_timestep) - timestamp

                            # If the action is 'RFP', assign zero as the duration of the action.
                            elif flight_action_type == 'RFP':
                                flight_action_duration = 0.0

                            # If the action is 'CA', assign zero as the duration of the action.
                            elif flight_action_type == 'CA':
                                flight_action_duration = 0.0

                            # For action type 'A1'/'A3', if it starts at the current timestep
                            # (based on the condition: int(flight_action_duration) == 0)
                            # find its duration based on the first timestep (from current timestep onwards)
                            # that an action 'CA' is not executed
                            # (or the last timestep that an action 'CA' is executed in case that
                            # the current flight (or the other conflicting flight) becomes inactive before the end of
                            # the action being executed).
                            # Note that flight_action['ActionInProgress'].split('_')[-1] is the selected duration of
                            # the action being in progress.
                            elif int(flight_action_duration) == 0 and flight_action_type != 'RFP':

                                flight_timestep_finished_action = \
                                    flight_htmp[
                                        (flight_htmp['flight'] == float(conflicting_flights[conflicting_flight_i])) &
                                        (flight_htmp['step'] > float(timestamp)) &
                                        (flight_htmp['action_type'] != 'continue_action')
                                                ]['step']

                                if len(flight_timestep_finished_action) > 0:
                                    flight_timestep_finished_action = flight_timestep_finished_action.iloc[0]

                                # This is the case at which the current flight becomes inactive before
                                # the end of the action being executed
                                else:
                                    flight_timestep_finished_action = \
                                        flight_htmp[
                                            (flight_htmp['flight'] ==
                                             float(conflicting_flights[conflicting_flight_i])) &
                                            (flight_htmp['step'] >= float(timestamp)) &
                                            (flight_htmp['action_type'] == 'continue_action')
                                                    ]['step']

                                    if len(flight_timestep_finished_action) > 0:
                                        flight_timestep_finished_action = flight_timestep_finished_action.iloc[-1]

                                        assert int(flight_timestep_finished_action) == int(flight_last_active_timestep), \
                                            "There are cases which have not been taken into account in function 'total_conflicts_solved' !!! Case 1."
                                    else:
                                        flight_timestep_finished_action = timestamp

                                flight_action_duration = flight_timestep_finished_action - timestamp

                                # This is the case at which the other conflicting flight becomes inactive before
                                # the end of the action being executed by the current flight.
                                # In case that both the current flight and the other conflicting flight become inactive,
                                # we keep the timestep with the lower value.
                                if other_conflicting_flight_last_active_timestep < flight_timestep_finished_action:
                                    flight_action_duration = other_conflicting_flight_last_active_timestep - timestamp

                            # As there could be cases that do not satisfy any of the above if.. elif.. conditions,
                            # we should check if the calculated duration of the current flight's action ends after
                            # the other conflicting flight becomes inactive. In this case, we should assign as duration
                            # the time interval [current timestep, other_conflicting_flight_last_active_timestep]
                            if timestamp + flight_action_duration > other_conflicting_flight_last_active_timestep:
                                flight_action_duration = other_conflicting_flight_last_active_timestep - timestamp

                            list_flight_action_duration.append(flight_action_duration)

                        else:

                            target_timestep = None
                            flight_total_action_duration = None

                            # If there is not any intermediate loss or conflict for any of the two flights
                            # (of the current group of conflicts)
                            # with another flight, find the last action for both of them and calculate which lasts more.
                            if (conflict_in_groups_check and loss_check_for_conflicts_in_groups) or loss_check_new:

                                # When there is an intermediate loss/conflict but not a later conflict
                                # (based on 'loss_check_new'),
                                # we take into consideration the duration of the last action in the calculation of the
                                # duration of the conflict resolution,
                                # because this action indeed contributes to the conflict resolution
                                # (as there is no loss in the end) of the current packet of conflicts.
                                # On the other hand, when 'loss_check_new' is False
                                # (i.e., there is at least one loss in the end),
                                # we do not care about the duration of the last action but only
                                # for the 'target_timestamp', as this action probably does not truly contribute
                                # to the resolution action (as there is at least one loss in the end).
                                timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = None
                                flag_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = False

                                if not conflict_in_groups_check or not loss_check_for_conflicts_in_groups:

                                    flag_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = True

                                    if not conflict_in_groups_check and loss_check_for_conflicts_in_groups:
                                        timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = \
                                            conflict_in_group_timestamp
                                    elif conflict_in_groups_check and not loss_check_for_conflicts_in_groups:
                                        timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = \
                                            loss_for_conflicts_in_groups_timestamp
                                    elif not conflict_in_groups_check and not loss_check_for_conflicts_in_groups:
                                        timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true = \
                                            conflict_in_group_timestamp \
                                            if conflict_in_group_timestamp < loss_for_conflicts_in_groups_timestamp \
                                            else loss_for_conflicts_in_groups_timestamp

                                flight_highest_timestep_with_at_most_one_valid_action = \
                                    (self.confl_resol_act[
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: x[1] + '_' + x[2]) ==
                                         str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: int(x[0])) >= int(timestamp)) &
                                        (self.confl_resol_act['RTKey'] ==
                                         str(conflicting_flights[conflicting_flight_i])) &
                                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                        ].iloc[-1]['ConflictID'].split('_')[0]) if \
                                        not flag_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true \
                                                                                                else \
                                        (self.confl_resol_act[
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: x[1] + '_' + x[2]) ==
                                             str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: (int(x[0]) >= int(timestamp)) &
                                                             (int(x[0]) <
                                                              int(timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true)))) &
                                            (self.confl_resol_act['RTKey'] ==
                                             str(conflicting_flights[conflicting_flight_i])) &
                                            (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                            ].iloc[-1]['ConflictID'].split('_')[0])

                                flight_last_action = \
                                    self.confl_resol_act[
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: x[1] + '_' + x[2]) == str(conflicting_flights[0]) + '_' +
                                         str(conflicting_flights[1])) &
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: int(x[0])) ==
                                         int(flight_highest_timestep_with_at_most_one_valid_action)) &
                                        (self.confl_resol_act['RTKey'] ==
                                         str(conflicting_flights[conflicting_flight_i])) &
                                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null')
                                                        ].iloc[0]

                                flight_last_action_timestep = int(flight_last_action['ConflictID'].split('_')[0])
                                flight_last_action_duration = flight_last_action['Duration']
                                flight_last_action_type = flight_last_action['ResolutionActionType']

                                # IF the selected last action is not 'A1'/'A3'/'CA'/'RFP' AND:
                                #   - at the end of this action (based on the specified duration)
                                #     the flight is still active, assign as duration the difference between
                                #     the timestep at which this action ends and the current timestep,
                                #   OR
                                #   - at the end of this action the flight is inactive, assign as duration
                                #     the difference between the last timestep the flight is active and
                                #     the current timestep. We also check if the other conflicting flight
                                #     becomes inactive before the end of the last action executed by the current flight.
                                #     If this happens for both of them, we keep the timestep with the lower value.
                                if int(flight_last_action_duration) != 0 and flight_last_action_type != 'CA':

                                    flight_total_action_duration = \
                                        (int(flight_last_action_duration) + int(flight_last_action_timestep)) - \
                                        int(timestamp)

                                    if int(timestamp) + flight_total_action_duration > int(flight_last_active_timestep):
                                        flight_total_action_duration = int(flight_last_active_timestep) - int(timestamp)

                                    if int(timestamp) + flight_total_action_duration > \
                                            int(other_conflicting_flight_last_active_timestep):

                                        flight_total_action_duration = \
                                            int(other_conflicting_flight_last_active_timestep) - int(timestamp)

                                # For action type 'A1'/'A3', if it starts at the timestep of the last action
                                # (based on the condition: int(flight_last_action_duration) == 0)
                                # find its duration based on the first timestep
                                # (from 'flight_last_action_timestep' onwards)
                                # that an action 'CA' is not executed
                                # (or the last timestep that an action 'CA' is executed in case that
                                # the current flight/the other conflicting flight becomes inactive before
                                # the end of the action being executed).
                                # Note that flight_last_action['ActionInProgress'].split('_')[-1] is the
                                # selected duration of the action being in progress.
                                elif int(flight_last_action_duration) == 0 and flight_last_action_type != 'RFP' and \
                                        flight_last_action_type != 'CA':

                                    flight_timestep_finished_last_action = \
                                        flight_htmp[(flight_htmp['flight'] ==
                                                     float(conflicting_flights[conflicting_flight_i])) &
                                                    (flight_htmp['step'] > float(flight_last_action_timestep)) &
                                                    (flight_htmp['action_type'] != 'continue_action')]['step']

                                    if len(flight_timestep_finished_last_action) > 0:
                                        flight_timestep_finished_last_action = \
                                            flight_timestep_finished_last_action.iloc[0]

                                    # This is the case at which the current flight becomes inactive
                                    # before the end of the action being executed.
                                    else:
                                        flight_timestep_finished_last_action = \
                                            flight_htmp[
                                                (flight_htmp['flight'] ==
                                                 float(conflicting_flights[conflicting_flight_i])) &
                                                (flight_htmp['step'] >= float(flight_last_action_timestep)) &
                                                (flight_htmp['action_type'] == 'continue_action')
                                                        ]['step']

                                        if len(flight_timestep_finished_last_action) > 0:
                                            flight_timestep_finished_last_action = \
                                                flight_timestep_finished_last_action.iloc[-1]
                                        else:
                                            flight_timestep_finished_last_action = float(flight_last_action_timestep)

                                        assert int(flight_timestep_finished_last_action) == \
                                               int(flight_last_active_timestep), \
                                               "There are cases which have not been taken into account in " \
                                                "function 'total_conflicts_solved'!!! Case 2."

                                    flight_total_action_duration = flight_timestep_finished_last_action - timestamp

                                    # This is the case at which the other conflicting flight becomes inactive
                                    # before the end of the last action being executed by the current flight.
                                    # In case that both the current flight and the other conflicting flight
                                    # become inactive, we keep the timestep with the lower value.
                                    if other_conflicting_flight_last_active_timestep < \
                                            flight_timestep_finished_last_action:

                                        flight_total_action_duration = \
                                            other_conflicting_flight_last_active_timestep - timestamp

                                elif flight_last_action_type == 'CA':
                                    flight_all_valid_actions_between_timestamp_flight_last_action_timestep = \
                                        self.confl_resol_act[
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: x[1] + '_' + x[2]) ==
                                             str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: (int(x[0]) >= int(timestamp)) &
                                                             (int(x[0]) < flight_last_action_timestep))) &
                                            (self.confl_resol_act['RTKey'] ==
                                             str(conflicting_flights[conflicting_flight_i])) &
                                            (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                            (self.confl_resol_act['ResolutionActionType'] != 'CA') &
                                            (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                            ]

                                    # If there is no other action except from 'CA'/'RFP'
                                    # at the time interval [timestamp, flight_last_action_timestep],
                                    # then we should assign zero action duration for the current flight as
                                    # it does not contribute to the current conflict resolution.
                                    if len(flight_all_valid_actions_between_timestamp_flight_last_action_timestep) == 0:
                                        flight_total_action_duration = 0.0

                                    # If there is another action except from 'CA'/'RFP'
                                    # at the time interval [timestamp, flight_last_action_timestep],
                                    # then we should compute the duration of the time interval
                                    # [timestamp, flight_action_before_last_action_end_timestep]
                                    else:
                                        flight_highest_timestep_between_timestamp_flight_last_action_timestep_with_at_most_one_valid_action = \
                                            flight_all_valid_actions_between_timestamp_flight_last_action_timestep. \
                                                iloc[-1]['ConflictID'].split('_')[0]

                                        flight_action_before_last_action = \
                                            self.confl_resol_act[
                                                (self.confl_resol_act['ConflictID'].str.split('_').
                                                 apply(lambda x: x[1] + '_' + x[2]) ==
                                                 str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                                (self.confl_resol_act['ConflictID'].str.split('_').
                                                 apply(lambda x: int(x[0]) ==
                                                                 int(flight_highest_timestep_between_timestamp_flight_last_action_timestep_with_at_most_one_valid_action))) &
                                                (self.confl_resol_act['RTKey'] ==
                                                 str(conflicting_flights[conflicting_flight_i])) &
                                                (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                                (self.confl_resol_act['ResolutionActionType'] != 'CA') &
                                                (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                                ].iloc[0]

                                        flight_action_before_last_action_timestep = \
                                            int(flight_action_before_last_action['ConflictID'].split('_')[0])
                                        flight_action_before_last_action_duration = \
                                            flight_action_before_last_action['Duration']
                                        flight_action_before_last_action_type = \
                                            flight_action_before_last_action['ResolutionActionType']

                                        # If the selected 'action_before_last_action' is not 'A1'/'A3'/'CA'/'RFP',
                                        # assign as duration the difference between the timestep at which this
                                        # action ends and the current timestep.
                                        # Note that the flight cannot be inactive before the end of this action
                                        # because afterwards there is the action 'CA'.
                                        if int(flight_action_before_last_action_duration) != 0 and \
                                                flight_action_before_last_action_type != 'CA':

                                            flight_total_action_duration = \
                                                (int(flight_action_before_last_action_duration) +
                                                 int(flight_action_before_last_action_timestep)) \
                                                - int(timestamp)

                                        # For the action type 'A1'/'A3', if it starts at the timestep of the action
                                        # before the last action find its duration based on the first timestep
                                        # (from 'flight_action_before_last_action_timestep' to
                                        # 'flight_last_action_timestep') that an action 'CA' is not executed.
                                        # Also, note that
                                        # flight_action_before_last_action['ActionInProgress'].split('_')[-1]
                                        # is the selected duration of the 'action_before_last_action' being in progress.
                                        elif int(flight_action_before_last_action_duration) == 0 and \
                                                flight_action_before_last_action_type != 'RFP':

                                            flight_timestep_finished_action_before_last_action = \
                                                flight_htmp[
                                                    (flight_htmp['flight'] ==
                                                     float(conflicting_flights[conflicting_flight_i])) &
                                                    (float(flight_last_action_timestep) >= flight_htmp['step']) &
                                                    (flight_htmp['step'] >
                                                     float(flight_action_before_last_action_timestep)) &
                                                    (flight_htmp['action_type'] != 'continue_action')
                                                            ]['step']

                                            if len(flight_timestep_finished_action_before_last_action) > 0:
                                                flight_timestep_finished_action_before_last_action = \
                                                    flight_timestep_finished_action_before_last_action.iloc[0]

                                                flight_total_action_duration = \
                                                    flight_timestep_finished_action_before_last_action - timestamp

                                            # This is:
                                            # - the case at which:
                                            #       - there is not any intermediate loss/conflict before the end
                                            #         of the last action being executed
                                            #         (and not any action other than 'RFP')
                                            #         but there is an action 'RFP' with
                                            #         timestamp = 'flight_timestep_finished_last_action' >
                                            #                     'flight_last_action_timestep' (CASE 1),
                                            #
                                            #       OR
                                            #
                                            #       -  the current flight becomes inactive before the end of the
                                            #          last action being executed (CASE 2),
                                            #
                                            #   OR
                                            #
                                            #   - the case at which:
                                            #       - there is an intermediate loss/conflict and before that, the action
                                            #         ('flight_last_action') with the highest timestamp in
                                            #         'self.confl_resol_act' is 'CA' continuing an action 'A1'/'A3' and
                                            #         certainly (as there is an intermediate conflict/loss)
                                            #         there is an action other than 'CA'/'RFP' for the timestamp of
                                            #         which (that is, TS) is True that:
                                            #         'flight_last_action_timestep' <= TS < 'target_timestamp' (CASE 3),
                                            #
                                            #       or
                                            #
                                            #       - TS == 'target_timestamp' (CASE 4).
                                            #
                                            #      Note that we need to separate CASE 3 and 4,
                                            #      because there might be an action 'RFP'
                                            #      (and maybe another 'CA' after 'RFP')
                                            #      between 'CA' and 'target_timestamp',
                                            #      thus CASE 3 conditions can identify it.
                                            else:

                                                # This is CASE 1/2.
                                                if not flag_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true:

                                                    flight_timestep_finished_last_action = \
                                                        flight_htmp[
                                                            (flight_htmp['flight'] ==
                                                             float(conflicting_flights[conflicting_flight_i])) &
                                                            (flight_htmp['step'] > float(flight_last_action_timestep)) &
                                                            (flight_htmp['action_type'] == 'resume_fplan')]['step']

                                                    # CASE 1.
                                                    if len(flight_timestep_finished_last_action) > 0:
                                                        flight_timestep_finished_last_action = \
                                                            flight_timestep_finished_last_action.iloc[0]

                                                        # Check if there is an action other than 'CA'/'RFP' with
                                                        # flight_timestep_new_action_after_timestep_finished_last_action > flight_last_action_timestep,
                                                        # and if so, check if it has
                                                        # flight_timestep_new_action_after_timestep_finished_last_action >
                                                        # flight_timestep_finished_last_action (which should be True).
                                                        # Otherwise, flight_timestep_new_action_after_timestep_finished_last_action <=
                                                        # flight_last_active_timestep should be True.
                                                        flight_timestep_new_action_after_timestep_finished_last_action = \
                                                            flight_htmp[
                                                                (flight_htmp['flight'] ==
                                                                 float(conflicting_flights[conflicting_flight_i])) &
                                                                (flight_htmp['step'] >
                                                                 float(flight_last_action_timestep)) &
                                                                (flight_htmp['action_type'] != 'resume_fplan') &
                                                                (flight_htmp['action_type'] != 'continue_action')
                                                                        ]['step']

                                                        # The below condition should not be True due
                                                        # to 'loss_check_new'.
                                                        if len(flight_timestep_new_action_after_timestep_finished_last_action) > 0:
                                                            print("There are cases which have not been taken"
                                                                  " into account in function "
                                                                  "'total_conflicts_solved' !!! Case 3.")
                                                            exit(0)
                                                        else:
                                                            assert flight_timestep_finished_last_action <= \
                                                                   flight_last_active_timestep, \
                                                                    "There are cases which have not been taken " \
                                                                    "into account in function " \
                                                                    "'total_conflicts_solved' !!! Case 4."

                                                        flight_total_action_duration = \
                                                            flight_timestep_finished_last_action - timestamp

                                                    # CASE 2.
                                                    else:
                                                        flight_timestep_finished_last_action_prematurely = \
                                                            flight_htmp[
                                                                (flight_htmp['flight'] ==
                                                                 float(conflicting_flights[conflicting_flight_i])) &
                                                                (flight_htmp['step'] >=
                                                                 float(flight_last_action_timestep)) &
                                                                (flight_htmp['action_type'] == 'continue_action')
                                                                        ]['step'].iloc[-1]

                                                        assert int(flight_timestep_finished_last_action_prematurely) == \
                                                               int(flight_last_active_timestep), \
                                                                "There are cases which have not been taken into " \
                                                                "account in function 'total_conflicts_solved' !!! " \
                                                                "Case 5."

                                                        flight_total_action_duration = \
                                                            flight_timestep_finished_last_action_prematurely - timestamp

                                                else:  # This is the CASE 3/4

                                                    flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated = \
                                                        flight_htmp[
                                                            (flight_htmp['flight'] ==
                                                             float(conflicting_flights[conflicting_flight_i])) &
                                                            (flight_htmp['step'] >=
                                                             float(flight_last_action_timestep)) &
                                                            (float(
                                                                timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true
                                                                    ) > flight_htmp['step']) &
                                                            (flight_htmp['action_type'] != 'continue_action')
                                                            ]['step']

                                                    # CASE 4.
                                                    if len(flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated) == 0:
                                                        flight_total_action_duration = \
                                                            timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true - timestamp

                                                    # CASE 3.
                                                    else:
                                                        flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated = \
                                                            flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated. \
                                                                iloc[-1]

                                                        assert flag_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true and \
                                                               (int(timestamp) <
                                                                int(flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated) <
                                                                int(timestamp_before_target_case_loss_or_conflict_in_groups_violated_loss_check_new_true)), \
                                                                "There are cases which have not been taken into account " \
                                                                "in function 'total_conflicts_solved' !!! Case 6."

                                                        flight_total_action_duration = \
                                                            flight_timestep_finished_last_valid_action_before_loss_or_conflict_in_groups_violated - timestamp

                                        else:
                                            print('There are cases that have not been taken into account in function'
                                                  ' "total_conflicts_solved" !!! Case 7.')
                                            exit(0)

                                        # If the current flight's action before its last action ends after
                                        # the other conflicting flight becomes inactive, we should assign as duration
                                        # the time interval
                                        # [current timestep, other_conflicting_flight_last_active_timestep]
                                        if timestamp + flight_total_action_duration > other_conflicting_flight_last_active_timestep:
                                            flight_total_action_duration = \
                                                other_conflicting_flight_last_active_timestep - int(timestamp)

                                        # If the current flight's action before its last action ends after
                                        # this flight becomes inactive, we should assign as duration
                                        # the time interval [current timestep, flight_last_active_timestep]
                                        if timestamp + flight_total_action_duration > flight_last_active_timestep:
                                            flight_total_action_duration = flight_last_active_timestep - int(timestamp)

                                elif flight_last_action_type == 'RFP':
                                    flight_all_valid_actions_between_timestamp_flight_last_action_timestep = \
                                        self.confl_resol_act[
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: x[1] + '_' + x[2]) ==
                                             str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                            (self.confl_resol_act['ConflictID'].str.split('_').
                                             apply(lambda x: (int(x[0]) >= int(timestamp)) &
                                                             (int(x[0]) < flight_last_action_timestep))) &
                                            (self.confl_resol_act['RTKey'] ==
                                             str(conflicting_flights[conflicting_flight_i])) &
                                            (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                            (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                            ]

                                    # If there is no other action except from 'RFP' at the time interval
                                    # [timestamp, flight_last_action_timestep],
                                    # then we should assign zero action duration for the current flight as
                                    # it does not contribute to the current conflict resolution.
                                    if len(flight_all_valid_actions_between_timestamp_flight_last_action_timestep) == 0:
                                        flight_total_action_duration = 0.0

                                    # If there is another action except from 'RFP' at the time interval
                                    # [timestamp, flight_last_action_timestep],
                                    # then we should compute the duration of the time interval
                                    # [timestamp, flight_action_before_last_action_end_timestep]
                                    else:
                                        flight_highest_timestep_between_timestamp_flight_last_action_timestep_with_at_most_one_valid_action = \
                                            flight_all_valid_actions_between_timestamp_flight_last_action_timestep. \
                                                iloc[-1]['ConflictID'].split('_')[0]

                                        flight_action_before_last_action = \
                                            self.confl_resol_act[
                                                (self.confl_resol_act['ConflictID'].str.split('_').
                                                 apply(lambda x: x[1] + '_' + x[2]) ==
                                                 str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                                (self.confl_resol_act['ConflictID'].str.split('_').
                                                 apply(lambda x: int(x[0]) ==
                                                                 int(flight_highest_timestep_between_timestamp_flight_last_action_timestep_with_at_most_one_valid_action))) &
                                                (self.confl_resol_act['RTKey'] ==
                                                 str(conflicting_flights[conflicting_flight_i])) &
                                                (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                                (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                                ].iloc[0]

                                        flight_action_before_last_action_timestep = \
                                            int(flight_action_before_last_action['ConflictID'].split('_')[0])
                                        flight_action_before_last_action_duration = \
                                            flight_action_before_last_action['Duration']
                                        flight_action_before_last_action_type = \
                                            flight_action_before_last_action['ResolutionActionType']

                                        # If the selected 'action_before_last_action' is not 'A1'/'A3'/'CA'/'RFP',
                                        # assign as duration the difference between the timestep at which this action
                                        # ends and the current timestep.
                                        # Note that the flight cannot be inactive before the end of this action because
                                        # afterwards there is the action 'RFP'.
                                        if int(flight_action_before_last_action_duration) != 0 and \
                                                flight_action_before_last_action_type != 'CA':

                                            flight_total_action_duration = \
                                                (int(flight_action_before_last_action_duration) +
                                                 int(flight_action_before_last_action_timestep)) \
                                                - int(timestamp)

                                        # For the action type 'A1'/'A3', if it starts at the timestep of the action
                                        # before the last action
                                        # (based on the condition: int(flight_action_before_last_action_duration) == 0)
                                        # find its duration based on the first timestep
                                        # (from 'flight_action_before_last_action_timestep' to
                                        # 'flight_last_action_timestep') that an action 'CA' is not executed.
                                        # Note again that the flight cannot be innactive before the end of this action
                                        # ('action_before_last_action')
                                        # because afterwards there is the action 'RFP'.
                                        # Also, note that flight_last_action['ActionInProgress'].split('_')[-1]
                                        # is the selected duration of the 'action_before_last_action' being in progress.
                                        elif (int(flight_action_before_last_action_duration) == 0 and
                                              flight_action_before_last_action_type != 'RFP') and \
                                                flight_last_action_type != 'CA':

                                            flight_timestep_finished_action_before_last_action = \
                                                flight_htmp[
                                                    (flight_htmp['flight'] ==
                                                     float(conflicting_flights[conflicting_flight_i])) &
                                                    (float(flight_last_action_timestep) >= flight_htmp['step']) &
                                                    (flight_htmp['step'] >
                                                     float(flight_action_before_last_action_timestep)) &
                                                    (flight_htmp['action_type'] != 'continue_action')]['step'].iloc[0]

                                            flight_total_action_duration = \
                                                flight_timestep_finished_action_before_last_action - timestamp

                                        elif flight_last_action_type == 'CA':
                                            flight_all_valid_actions_between_timestamp_flight_action_before_last_action_timestep = \
                                                self.confl_resol_act[
                                                    (self.confl_resol_act['ConflictID'].str.split('_').
                                                     apply(lambda x: x[1] + '_' + x[2]) ==
                                                     str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                                    (self.confl_resol_act['ConflictID'].str.split('_').
                                                     apply(lambda x: (int(x[0]) >= int(timestamp)) &
                                                                     (int(x[0]) <
                                                                      int(flight_action_before_last_action_timestep)))) &
                                                    (self.confl_resol_act['RTKey'] ==
                                                     str(conflicting_flights[conflicting_flight_i])) &
                                                    (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                                    (self.confl_resol_act['ResolutionActionType'] != 'CA') &
                                                    (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                                    ]

                                            # If there is no other action except from 'CA'/'RFP' at the time interval
                                            # [timestamp, flight_action_before_last_action_timestep],
                                            # then we should assign zero action duration for the current flight as
                                            # it does not contribute to the current conflict resolution.
                                            if len(flight_all_valid_actions_between_timestamp_flight_action_before_last_action_timestep) == 0:
                                                flight_total_action_duration = 0.0

                                            # If there is another action except from 'CA'/'RFP' at the time interval
                                            # [timestamp, flight_action_before_last_action_timestep],
                                            # then we should compute the duration of the time interval
                                            # [timestamp, flight_action_before_action_before_last_action_end_timestep]
                                            else:
                                                flight_highest_timestep_between_timestamp_flight_action_before_last_action_timestep_with_at_most_one_valid_action = \
                                                    flight_all_valid_actions_between_timestamp_flight_action_before_last_action_timestep. \
                                                        iloc[-1]['ConflictID'].split('_')[0]

                                                flight_action_before_action_before_last_action = \
                                                    self.confl_resol_act[
                                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                                         apply(lambda x: x[1] + '_' + x[2]) ==
                                                         str(conflicting_flights[0]) + '_' +
                                                         str(conflicting_flights[1])) &
                                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                                         apply(lambda x: int(x[0]) ==
                                                                         int(
                                                                             flight_highest_timestep_between_timestamp_flight_action_before_last_action_timestep_with_at_most_one_valid_action
                                                                             ))) &
                                                        (self.confl_resol_act['RTKey'] ==
                                                         str(conflicting_flights[conflicting_flight_i])) &
                                                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                                        (self.confl_resol_act['ResolutionActionType'] != 'CA') &
                                                        (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                                        ].iloc[0]

                                                flight_action_before_action_before_last_action_timestep = \
                                                    int(flight_action_before_action_before_last_action['ConflictID'].
                                                        split('_')[0])

                                                flight_action_before_action_before_last_action_duration = \
                                                    flight_action_before_action_before_last_action['Duration']

                                                flight_action_before_action_before_last_action_type = \
                                                    flight_action_before_action_before_last_action[
                                                        'ResolutionActionType']

                                                # If the selected 'flight_action_before_action_before_last_action'
                                                # is not 'A1'/'A3'/'CA'/'RFP',
                                                # assign as duration the difference between the timestep at which
                                                # this action ends and the current timestep.
                                                # Note that the flight cannot be inactive before the end of this action
                                                # because afterwards there is the action 'CA'.
                                                if int(flight_action_before_action_before_last_action_duration) != 0 and \
                                                        flight_action_before_action_before_last_action_type != 'CA':

                                                    flight_total_action_duration = \
                                                        (int(flight_action_before_action_before_last_action_duration) +
                                                         int(flight_action_before_action_before_last_action_timestep)) \
                                                        - int(timestamp)

                                                # For the action type 'A1'/'A3', if it starts at the timestep of
                                                # 'flight_action_before_action_before_last_action'
                                                # find its duration based on the first timestep
                                                # (from 'flight_action_before_action_before_last_action_timestep' to
                                                # 'flight_action_before_last_action_timestep')
                                                # that an action 'CA' is not executed.
                                                # Note again that the flight cannot be innactive before
                                                # the end of this action
                                                # ('action_before_action_before_last_action')
                                                # because afterwards there is the action 'CA'.
                                                # Also, note that
                                                # flight_action_before_action_before_last_action['ActionInProgress'].split('_')[-1]
                                                # is the selected duration of the
                                                # 'action_before_action_before_last_action' being in progress.
                                                elif int(flight_action_before_action_before_last_action_duration) == 0 and \
                                                        flight_action_before_action_before_last_action_type != 'RFP':

                                                    flight_timestep_finished_action_before_action_before_last_action = \
                                                        flight_htmp[
                                                            (flight_htmp['flight'] ==
                                                             float(conflicting_flights[conflicting_flight_i])) &
                                                            (float(flight_action_before_last_action_timestep) >=
                                                             flight_htmp['step']) &
                                                            (flight_htmp['step'] >
                                                             float(flight_action_before_action_before_last_action_timestep)) &
                                                            (flight_htmp['action_type'] != 'continue_action')
                                                                    ]['step'].iloc[0]

                                                    flight_total_action_duration = \
                                                        flight_timestep_finished_action_before_action_before_last_action \
                                                        - timestamp

                                                else:
                                                    print('There are cases that have not been taken into account in '
                                                          'function "total_conflicts_solved" !!! Case 8.')
                                                    exit(0)

                                                # If the current flight's action before the action before its
                                                # last action ends after the other conflicting flight becomes inactive,
                                                # we should assign as duration the time interval
                                                # [current timestep, other_conflicting_flight_last_active_timestep]
                                                if timestamp + flight_total_action_duration > \
                                                        other_conflicting_flight_last_active_timestep:

                                                    flight_total_action_duration = \
                                                        other_conflicting_flight_last_active_timestep - int(timestamp)

                                                # If the current flight's action before the action before its
                                                # last action ends after this flight becomes inactive,
                                                # we should assign as duration
                                                # the time interval [current timestep, flight_last_active_timestep]
                                                if timestamp + flight_total_action_duration > \
                                                        flight_last_active_timestep:

                                                    flight_total_action_duration = \
                                                        flight_last_active_timestep - int(timestamp)

                                        else:
                                            print('There are cases that have not been taken into account '
                                                  'in function "total_conflicts_solved" !!! Case 9.')
                                            exit(0)

                                        # If the current flight's action before its last action ends after
                                        # the other conflicting flight becomes inactive, we should assign as duration
                                        # the time interval
                                        # [current timestep, other_conflicting_flight_last_active_timestep]
                                        if timestamp + flight_total_action_duration > \
                                                other_conflicting_flight_last_active_timestep:

                                            flight_total_action_duration = \
                                                other_conflicting_flight_last_active_timestep - int(timestamp)

                                else:
                                    print('There are cases that have not been taken into account '
                                          'in function "total_conflicts_solved" !!! Case 10.')
                                    exit(0)

                                target_timestep = int(timestamp) + flight_total_action_duration

                            else:

                                assert not loss_check_new, \
                                    'There are cases that have not been taken into account ' \
                                    'in function "total_conflicts_solved" !!! Case 11.'

                                # If there is an intermediate loss (and not a conflict) for any of the two flights
                                # (of the current group of conflicts)
                                # with another flight, consider the timestep (that is, the 'target' timestep)
                                # that the loss occurs as the end of the conflict resolution.
                                if conflict_in_groups_check and not loss_check_for_conflicts_in_groups:
                                    target_timestep = loss_for_conflicts_in_groups_timestamp

                                # If there is an intermediate conflict (and not a loss) for any of the two flights
                                # (of the current group of conflicts)
                                # with another flight, consider as the target timestep the timestep at which
                                # the conflict occurs.
                                elif not conflict_in_groups_check and loss_check_for_conflicts_in_groups:
                                    target_timestep = conflict_in_group_timestamp

                                # If there is an intermediate conflict and an intermediate loss for any of the
                                # two flights (of the current group of conflicts)
                                # with another flight, consider as the target timestep the lower timestep
                                # at which the conflict/loss occurs.
                                else:
                                    assert not conflict_in_groups_check and not loss_check_for_conflicts_in_groups, \
                                        'There are cases that have not been taken into account in ' \
                                        'function "total_conflicts_solved" !!! Case 12.'

                                    target_timestep = \
                                        conflict_in_group_timestamp if \
                                            conflict_in_group_timestamp < loss_for_conflicts_in_groups_timestamp \
                                                                    else \
                                        loss_for_conflicts_in_groups_timestamp

                                flight_all_valid_actions_between_timestep_target_timestep = \
                                    self.confl_resol_act[
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: x[1] + '_' + x[2]) ==
                                         str(conflicting_flights[0]) + '_' + str(conflicting_flights[1])) &
                                        (self.confl_resol_act['ConflictID'].str.split('_').
                                         apply(lambda x: (int(x[0]) >= int(timestamp)) &
                                                         (int(x[0]) < int(target_timestep)))) &
                                        (self.confl_resol_act['RTKey'] ==
                                         str(conflicting_flights[conflicting_flight_i])) &
                                        (self.confl_resol_act['FilteredOut'].astype(str) == 'null') &
                                        (self.confl_resol_act['ResolutionActionType'] != 'CA') &
                                        (self.confl_resol_act['ResolutionActionType'] != 'RFP')
                                                        ]

                                # If there are no actions other than 'RFP'/'CA' at the time interval
                                # [timestep, target_timestep),
                                # then the duration of the resolution action is zero.
                                if len(flight_all_valid_actions_between_timestep_target_timestep) == 0:
                                    flight_total_action_duration = 0.0

                                # Compute the duration of the conflict resolution as the difference
                                # between the target timestep and the current timestep.
                                else:
                                    flight_total_action_duration = target_timestep - int(timestamp)

                                    # If the current flight's action ends after
                                    # the other conflicting flight becomes inactive, we should assign as duration
                                    # the time interval
                                    # [current timestep, other_conflicting_flight_last_active_timestep]
                                    if timestamp + flight_total_action_duration > \
                                            other_conflicting_flight_last_active_timestep:

                                        flight_total_action_duration = \
                                            other_conflicting_flight_last_active_timestep - int(timestamp)

                                    # If the current flight's action ends after
                                    # this flight becomes inactive, we should assign as duration
                                    # the time interval [current timestep, flight_last_active_timestep]
                                    if timestamp + flight_total_action_duration > flight_last_active_timestep:
                                        flight_total_action_duration = flight_last_active_timestep - int(timestamp)

                            list_flight_action_duration.append(flight_total_action_duration)

                            list_conflicts_between_current_timestep_target_timestep. \
                                extend([conflict_item
                                        for conflict_item in
                                        self.conflicts_file[self.conflicts_file['conflictID'].
                                                            apply(lambda x: (int(x[0]) > int(timestamp)) &
                                                                            (int(x[0]) < int(target_timestep)) &
                                                                            (int(x[1]) == int(conflicting_flights[0])) &
                                                                            (int(x[2]) == int(conflicting_flights[1])))
                                                            ]['conflictID'].tolist()
                                        if
                                        conflict_item not in list_conflicts_between_current_timestep_target_timestep])

                    # Check if the conflicts in 'list_conflicts_between_current_timestep_target_timestep' are already in
                    # 'flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration' and if so
                    # stop the execution and print a message.
                    # Otherwise, append these conflicts to the aforementioned list
                    assert (np.array([conflict_item in
                                      flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration
                                      for conflict_item in list_conflicts_between_current_timestep_target_timestep])
                            == False).all(), \
                        'New conflicts are already in ' \
                        '"flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration" !!!'

                    flight_pairs_and_timestamp_already_accounted_for_conflict_resolution_duration. \
                        extend(list_conflicts_between_current_timestep_target_timestep)

                    self.conflict_resolution_duration_in_groups += \
                        max(list_flight_action_duration[0], list_flight_action_duration[1])

            ###End of computing conflict_resolution_duration_in_groups###

        ####Compute loss_flight_pairs_without_conflict_or_loss_before and check if loss_flight_pairs is complete####
        loss_flight_pairs_without_conflict_or_loss_before = []
        for loss_dataframe_row in range(self.total_loss):
            flight_pair_in_loss = tuple((self.losses_file.loc[loss_dataframe_row]['conflictID'][1],
                                         self.losses_file.loc[loss_dataframe_row]['conflictID'][2]))
            timestamp = self.losses_file.loc[loss_dataframe_row]['conflictID'][0]
            if flight_pair_in_loss not in loss_flight_pairs:
                loss_flight_pairs.append(flight_pair_in_loss)

            # Check if there is any conflict for the current flight_pair_in_loss before the specific loss
            conflict_before_loss_check = False
            for dataframe_row in range(self.total_conflicts):
                if self.conflicts_file.loc[dataframe_row]['conflictID'][0] >= timestamp:
                    break
                elif self.conflicts_file.loc[dataframe_row]['conflictID'][1] == flight_pair_in_loss[0] and \
                        self.conflicts_file.loc[dataframe_row]['conflictID'][2] == flight_pair_in_loss[1]:
                    conflict_before_loss_check = True
                    break

            # Check if there is any loss for the current flight_pair_in_loss before the specific loss
            loss_before_loss_check = False
            if not conflict_before_loss_check:
                for previous_loss_dataframe_row in range(loss_dataframe_row - 1, -1, -1):
                    if self.losses_file.loc[previous_loss_dataframe_row]['conflictID'][1] == flight_pair_in_loss[0] and \
                            self.losses_file.loc[previous_loss_dataframe_row]['conflictID'][2] == \
                            flight_pair_in_loss[1]:
                        loss_before_loss_check = True
                        break

            if not conflict_before_loss_check and not loss_before_loss_check:
                if flight_pair_in_loss in loss_flight_pairs_without_conflict_or_loss_before:
                    print("There is a problem in total_conflicts_solved function!! "
                          "There are two losses without any conflict/loss before for the same pair of flights!!")
                    exit(0)
                loss_flight_pairs_without_conflict_or_loss_before.append(flight_pair_in_loss)

        self.number_conflicting_flight_pairs = len(conflicting_flight_pairs)
        self.number_loss_flight_pairs = len(loss_flight_pairs)
        self.number_loss_flight_pairs_without_conflict_or_loss_before = \
            len(loss_flight_pairs_without_conflict_or_loss_before)

    def write_log_files_of_episode_for_all_env(self):

        ########Print and store useful information for all scenarios run#########
        total_sum_episode_score = sum(self.total_episode_score) / np.count_nonzero(self.total_episode_score) \
                                  if np.count_nonzero(self.total_episode_score) > 0 \
                                  else 0
        print('Episode mean reward: ' + str(total_sum_episode_score), end='\t')
        total_sum_episode_losses_of_separation = sum(self.total_episode_losses_of_separation)
        print('Episode absolute losses of separation: ' + str(total_sum_episode_losses_of_separation), end='\t')
        mean_total_losses_of_separation_per_scenario = \
            [self.total_episode_losses_of_separation[scen_] / self.steps_per_scenario[scen_]
             for scen_ in range(len(self.scenario_list))]
        mean_total_losses_of_separation_for_all_scenarios = \
            sum(mean_total_losses_of_separation_per_scenario) / \
            np.count_nonzero(mean_total_losses_of_separation_per_scenario) \
            if np.count_nonzero(mean_total_losses_of_separation_per_scenario) > 0 \
            else 0
        print('Episode mean losses of separation: ' + str(mean_total_losses_of_separation_for_all_scenarios), end='\t')
        total_sum_episode_alerts = sum(self.total_episode_alerts)
        print('Episode absolute alerts: ' + str(total_sum_episode_alerts), end='\t')
        mean_total_alerts_per_scenario = \
            [self.total_episode_alerts[scen_] / self.steps_per_scenario[scen_]
             for scen_ in range(len(self.scenario_list))]
        mean_total_alerts_for_all_scenarios = \
            sum(mean_total_alerts_per_scenario) / \
            np.count_nonzero(mean_total_alerts_per_scenario) \
            if np.count_nonzero(mean_total_alerts_per_scenario) > 0 else 0
        print('Episode mean alerts: ' + str(mean_total_alerts_for_all_scenarios), end='\t')
        total_sum_episode_num_ATC_instr = sum(self.total_episode_num_ATC_instr)
        print('Total episode number of ATC instructions: ' + str(total_sum_episode_num_ATC_instr), end='\t')
        total_sum_episode_additional_nautical_miles = sum(self.total_additional_nautical_miles)
        print('Total additional NM: ' + str(total_sum_episode_additional_nautical_miles), end='\t')
        total_sum_confls_with_positive_tcpa = sum(self.total_confls_with_positive_tcpa)
        print('Total conflicts with positive tcpa (not alerts): ' + str(total_sum_confls_with_positive_tcpa), end='\t')
        total_sum_episode_additional_duration = sum(self.total_additional_duration)
        print('Total additional duration: ' + str(total_sum_episode_additional_duration), end='\t')

        self.f.write('Episode mean reward: ' + str(total_sum_episode_score) + ' '
                     + 'Episode absolute losses of separation: ' + str(total_sum_episode_losses_of_separation) + ' '
                     + 'Episode mean losses of separation: ' + str(mean_total_losses_of_separation_for_all_scenarios) +
                     ' '
                     + 'Episode absolute alerts: ' + str(total_sum_episode_alerts) + ' '
                     + 'Episode mean alerts: ' + str(mean_total_alerts_for_all_scenarios) + ' '
                     + 'Total episode number of ATC instructions: ' + str(total_sum_episode_num_ATC_instr) + ' '
                     + 'Total additional NM: ' + str(total_sum_episode_additional_nautical_miles) + ' '
                     + 'Total conflicts with positive tcpa (not alerts): ' + str(total_sum_confls_with_positive_tcpa) +
                     ' '
                     + 'Total additional duration: ' + str(total_sum_episode_additional_duration))

        if self.i_episode < self.episode_before_train:
            self.f.write(' ' + 'Total episode loss divided by steps (' + str(self.train_step_per_episode) + '): ' +
                         str(0) + '\n')

    def training(self):

        for train_step in range(self.train_step_per_episode):

            # Get a batch of samples from the replay buffer or PER
            if self.prioritized_replay_buffer:
                [batch, batch_indices, batch_priorities] = self.priorit_buff.sample(self.batch_size)
            else:
                batch = self.buff.getBatch(self.batch_size)

            states = []
            actions = []
            rewards = []
            self.new_states = []
            self.dones = []
            self.active_flights_m = []
            fls_with_loss_of_separation_m = []
            fls_with_conflicts_m = []
            self.next_fls_with_loss_of_separation_m = []
            self.next_fls_with_conflicts_m = []
            self.history_loss_confl_m = []
            self.next_active_flights_m = []
            reward_history_b = []
            self.durations_of_actions_b = []
            self.next_timestamp_b = []
            self.data_needed_for_delayed_update_b = []
            self.np_mask_climb_descend_res_fplan_with_confl_loss_b = []
            self.np_mask_res_fplan_after_maneuv_b = []
            self.next_executing_FL_change_b = []
            self.next_executing_direct_to_b = []
            self.next_flight_phases_b = []
            self.executing_FL_change_b = []
            self.executing_direct_to_b = []
            self.next_available_wps_b = []
            self.next_executing_resume_fplan_b = []

            self.num_of_different_matrices_needed = 3
            for i_ in range(self.num_of_different_matrices_needed):
                states.append([])
                self.new_states.append([])

            for e in batch:
                states[0].append(e[0])
                states[1].append(e[5])
                self.new_states[0].append(e[2])
                self.new_states[1].append(e[5])
                states[2].append(e[7])
                self.new_states[2].append(e[8])
                actions.append(e[1])
                rewards.append(e[3])
                self.dones.append(e[4])
                self.active_flights_m.append(e[6])
                fls_with_loss_of_separation_m.append(e[9])
                fls_with_conflicts_m.append(e[10])
                self.next_fls_with_loss_of_separation_m.append(e[11])
                self.next_fls_with_conflicts_m.append(e[12])
                self.history_loss_confl_m.append(e[13])
                self.next_active_flights_m.append(e[14])
                reward_history_b.append(e[15])
                self.durations_of_actions_b.append(e[16])
                self.next_timestamp_b.append(e[17])
                self.data_needed_for_delayed_update_b.append(e[19])
                self.next_executing_direct_to_b.append(e[21])
                self.next_executing_FL_change_b.append(e[22])
                self.np_mask_res_fplan_after_maneuv_b.append(e[23])
                self.np_mask_climb_descend_res_fplan_with_confl_loss_b.append(e[24])
                self.next_flight_phases_b.append(e[25])
                self.executing_FL_change_b.append(e[26])
                self.executing_direct_to_b.append(e[27])
                self.next_available_wps_b.append(e[28])
                self.next_executing_resume_fplan_b.append(e[29])

            actions = np.asarray(actions)
            rewards = np.asarray(rewards)
            self.dones = np.asarray(self.dones)
            self.active_flights_m = np.asarray(self.active_flights_m)
            self.next_active_flights_m = np.asarray(self.next_active_flights_m)

            for i_ in range(3):
                states[i_] = np.asarray(states[i_])
                self.new_states[i_] = np.asarray(self.new_states[i_])

            # Get q_values for the batch states
            q_values = self.DGN_m.model_predict(states)

            # Get predictions for the next states of the batch samples using the target network
            self.predict_q_values_for_next_states_using_target_net(self.batch_size)

            # Filter batch maxq
            self.get_maxq_valid_actions(batch)

            #####UPDATE Q-VALUES###
            q_values_ = DGN_m.update_q_values(len(batch),
                                              self.dones,
                                              self.n_agent,
                                              self.active_flights_m,
                                              fls_with_loss_of_separation_m,
                                              self.next_fls_with_loss_of_separation_m,
                                              fls_with_conflicts_m,
                                              self.next_fls_with_conflicts_m,
                                              self.history_loss_confl_m,
                                              q_values.copy(),
                                              actions,
                                              rewards,
                                              reward_history_b,
                                              self.next_active_flights_m,
                                              self.GAMMA,
                                              self.maxq_actions,
                                              self.durations_of_actions_b,
                                              self.data_needed_for_delayed_update_b,
                                              self.next_timestamp_b,
                                              self.different_target_q_values_mask,
                                              self.default_action_q_value_mask,
                                              self.np_mask_res_fplan_after_maneuv_b,
                                              self.np_mask_climb_descend_res_fplan_with_confl_loss_b,
                                              self.not_use_max_next_q_mask,
                                              self.next_flight_phases_b)

            if self.prioritized_replay_buffer:
                sample_weights, errors = self.priorit_buff.calculate_sample_weights(batch_priorities, self.batch_size,
                                                                                    q_values.copy(), q_values_.copy())

            else:
                sample_weights = np.ones(self.batch_size)

            history = self.DGN_m.model_fit(states, q_values_, epochs=1, batch_size=self.batch_size, verbose=0,
                                           sample_weight=sample_weights)
            self.loss += history.history['loss'][0]

            ####update prioritized replay buffer priorities####
            if self.prioritized_replay_buffer:
                self.priorit_buff.update(batch_indices, errors)

            #########train target model#########
            self.DGN_m.train_target_model()

        ########Print and store loss value########
        print('Total episode loss divided by steps (' + str(self.train_step_per_episode) + '): ' +
              str(self.loss / self.train_step_per_episode), end='\n')
        self.f.write(' ' + 'Total episode loss divided by steps (' + str(self.train_step_per_episode) + '): ' +
                     str(self.loss / self.train_step_per_episode) + '\n')

        #######save model###############
        if not self.i_episode < self.episode_before_train:
            self.DGN_m.save_model('gdn.h5')

    def close_log_files(self):

        #####close log files#####
        self.f.close()

        if self.evaluation:
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            #####Attention dictionary######
            actions_attentions_dict_file = open("./logs/actions_attentions_dict.pkl", "wb")
            pickle.dump(self.dict_actions_attentions, actions_attentions_dict_file)
            actions_attentions_dict_file.close()
            ######Actions dictionary#########
            dict_file = open("./logs/actions_dict.pkl", "wb")
            pickle.dump(self.dict_actions, dict_file)
            dict_file.close()

        # Delete unnecessary scenario debug files
        if not self.debug_:
            for scen in self.scenario_list:
                current_scenario_debug_folder = scen + '_interpolated.rdr_debug_files'
                if os.path.exists('./' + current_scenario_debug_folder):
                    os.system('rm -r ./' + current_scenario_debug_folder)
                elif os.path.exists('./env/environment/' + current_scenario_debug_folder):
                    os.system('rm -r ./env/environment/' + current_scenario_debug_folder)
                else:
                    print("There are no debug files for the scenario {} at the specified paths {} and {}".
                          format(scen, './' + current_scenario_debug_folder,
                                 './env/environment/' + current_scenario_debug_folder))

        # Sent notification to slack
        if self.with_slack_notifications and self.i_episode % self.send_slack_notifications_every_episode == 0:
            message = '"The experiment running in host {} has just ended."'.format(self.hostname)
            self.send_slack_message(message)

        ###To read attention pickle file just write:
        # actions_attentions_dict_file = open("actions_attentions_dict.pkl", "rb")
        # dict_attentions_actions = pickle.load(actions_attentions_dict_file)
        ###Dictionary form is:
        #{episode(starting from 0):
        # {timestamp:
        #   {'actions':
        #       {flight_id(including only active flights):
        #           {'action_id': e.g. 0 or 1 or 2 or ... (include zero-action with id 6),
        #           'true_action': e.g. 45 or -45 or ...,
        #           'action_type': "Delta_Course" or "Delta_speed" or ...}}},
        #   {'attentions':
        #       {'values': attention array (shape: [agent, layer, n_head, neighbor]),
        #       'labels': adjacency matrix with flight ids (shape: [agent, neighbor])}}}}


if __name__ == '__main__':

    np.random.seed(16)

    parser = argparse.ArgumentParser()
    parser.add_argument("--DGN_model_path", type=str, default="./../results/without_reg/training/1st exp")
    parser.add_argument("--evaluation", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--LRA", type=float, default=0.0001)
    parser.add_argument("--train_episodes", type=int, default=2000)
    # Exploration episodes are included in total "train_episodes"
    parser.add_argument("--exploration_episodes", type=int, default=1021)
    parser.add_argument("--prioritized_replay_buffer", type=bool, default=False)
    parser.add_argument("--scenario", type=str, default="A3_1675139981_LECBVNI_AAF765")
    parser.add_argument("--multi_scenario_training", type=bool, default=False)
    parser.add_argument("--debug_", type=bool, default=False)
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--with_RoC_term_reward", type=bool, default=False)
    parser.add_argument("--with_slack_notifications", type=bool, default=False)
    parser.add_argument("--send_slack_notifications_every_episode", type=int, default=500)
    parser.add_argument("--conc_observations_edges", type=bool, default=False)

    tt = parser.parse_args()

    #Debug mode
    debug_ = tt.debug_

    # Evaluation specifications
    evaluation = tt.evaluation

    #Edges mode
    conc_observations_edges = tt.conc_observations_edges

    # Training specifications
    DGN_model_path = tt.DGN_model_path
    continue_train = tt.continue_train
    with_slack_notifications = tt.with_slack_notifications
    send_slack_notifications_every_episode = tt.send_slack_notifications_every_episode
    batch_size = tt.batch_size
    LRA = tt.LRA
    train_episodes = tt.train_episodes
    exploration_episodes = tt.exploration_episodes
    prioritized_replay_buffer = tt.prioritized_replay_buffer
    scenario = tt.scenario
    multi_scenario_training = tt.multi_scenario_training
    with_RoC_term_reward = tt.with_RoC_term_reward

    conflict_detection_and_resolution = \
        CDR(DGN_model_path,
            evaluation,
            batch_size,
            LRA,
            train_episodes,
            exploration_episodes,
            prioritized_replay_buffer,
            scenario,
            multi_scenario_training,
            debug_,
            continue_train,
            with_RoC_term_reward,
            with_slack_notifications,
            send_slack_notifications_every_episode,
            conc_observations_edges)

    conflict_detection_and_resolution.run_CDR()


