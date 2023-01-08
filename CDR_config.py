"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

from env.environment.env_config import env_config
import numpy as np

cdr_config = {}

# Separation specifications
cdr_config['horizontal_minimum'] = env_config['horizontal_sep_minimum'] * 1852 # in meters
cdr_config['vertical_minimum'] = 1000 # in feet

# Number of next way points from which the model can select one to steer the aircraft
temp_num_dir_wp = 4

# Threshold to filter conflicts in [-t_CPA_threshold, t_CPA_threshold]
cdr_config['t_CPA_threshold'] = 10 * 60

#Edges specifications
cdr_config['num_edges_features'] = 9
cdr_config['num_edges_feats_for_ROC'] = 2
cdr_config['num_norm_edges_features'] = cdr_config['num_edges_features'] + 2 + cdr_config['num_edges_feats_for_ROC']
cdr_config['mask_edges_features'] = np.array([-10.0, -10.0, 2, 2, 2, 2, 20.0, -10.0, -10, -10.0, -10.0, -10, -10])
cdr_config['mask_self_edges_features'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

#Reward preferences
cdr_config['RoC_term_weight'] = 1
cdr_config['r_norm'] = 20
cdr_config['altitude_drift_from_exit_point'] = 6.15

#ROC specifications
cdr_config['max_rateOfClosureHV'] = 5

#Actions duration specifications
cdr_config['min_duration'] = 60 #in secs
cdr_config['max_duration'] = 3*60 #in secs

# Except the duration which is equal to the interval between two consecutive timestep
number_of_discretized_duration_intervals = 3
cdr_config['interval_between_two_steps'] = 30 #in secs
#IMPORTANT!! We assume that the minimum available duration value is equal to the interval between two steps.
duration_values = [cdr_config['interval_between_two_steps']] + \
                  [cdr_config['min_duration'] +
                   ((int((cdr_config['max_duration'] - cdr_config['min_duration']) /
                         (number_of_discretized_duration_intervals-1)))*duration)
                   for duration in range(number_of_discretized_duration_intervals)]

########Actions#########
actions_delta_course_values = [+10., -10., +20, - 20]  # In degrees
true_num_actions_dc = len(actions_delta_course_values)
cdr_config['actions_delta_course'] = [action_delta_course
                                      for action_delta_course in actions_delta_course_values
                                      for _ in range(len(duration_values))]  # In degrees
actions_delta_course_durations = [duration_value for duration_value in duration_values] * \
                                 int((len(cdr_config['actions_delta_course']) / len(duration_values)))
cdr_config['num_actions_dc'] = len(cdr_config['actions_delta_course'])

actions_delta_speed_values = [+7 * 0.5144, -7 * 0.5144]  # +/- 7 in knots. Each knot is equivalent to 0.5144 m/s.
true_num_actions_ds = len(actions_delta_speed_values)
actions_delta_speed = [action_delta_speed for action_delta_speed in actions_delta_speed_values
                                          for _ in range(len(duration_values))]
actions_delta_speed_durations = [duration_value for duration_value in duration_values] * \
                                int((len(actions_delta_speed)/len(duration_values)))
cdr_config['num_actions_ds'] = len(actions_delta_speed)
cdr_config['total_duration_values'] = actions_delta_course_durations + actions_delta_speed_durations

# The following actions have no duration ('zero_action', 'follow_flight_plan' and 'continue_action'
# have no duration, or in other words, they have duration equal to the interval between two steps)
# OR the duration
# is determined by the environment
# (only for the actions:
# 'change_FL', 'direct_to_next_wp', 'direct_to_2nd_wp', 'direct_to_3rd_wp' and 'direct_to_4th_wp',
# 'resume_to_flight_plan'), e.g., when the action 'change_FL' is chosen, its duration
# depends on the time that the flight will reach the next flight level, and therefore it depends on the environment.
# For this reason, special matrices are maintained in the environment,
# ('finished_FL_change', 'finished_direct_to', 'finished_resume_to_fplan', 'executing_FL_change', 'executing_direct_to'
# and 'executing_resume_to_fplan'),
# based on which we can conclude if a flight has reached the next flight level or the desired way point.

# 'actions_delta_altitude_speed' actions change the flight level of the aircraft by changing its vertical speed.
# This means that the aircraft will increase/decrease its vertical speed according to
# the specified value until the next flight level is reached.
# Note that until the next flight level is reached, the action "continue_action" should be executed.
cdr_config['actions_delta_altitude_speed'] = [+17., -17.]  # In feet/s
cdr_config['num_actions_as'] = len(cdr_config['actions_delta_altitude_speed'])
to_next_wp = [1.0]
direct_to_2nd = [1.0]
direct_to_3rd = [1.0]
direct_to_4th = [1.0]
cdr_config['num_dir_wp'] = len(to_next_wp+direct_to_2nd+direct_to_3rd+direct_to_4th)
if temp_num_dir_wp != cdr_config['num_dir_wp']:
    print("'temp_num_dir_wp' and 'num_dir_wp' should be equal but their values are {} and {}, respectively!!!"
          .format(temp_num_dir_wp, cdr_config['num_dir_wp']))

# Zero for delta_course, delta_speed, delta_altitude_speed, to_next_wp, from_historical, direct_to_2nd, direct_to_3rd,
# direct_to_4th, continue_action
zero_action = [0.0]
continue_action = [1.0]
resume_to_fplan = [1.0]
cdr_config['actions_list'] = cdr_config['actions_delta_course'] + \
                             actions_delta_speed + \
                             cdr_config['actions_delta_altitude_speed'] + \
                             to_next_wp + \
                             direct_to_2nd + \
                             direct_to_3rd + \
                             direct_to_4th + \
                             zero_action

# The following is the total number of the different columns of 'actions' numpy array which should be
# passed to 'step' function.
# The first number is the actual number of action types (course change, horizontal speed change, vertical speed change,
# to next way point, to 2nd way point, to 3rd way point, to 4rth way point)
# and the second number is the rest actions
# which are not considered ATC instructions ("continue" action, "follow flight plan" action, "resume_to_fplan").
# 'zero' action is not referred because there is no extra column for that. It is denoted by zeros in all columns.
cdr_config['num_types_of_actions'] = 7 + 3

cdr_config['types_of_actions'] = (["S2"] * cdr_config['num_actions_dc']) + \
                                 (["A2"] * cdr_config['num_actions_ds']) + \
                                 (["A1"] * cdr_config['num_actions_as']) + \
                                 (["A3"] * cdr_config['num_dir_wp']) + \
                                 ["A4", "continue_action", "resume_fplan", "follow_plan"]

cdr_config['types_of_actions_for_evaluation'] = (["S2"] * cdr_config['num_actions_dc']) + \
                                                (["A2"] * cdr_config['num_actions_ds']) + \
                                                (["A1"] * cdr_config['num_actions_as']) + \
                                                (["A3"] * cdr_config['num_dir_wp']) + \
                                                ["A4"] + \
                                                ["CA"] + \
                                                ["RFP"]

# The following are the action values for evaluation.
# Each one is referred to an action (except the first two which are referred to the same type of action)
# in the same order as in 'types_of_action'.
cdr_config['action_values_for_evaluation'] = ([10] * int(cdr_config['num_actions_dc']/true_num_actions_dc)) + \
                                             ([-10] * int(cdr_config['num_actions_dc']/true_num_actions_dc)) + \
                                             ([20] * int(cdr_config['num_actions_dc']/true_num_actions_dc)) + \
                                             ([-20] * int(cdr_config['num_actions_dc']/true_num_actions_dc)) + \
                                             ([7] * int(cdr_config['num_actions_ds']/true_num_actions_ds)) + \
                                             ([-7] * int(cdr_config['num_actions_ds']/true_num_actions_ds)) + \
                                             [1] + [-1] + \
                                             [1] + \
                                             [2] + \
                                             [3] + \
                                             [4] + \
                                             [0] + \
                                             ["null"] + \
                                             ["null"] #null for "continue" action and "resume to flight plan" action

# 'len(cdr_config['types_of_actions'])-3' is used because of
# the extra actions 'continue', 'resume to fplan' and 'follow_flight_plan'.
# 'len(cdr_config['action_values_for_evaluation'])-2' is used because of
# the extra actions 'continue' and 'resume to fplan'.
if len(cdr_config['action_values_for_evaluation'])-2 != len(cdr_config['actions_list']) or \
        len(cdr_config['types_of_actions'])-3 != len(cdr_config['actions_list']):
    print("actions_list, action_values_for_evaluation and types_of_actions "
          "should have the same number of actions but they don't!!")
    exit(0)

cdr_config['max_poss_actions_for_evaluation'] = len(cdr_config['actions_list'])
cdr_config['max_poss_actions_for_evaluation_to_be_passed_to_env'] = 3
cdr_config['n_actions'] = len(cdr_config['actions_list'])

# From the model outputs we keep only "n_actions"
# because the last one ("continue" action) is for being used in max_net_q or for being evaluated
# in specific situations (when selected deterministically).
cdr_config['end_action_array'] = cdr_config['n_actions']

# Add 2 for the "continue" action and "resume to fplan" action.
# This is because these actions are evaluated only
# in specific situations (when selected deterministically).
cdr_config['plus_model_action'] = 2

# Define the columns of the output files
cdr_config['flights_positions_df_columns'] = ['lon',
                                              'lat',
                                              'flight',
                                              'step',
                                              'utc_timestamp',
                                              'Flight phase',
                                              'alt',
                                              'x',
                                              'speed_h_magn',
                                              'speed_v',
                                              'x-y',
                                              'alt-exit_point_alt',
                                              'd']

cdr_config['confl_resol_act_df_columns'] = ['ResolutionID',
                                            'RTKey',
                                            'ConflictID',
                                            'ResolutionActionType',
                                            'ResolutionAction',
                                            'Q-Value',
                                            'ActionRank',
                                            'AdditionalNauticalMiles',
                                            'AdditionalDuration',
                                            'Duration',
                                            'FilteredOut',
                                            'ActionInProgress',
                                            'Prioritization',
                                            'VSpeedChange',
                                            'HSpeedChange',
                                            'CourseChange',
                                            'HShiftFromExitPoint',
                                            'VShiftFromExitPoint',
                                            'Bearing']

cdr_config['htmp_by_agent_df_columns'] = ['Current_flight',
                                          'lon',
                                          'lat',
                                          'flight',
                                          'step',
                                          'utc_timestamp',
                                          'mean_all_attention',
                                          'mean_all_attention_ranking',
                                          'mean_1st_conv',
                                          'mean_1st_conv_ranking',
                                          'mean_2nd_conv',
                                          'mean_2nd_conv_ranking',
                                          'max_all',
                                          'max_all_ranking',
                                          '1st_conv_max',
                                          '1st_conv_max_ranking',
                                          '2nd_conv_max',
                                          '2nd_conv_max_ranking',
                                          'action_type',
                                          'action',
                                          'duration',
                                          'Flight phase',
                                          'alt',
                                          'x',
                                          'speed_h_magn',
                                          'speed_v',
                                          'x-y',
                                          'alt-exit_point_alt',
                                          'd',
                                          'tcpa',
                                          'dcpa',
                                          'aij',
                                          'bij',
                                          'vdcpa',
                                          'dcp',
                                          'tcp',
                                          'hdij',
                                          'vdij',
                                          'warn_type']
