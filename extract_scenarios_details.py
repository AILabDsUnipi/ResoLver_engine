"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os
import argparse

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from env.environment import environment

def compute_conflicts_and_losses(edges_, t_cpa_threshold):
    edges_with_loss = edges_[edges_[:, 30] == 1]
    current_timestep_list_losses_ = edges_with_loss[:, 0:2].tolist()
    edges_with_conflicts_included_alerts = edges_[(~(edges_[:, 30] == 1)) &
                                                  (edges_[:, 31] == 1) &
                                                  (-t_cpa_threshold <= edges_[:, 2]) &
                                                  (edges_[:, 2] <= t_cpa_threshold)]
    current_timestep_list_conflicts_alerts_included_ = edges_with_conflicts_included_alerts[:, 0:2].tolist()
    return current_timestep_list_losses_, current_timestep_list_conflicts_alerts_included_

def append_new_losses_and_conflicts(edges_,
                                    current_scenario_total_losses_,
                                    current_scenario_total_losses_for_different_pairs_of_flights_,
                                    current_scenario_total_conflicts_alerts_included_,
                                    current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights_,
                                    t_cpa_threshold,
                                    current_scenario_total_losses_wo_conflict_or_loss_before_):

    current_timestep_list_losses_, current_timestep_list_conflicts_alerts_included_ = \
        compute_conflicts_and_losses(edges_, t_cpa_threshold)

    current_scenario_total_losses_wo_conflict_or_loss_before_.\
        extend([loss.copy() for loss in current_timestep_list_losses_
                if(loss not in current_scenario_total_losses_ and
                   loss not in current_scenario_total_conflicts_alerts_included_)])

    current_scenario_total_losses_.extend(current_timestep_list_losses_.copy())
    current_scenario_total_losses_for_different_pairs_of_flights_.\
        extend([loss.copy() for loss in current_timestep_list_losses_
                if loss not in current_scenario_total_losses_for_different_pairs_of_flights_])

    current_scenario_total_conflicts_alerts_included_.extend(current_timestep_list_conflicts_alerts_included_.copy())
    current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights_.\
        extend([conflict.copy() for conflict in current_timestep_list_conflicts_alerts_included_
                if conflict not in current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights_])

    return current_scenario_total_losses_, \
           current_scenario_total_losses_for_different_pairs_of_flights_, \
           current_scenario_total_conflicts_alerts_included_, \
           current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights_, \
           current_scenario_total_losses_wo_conflict_or_loss_before_


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_scenarios_only", type=bool, default=True)
    parser.add_argument("--file_path_selected_scenarios_for_testing", type=str, default="./selected_scenarios_files/")

    tt = parser.parse_args()
    selected_scenarios_only = tt.selected_scenarios_only
    file_path_selected_scenarios_for_testing = tt.file_path_selected_scenarios_for_testing

    #!!!!!Determine time to CPA threshold!!!!!!!!!
    t_cpa_threshold_ = 10 * 60  # Threshold to filter conflicts in [-t_CPA_threshold, t_CPA_threshold]

    #!!!!Determine time (in seconds) between two timesteps!!!!!!
    time_between_two_timesteps = 30

    scenarios_list = []
    if selected_scenarios_only:
        selected_scenarios_for_testing_file = open(file_path_selected_scenarios_for_testing +
                                                   'selected_scenarios_for_testing.txt', 'r')
        selected_scenarios_for_testing_file_lines = selected_scenarios_for_testing_file.readlines()
        for line in selected_scenarios_for_testing_file_lines:
            scenarios_list.append(line.split('\n')[0])
    else:
        current_path_for_training_data = 'null'
        if os.path.exists('./env/training_data'):
            current_path_for_training_data = './env/training_data'
        else:
            print("\n Path to training data is not valid!!!")
            exit(0)
        files_list = os.listdir(current_path_for_training_data)
        for file in files_list:
            splitted_file_name = file.split("_")
            if splitted_file_name[-1] == 'interpolated':
                scenarios_list.append(splitted_file_name[0]+"_"+splitted_file_name[1]+"_"+
                                      splitted_file_name[2]+"_"+splitted_file_name[3])

    scenarios_details = pd.DataFrame(columns=['Scenario',
                                              'Total conflicts',
                                              'Different pairs of flights in conflict',
                                              'Total losses',
                                              'Different pairs of flights in loss',
                                              'Losses without loss/conflict before',
                                              'Alerts',
                                              'Total number of flights',
                                              'Total time'])

    for scenario_idx, current_scenario in enumerate(scenarios_list):

        print('\nScenario: ' + current_scenario + ', number: ' + str(scenario_idx))

        current_scenario_total_losses = []
        current_scenario_total_losses_for_different_pairs_of_flights = []
        current_scenario_total_conflicts_alerts_included = []
        current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights = []
        current_scenario_total_losses_wo_conflict_or_loss_before = []

        ###One episode from historical data###
        env = environment.Environment(current_scenario)
        concated_states,\
        edges, \
        available_wps, \
        flight_phases, \
        finished_FL_change, \
        finished_direct_to, \
        finished_resume_to_fplan, \
        executing_FL_change,\
        executing_direct_to, \
        executing_resume_to_fplan = env.initialize()

        current_scenario_total_losses, \
        current_scenario_total_losses_for_different_pairs_of_flights, \
        current_scenario_total_conflicts_alerts_included, \
        current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights,\
        current_scenario_total_losses_wo_conflict_or_loss_before = \
            append_new_losses_and_conflicts(edges.copy(),
                                            current_scenario_total_losses.copy(),
                                            current_scenario_total_losses_for_different_pairs_of_flights.copy(),
                                            current_scenario_total_conflicts_alerts_included.copy(),
                                            current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights.copy(),
                                            t_cpa_threshold_,
                                            current_scenario_total_losses_wo_conflict_or_loss_before.copy())

        res_acts_ID = np.array(['res_act_dummy_ID'] * env.flight_arr.shape[0])
        done = False
        i = 0

        while not done:
            # actions: [dcourse, dspeed, d_alt_speed, to_next_wp, from_historical, direct_to_2nd,
            #           direct_to_3rd, direct_to_4th, continue_action, duration]
            actions = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for l in range(env.flight_arr.shape[0])])
            concated_states, \
            edges, \
            reward, \
            reward_per_factor, \
            done, \
            actions, \
            available_wps, \
            flight_phases, \
            finished_FL_change, \
            finished_direct_to, \
            finished_resume_to_fplan, \
            executing_FL_change, \
            executing_direct_to, \
            executing_resume_to_fplan = env.step(actions, res_acts_ID)

            current_scenario_total_losses, \
            current_scenario_total_losses_for_different_pairs_of_flights, \
            current_scenario_total_conflicts_alerts_included, \
            current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights,\
            current_scenario_total_losses_wo_conflict_or_loss_before = \
                append_new_losses_and_conflicts(edges,
                                                current_scenario_total_losses,
                                                current_scenario_total_losses_for_different_pairs_of_flights,
                                                current_scenario_total_conflicts_alerts_included,
                                                current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights,
                                                t_cpa_threshold_,
                                                current_scenario_total_losses_wo_conflict_or_loss_before.copy())

            i += 1

        current_scenario_num_of_flights = env.flight_arr.shape[0]
        current_scenario_num_of_total_losses = env.total_losses_of_separation
        current_scenario_num_total_pairs_of_flights_in_loss = \
            len(current_scenario_total_losses_for_different_pairs_of_flights)/2
        current_scenario_num_total_alerts = env.total_alerts
        current_scenario_num_of_total_conflicts_included_alerts = \
            len(current_scenario_total_conflicts_alerts_included)/2
        current_scenario_num_of_total_pairs_of_flights_in_conflict_included_alerts = \
            len(current_scenario_total_conflicts_alerts_included_for_different_pairs_of_flights)/2
        current_scenario_total_time = i*time_between_two_timesteps #In seconds
        current_scenario_total_number_of_losses_wo_conflict_or_loss_before = \
            len(current_scenario_total_losses_wo_conflict_or_loss_before)/2

        # Check if the counted losses here is equal to the losses counted by the environment
        if current_scenario_num_of_total_losses != len(current_scenario_total_losses)/2:
            print("\n The number of losses counted here does not match with the "
                  "number of losses counted by the environment!!!!")
            print("Environment losses: {}".format(current_scenario_num_of_total_losses))
            print("Losses counted here: {}".format(len(current_scenario_total_losses)/2))
            exit(0)

        # Check if the counted conflicts here are more than the corresponding conflicts of environment.
        # Note that the environment conflicts could be more than those counted here because we apply a threshold here.
        if env.total_conflicts < current_scenario_num_of_total_conflicts_included_alerts:
            print("\n The number of conflicts counted here is greater than those counted by the environment!!!!")
            print("Environment conflicts: {}".format(env.total_conflicts))
            print("Conflicts counted here: {}".format(current_scenario_num_of_total_conflicts_included_alerts))
            exit(0)

        #Right scenario details to dataframe
        next_index = len(scenarios_details.index)+1
        scenarios_details.loc[next_index] = [current_scenario,
                                             current_scenario_num_of_total_conflicts_included_alerts,
                                             current_scenario_num_of_total_pairs_of_flights_in_conflict_included_alerts,
                                             current_scenario_num_of_total_losses,
                                             current_scenario_num_total_pairs_of_flights_in_loss,
                                             current_scenario_total_number_of_losses_wo_conflict_or_loss_before,
                                             current_scenario_num_total_alerts,
                                             current_scenario_num_of_flights,
                                             current_scenario_total_time]

    #Write dataframe to csv
    scenarios_details.to_csv('./scenarios_details.csv', index=False)

    # Delete unnecessary scenario debug files
    for scen in scenarios_list:
        current_scenario_debug_folder = scen + '_interpolated.rdr_debug_files'
        if os.path.exists('./' + current_scenario_debug_folder):
            os.system('rm -r ./' + current_scenario_debug_folder)
        elif os.path.exists('./env/environment/' + current_scenario_debug_folder):
            os.system('rm -r ./env/environment/' + current_scenario_debug_folder)
        else:
            print("There are no debug files for the scenario {} at the specified paths {} and {}".
                  format(scen, './' + current_scenario_debug_folder, './env/environment/'
                         + current_scenario_debug_folder))
