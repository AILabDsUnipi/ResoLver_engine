import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios_details_from_selected_scenarios_only", type=bool, default=True)
    parser.add_argument("--file_path_selected_scenarios_for_testing", type=str, default="./selected_scenarios_files/")
    parser.add_argument("--file_path_selected_scenarios_of_training_for_testing", type=str, default="./selected_scenarios_files/")
    parser.add_argument("--file_path_selected_scenarios_details", type=str, default="./selected_scenarios_files/")
    # In the above number we should include any possible duplicate scenario
    parser.add_argument("--num_of_the_last_scenarios_only_for_testing", type=int, default=24)
    parser.add_argument("--dynamic_edges", type=bool, default=False)
    parser.add_argument("--conc_observations_edges", type=bool, default=False)
    parser.add_argument("--with_neighbors_without_conflict", type=bool, default=False)
    parser.add_argument("--with_mean_and_std", type=bool, default=False)
    parser.add_argument("--with_median_instead_of_mean", type=bool, default=False)
    parser.add_argument("--with_iqr_instead_of_std", type=bool, default=False)
    parser.add_argument("--assign_zero_action_randomly", type=bool, default=False)
    parser.add_argument("--randomness_percentage_for_assigning_zero_action", type=int, default=25)
    parser.add_argument("--boxplots", type=bool, default=False)
    parser.add_argument("--boxplots_font_size", type=int, default=15)
    parser.add_argument("--boxplots_linewidth", type=float, default=1.5)
    parser.add_argument("--boxplots_mean_markers_size", type=int, default=15)
    parser.add_argument("--boxplots_outlier_markets_size", type=int, default=15)
    parser.add_argument("--boxplots_title", type=str, default='6Seq6 results')
    parser.add_argument("--save_data_for_boxplots", type=bool, default=False)

    tt = parser.parse_args()
    scenarios_details_from_selected_scenarios_only = tt.scenarios_details_from_selected_scenarios_only
    file_path_selected_scenarios_for_testing = tt.file_path_selected_scenarios_for_testing
    file_path_selected_scenarios_of_training_for_testing = tt.file_path_selected_scenarios_of_training_for_testing
    file_path_selected_scenarios_details = tt.file_path_selected_scenarios_details
    num_of_the_last_scenarios_only_for_testing = tt.num_of_the_last_scenarios_only_for_testing
    dynamic_edges = tt.dynamic_edges
    conc_observations_edges = tt.conc_observations_edges
    with_neighbors_without_conflict = tt.with_neighbors_without_conflict
    with_mean_and_std = tt.with_mean_and_std
    with_median_instead_of_mean = tt.with_median_instead_of_mean
    with_iqr_instead_of_std = tt.with_iqr_instead_of_std
    assign_zero_action_randomly = tt.assign_zero_action_randomly
    randomness_percentage_for_assigning_zero_action = tt.randomness_percentage_for_assigning_zero_action
    boxplots = tt.boxplots
    boxplots_font_size = tt.boxplots_font_size
    boxplots_linewidth = tt.boxplots_linewidth
    boxplots_mean_markers_size = tt.boxplots_mean_markers_size
    boxplots_outlier_markers_size = tt.boxplots_mean_markers_size
    boxplots_title = tt.boxplots_title
    save_data_for_boxplots = tt.save_data_for_boxplots

    if dynamic_edges and conc_observations_edges:
        print("\n You should choose either 'dynamic_edges' or 'conc_observations_edges' but not both of them!!!")
        exit(0)

    if with_median_instead_of_mean and not with_mean_and_std:
        print("\n You should set the flag 'with_mean_and_std' to True when 'with_median_instead_of_mean' is selected to be True!!!")
        exit(0)

    if boxplots and not with_mean_and_std:
        print("\n You should set the flag 'with_mean_and_std' to True when 'boxplots' is selected to be True!!!")
        exit(0)

    if with_iqr_instead_of_std and (not with_mean_and_std or not with_median_instead_of_mean):
        print("\n You should set the flags 'with_mean_and_std' and 'with_median_instead_of_mean' to True "
              "when 'with_iqr_instead_of_std' is selected to be True!!!")
        exit(0)

    if save_data_for_boxplots and not with_mean_and_std:
        print("\n You should set the flag 'with_mean_and_std' to True when 'save_data_for_boxplots' is selected to be True!!!")
        exit(0)

    if not os.path.exists('./testing_scenarios'):
        os.mkdir("./testing_scenarios")

    testing_dataframe = \
        pd.DataFrame(columns=['Scenario', 'Total conflicts', 'Conflicts solved', 'Conflicts in groups solved',
                              'Conflict (in groups) resolution duration', 'Different pairs of flights in conflict',
                              'Total losses', 'Different pairs of flights in loss',
                              'Losses without loss/conflict before', 'Alerts', 'ATC instructions', 'Mean reward',
                              'Add. NMs', 'Trained'])

    # File with all scenarios which are to be used for testing.
    # This file should contain the testing only scenarios in the last 'num_of_the_last_scenarios_only_for_testing' lines.
    # No duplicates are accepted for only testing scenarios (the duplicates might be from the other scenarios, too).
    # The duplicates of the other scenarios (e.g, a scenario is contained two or more times to training scenarios,
    # or a scenario which belongs to training scenarios exists also in the set of the other scenarios) will be removed.
    selected_scenarios_for_testing_file = \
        open(file_path_selected_scenarios_for_testing + 'selected_scenarios_for_testing.txt', 'r')
    selected_scenarios_for_testing_file_lines = selected_scenarios_for_testing_file.readlines()
    selected_scenarios_for_testing = []
    for line_id, line in enumerate(selected_scenarios_for_testing_file_lines):
        current_scenario = line.split('\n')[0]
        if current_scenario in selected_scenarios_for_testing:
            if line_id >= (len(selected_scenarios_for_testing_file_lines)-num_of_the_last_scenarios_only_for_testing):
                print("\n A scenario which belongs in the only testing scenarios is already in the selected scenarios!!!")
                exit(0)
            continue
        else:
            selected_scenarios_for_testing.append(current_scenario)
    total_num_of_scenarios_for_testing = len(selected_scenarios_for_testing)

    #File with the scenarios used only for training
    file_of_selected_scenarios_of_training = \
        open(file_path_selected_scenarios_of_training_for_testing + 'selected_scenarios_of_training_for_testing.txt', 'r')
    file_of_selected_scenarios_of_training_lines = file_of_selected_scenarios_of_training.readlines()
    selected_scenarios_of_training = np.unique([line.split('\n')[0] for line in file_of_selected_scenarios_of_training_lines]).tolist()
    num_of_scenarios_trained_with = len(selected_scenarios_of_training)

    #Check if there are training scenarios in testing scenarios
    for scenario in selected_scenarios_for_testing[-num_of_the_last_scenarios_only_for_testing:]:
        if scenario in selected_scenarios_of_training:
            print("\n Scenario {} belongs both to training and only testing scenarios!!!!!".format(scenario))
            exit(0)

    #Check if there are training scenarios which do not exist in testing scenarios (not in only testing)
    for scenario in selected_scenarios_of_training:
        if scenario not in selected_scenarios_for_testing:
            print("\n The training scenario {} is not included in testing scenarios!!".format(scenario))

    for idx, current_scenario in enumerate(selected_scenarios_for_testing):
        if os.path.exists('./testing_scenarios/scenario='+current_scenario):
            episodes_log_file = open('./testing_scenarios/scenario='+current_scenario+'/episodes_log.txt', 'r')
            solved_confl_and_further_info_file = open('./testing_scenarios/scenario='+current_scenario +
                                                      '/logs/solved_confl_and_further_info.txt', 'r')
        else:
            print("\n \n Scenario: {}, number: {}".format(current_scenario, idx+1))
            os.system("python runexp.py --DGN_model_path='.' --evaluation=True --with_RoC_term_reward=True --scenario=" +
                      current_scenario +
                      (' --dynamic_edges=True' if dynamic_edges else ('' if not conc_observations_edges else
                                                                      ' --conc_observations_edges=True')) +
                      (' --with_neighbors_without_conflict=True' if with_neighbors_without_conflict else '') +
                      ((' --assign_zero_action_randomly=True --randomness_percentage_for_assigning_zero_action=' +
                       str(randomness_percentage_for_assigning_zero_action)) if assign_zero_action_randomly else ''))

            episodes_log_file = open('episodes_log.txt', 'r')
            solved_confl_and_further_info_file = open('./logs/solved_confl_and_further_info.txt', 'r')

        splitted_line_episodes_log_file = episodes_log_file.readline().split(' ')
        lines_solved_confl_and_further_info_file = solved_confl_and_further_info_file.readlines()
        testing_dataframe.loc[len(testing_dataframe.index)] = [current_scenario,
                                                               float(lines_solved_confl_and_further_info_file[1].split(' ')[2]),
                                                               float(lines_solved_confl_and_further_info_file[0].split(' ')[2]),
                                                               '-' if len(lines_solved_confl_and_further_info_file) < 7 else
                                                               float(lines_solved_confl_and_further_info_file[6].split(' ')[4]),
                                                               '-' if len(lines_solved_confl_and_further_info_file) < 8 else
                                                               float(lines_solved_confl_and_further_info_file[7].split(' ')[4]),
                                                               float(lines_solved_confl_and_further_info_file[2].split(' ')[4]),
                                                               float(lines_solved_confl_and_further_info_file[3].split(' ')[2]),
                                                               float(lines_solved_confl_and_further_info_file[4].split(' ')[5]),
                                                               float(lines_solved_confl_and_further_info_file[5].split(' ')[4]),
                                                               float(splitted_line_episodes_log_file[19]),
                                                               float(splitted_line_episodes_log_file[30]),
                                                               float("{0:.2f}".format(float(splitted_line_episodes_log_file[3]))),
                                                               float("{0:.2f}".format(float(splitted_line_episodes_log_file[34]))),
                                                               True if current_scenario in selected_scenarios_of_training else False]

        episodes_log_file.close()
        solved_confl_and_further_info_file.close()
        if not os.path.exists('./testing_scenarios/scenario=' + current_scenario):
            os.mkdir('./testing_scenarios/scenario='+current_scenario)
            os.system('mv episodes_log.txt ./testing_scenarios/scenario=' + current_scenario + '/episodes_log.txt')
            os.system('mv logs ./testing_scenarios/scenario=' + current_scenario + '/logs')
            os.system('mv ./env/environment/log ./testing_scenarios/scenario=' + current_scenario + '/log')
            os.system('mv heatmaps_by_agent ./testing_scenarios/scenario=' + current_scenario + '/heatmaps_by_agent')

    #Get the corresponding details of the original scenarios
    path_to_scenarios_details = 'null'
    if scenarios_details_from_selected_scenarios_only:
        path_to_scenarios_details = file_path_selected_scenarios_details + 'selected_scenarios_details.csv'
    else:
        path_to_scenarios_details = file_path_selected_scenarios_details + 'scenarios_details.csv'
    scenarios_details_dataframe = pd.read_csv(path_to_scenarios_details)
    #Initialize 'scenarios_details_trained_with' as an empty dataframe with column names only
    scenarios_details_trained_with = pd.DataFrame(columns=['Scenario',
                                                            'Total conflicts',
                                                            'Different pairs of flights in conflict',
                                                            'Total losses',
                                                            'Different pairs of flights in loss',
                                                            'Losses without loss/conflict before',
                                                            'Alerts',
                                                            'Total number of flights',
                                                            'Total time'])
    #Initialize 'scenarios_details_testing_only' as an empty dataframe with column names only
    scenarios_details_testing_only = scenarios_details_trained_with.copy()
    #Initialize 'all_scenarios_details' as an empty dataframe with column names only
    all_scenarios_details = scenarios_details_trained_with.copy()
    for idx, current_scenario in enumerate(selected_scenarios_for_testing):
        # Check if the 'current scenario exists in 'scenarios_details_dataframe'.
        if len(scenarios_details_dataframe[scenarios_details_dataframe['Scenario'] == current_scenario]) == 0:
            print("\n Scenario {} does not exist in 'scenarios_details_dataframe'!!!".format(current_scenario))
            exit(0)
        # Get only the first element because many duplicates might exist.
        all_scenarios_details = \
            all_scenarios_details.append(scenarios_details_dataframe[scenarios_details_dataframe['Scenario'] == current_scenario][0:1],
                                         ignore_index=True)
        if current_scenario in selected_scenarios_of_training:
            scenarios_details_trained_with = \
                scenarios_details_trained_with.append(scenarios_details_dataframe[scenarios_details_dataframe['Scenario'] == current_scenario][0:1],
                                                      ignore_index=True)
        elif idx >= (total_num_of_scenarios_for_testing-num_of_the_last_scenarios_only_for_testing):
            scenarios_details_testing_only = \
                scenarios_details_testing_only.append(scenarios_details_dataframe[scenarios_details_dataframe['Scenario'] == current_scenario][0:1],
                                                      ignore_index=True)

    #Write percentages and total numbers. Note that for testing we count only the scenarios that were used exclusively for testing.
    if not with_mean_and_std:
        total_conflicts = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum()/total_num_of_scenarios_for_testing, ".2f"))

        conflicts_solved = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].sum()) + "/" + \
            str(format(0 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].sum() /
                         testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum())*100), ".2f")) + "%/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].sum()/total_num_of_scenarios_for_testing, ".2f"))

        conflicts_in_groups_solved = \
            '-/-/-' if isinstance(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum(), str) else \
                (str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum()) + "/" +
                 str(format(0 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                            ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() /
                              testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum()) * 100), ".2f")) + "%/" +
                 str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() /
                            total_num_of_scenarios_for_testing, ".2f")))

        conflict_in_groups_resolution_duration = \
            '-/-/-' if isinstance(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflict (in groups) resolution duration'].sum(), str) else\
            ((str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflict (in groups) resolution duration'].sum()) if
              testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() > 0 else '-') + "/-/" +
             ('-' if testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() == 0 else
              str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflict (in groups) resolution duration'].sum() /
                         len(testing_dataframe[-total_num_of_scenarios_for_testing:][testing_dataframe['Conflicts in groups solved'] > 0]), ".2f"))))

        different_pairs_of_flights_in_conflict = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in conflict'].sum() /
                       total_num_of_scenarios_for_testing, ".2f"))

        total_losses = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total losses'].sum()) + "/-/" +\
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total losses'].sum()/total_num_of_scenarios_for_testing, ".2f"))

        different_pairs_of_flights_in_loss = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in loss'].sum() /
                       total_num_of_scenarios_for_testing, ".2f"))

        losses_without_loss_conflict_before = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Losses without loss/conflict before'].sum() /
                       total_num_of_scenarios_for_testing, ".2f"))

        alerts = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Alerts'].sum() / total_num_of_scenarios_for_testing, ".2f"))

        ATC_instructions = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['ATC instructions'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['ATC instructions'].sum() / total_num_of_scenarios_for_testing, ".2f"))

        mean_reward = \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Mean reward'].sum() / total_num_of_scenarios_for_testing, ".2f"))

        add_NMs = \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Add. NMs'].sum(), ".2f")) + "/-/" +\
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Add. NMs'].sum() / total_num_of_scenarios_for_testing, ".2f"))
    else:
        assert total_num_of_scenarios_for_testing == len(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].index), \
            'Problem, case 1!'

        total_conflicts = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        conflicts_solved = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].sum()) + "/" + \
            str(format(0.00 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] /
                         testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].
                         where(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                                 & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] == 0)),
                               np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(0.00 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] /
                         testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].
                         where(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                                 & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] == 0)),
                               np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + "%/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].
                       where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                               & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] == 0)),
                             np.nan).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'].
                       where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                               & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts solved'] == 0)),
                             np.nan).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        conflicts_in_groups_solved = \
            '-/-/-' if isinstance(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum(), str) else \
                (str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum()) + "/" +
                 str(format(0.00 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                     ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] /
                       testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].
                       where(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] != 0, 1).
                       where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                               & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] == 0)),
                             np.nan)) * 100).astype(float).to_frame().
                            apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" +
                 str(format(0.00 if testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].sum() == 0.0 else
                     ((testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] /
                       testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'].
                       where(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] != 0, 1).
                       where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                               & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] == 0)),
                             np.nan)) * 100).astype(float).to_frame().
                            apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) +
                 "%/" +
                 str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].
                            where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                                    & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] == 0)),
                                  np.nan).astype(float).to_frame().
                            apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                 str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].
                            where(~((testing_dataframe[-total_num_of_scenarios_for_testing:]['Total conflicts'] == 0)
                                    & (testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'] == 0)),
                                  np.nan).astype(float).to_frame().
                            apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")))

        conflict_in_groups_resolution_duration = \
            '-/-/-' if isinstance(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflict (in groups) resolution duration'].sum(), str) else \
                ((str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflict (in groups) resolution duration'].sum()) if
                  testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() > 0 else '-') + "/-/" +
                 ('-' if testing_dataframe[-total_num_of_scenarios_for_testing:]['Conflicts in groups solved'].sum() == 0 else
                  str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]
                                              [testing_dataframe['Conflicts in groups solved'] > 0]
                                              ['Conflict (in groups) resolution duration'].astype(float).to_frame().
                             apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                  str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]
                                              [testing_dataframe['Conflicts in groups solved'] > 0]
                                              ['Conflict (in groups) resolution duration'].astype(float).to_frame().
                             apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))))

        different_pairs_of_flights_in_conflict = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        total_losses = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total losses'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        different_pairs_of_flights_in_loss = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        losses_without_loss_conflict_before = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Losses without loss/conflict before'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Losses without loss/conflict before'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        alerts = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        ATC_instructions = \
            str(testing_dataframe[-total_num_of_scenarios_for_testing:]['ATC instructions'].sum()) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['ATC instructions'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['ATC instructions'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        mean_reward = \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Mean reward'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Mean reward'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        add_NMs = \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Add. NMs'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Add. NMs'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[-total_num_of_scenarios_for_testing:]['Add. NMs'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

    testing_dataframe.loc[len(testing_dataframe.index)+1] = \
        ["Total/Percent/Mean",
         total_conflicts,
         conflicts_solved,
         conflicts_in_groups_solved,
         conflict_in_groups_resolution_duration,
         different_pairs_of_flights_in_conflict,
         total_losses,
         different_pairs_of_flights_in_loss,
         losses_without_loss_conflict_before,
         alerts,
         ATC_instructions,
         mean_reward,
         add_NMs,
         " "]

    if not with_mean_and_std:
        total_conflicts_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum() / num_of_scenarios_trained_with, ".2f"))

        conflicts_solved_ = \
            str(testing_dataframe[testing_dataframe['Trained'] == True]['Conflicts solved'].sum()) + "/" + \
            str(format(0.00 if testing_dataframe[testing_dataframe['Trained'] == True]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[testing_dataframe['Trained'] == True]['Conflicts solved'].sum() /
                         testing_dataframe[testing_dataframe['Trained'] == True]['Total conflicts'].sum()) * 100), ".2f")) + "%/" + \
            str(format(testing_dataframe[testing_dataframe['Trained'] == True]['Conflicts solved'].sum() / num_of_scenarios_trained_with, ".2f"))

        conflicts_in_groups_solved_ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum(), str) else\
                (str(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum()) + "/" +
                 str(format(0.00 if testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum() == 0.0 else
                            ((testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() /
                              testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum()) * 100), ".2f")) + "%/" +
                 str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() /
                            num_of_scenarios_trained_with, ".2f")))

        conflict_in_groups_resolution_duration_ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==True]['Conflict (in groups) resolution duration'].sum(), str) else \
                ((str(testing_dataframe[testing_dataframe['Trained']==True]['Conflict (in groups) resolution duration'].sum()) if
                  testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() > 0 else '-') + "/-/" +
                 ('-' if testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() == 0 else
                  str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflict (in groups) resolution duration'].sum() /
                             (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'] > 0).sum(), ".2f"))))

        different_pairs_of_flights_in_conflict_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in conflict'].sum() /
                       num_of_scenarios_trained_with, ".2f"))

        total_losses_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Total losses'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total losses'].sum() / num_of_scenarios_trained_with, ".2f"))

        different_pairs_of_flights_in_loss_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in loss'].sum() /
                       num_of_scenarios_trained_with, ".2f"))

        losses_without_loss_conflict_before_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Losses without loss/conflict before'].sum() /
                       num_of_scenarios_trained_with, ".2f"))

        alerts_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Alerts'].sum() / num_of_scenarios_trained_with, ".2f"))

        ATC_instructions_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['ATC instructions'].sum()) + "/-/" +\
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['ATC instructions'].sum() / num_of_scenarios_trained_with, ".2f"))

        mean_reward_ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Mean reward'].sum() / num_of_scenarios_trained_with, ".2f"))

        add_NMs_ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Add. NMs'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Add. NMs'].sum() / num_of_scenarios_trained_with, ".2f"))
    else:
        assert num_of_scenarios_trained_with == len(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].index), \
            'Problem, case 2!'

        total_conflicts_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        conflicts_solved_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'].sum()) + "/" + \
            str(format(0.00 if testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] /
                         testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].
                         where(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] == 0)),
                               np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(0.00 if testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] /
                         testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].
                         where(testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] == 0)),
                               np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + \
            "%/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'].
                       where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                               & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] == 0)),
                             np.nan).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'].
                       where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                               & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts solved'] == 0)),
                             np.nan).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_conflicts_in_groups_solved_in_training_w_nans = \
            0.00 if testing_dataframe[testing_dataframe['Trained'] == True]['Total conflicts'].sum() == 0.0 else \
                ((testing_dataframe[testing_dataframe['Trained'] == True]['Conflicts in groups solved'] /
                  testing_dataframe[testing_dataframe['Trained'] == True]['Total conflicts'].
                  where(testing_dataframe[testing_dataframe['Trained'] == True]
                                         ['Total conflicts'] != 0, 1).
                  where(~((testing_dataframe[testing_dataframe['Trained'] == True]['Total conflicts'] == 0)
                          & (testing_dataframe[testing_dataframe['Trained'] == True]['Conflicts in groups solved'] == 0)),
                        np.nan)) * 100).astype(float).to_frame()

        conflicts_in_groups_solved_ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum(), str) else\
                (str(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum()) + "/" +
                 str(format(df_conflicts_in_groups_solved_in_training_w_nans.
                            apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" +
                 str(format(df_conflicts_in_groups_solved_in_training_w_nans.
                            apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) +
                 "%/" +
                 str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].
                            where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                                    & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'] == 0)),
                                  np.nan).astype(float).to_frame().
                            apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                 str(format(testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].
                            where(~((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'] == 0)
                                    & (testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'] == 0)),
                                  np.nan).astype(float).to_frame().
                            apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")))

        df_conflict_in_groups_resolution_duration_in_training = \
            np.nan if testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() == 0 \
                   else \
            (testing_dataframe[testing_dataframe['Trained']==True]
                              [testing_dataframe[testing_dataframe['Trained']==True]
                                                ['Conflicts in groups solved'] > 0]
                              ['Conflict (in groups) resolution duration'].astype(float).to_frame())

        conflict_in_groups_resolution_duration_ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==True]['Conflict (in groups) resolution duration'].sum(), str) else \
                ((str(testing_dataframe[testing_dataframe['Trained']==True]['Conflict (in groups) resolution duration'].sum()) if
                  testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() > 0 else '-') + "/-/" +
                 ('-' if testing_dataframe[testing_dataframe['Trained']==True]['Conflicts in groups solved'].sum() == 0 else
                  str(format(df_conflict_in_groups_resolution_duration_in_training.
                             apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                  str(format(df_conflict_in_groups_resolution_duration_in_training.
                             apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))))

        different_pairs_of_flights_in_conflict_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        total_losses_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Total losses'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        different_pairs_of_flights_in_loss_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        losses_without_loss_conflict_before_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Losses without loss/conflict before'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Losses without loss/conflict before'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        alerts_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_ATC_instructions_in_training = testing_dataframe[testing_dataframe['Trained']==True]['ATC instructions'].astype(float).to_frame()

        ATC_instructions_ = \
            str(testing_dataframe[testing_dataframe['Trained']==True]['ATC instructions'].sum()) + "/-/" + \
            str(format(df_ATC_instructions_in_training.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_ATC_instructions_in_training.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_mean_reward_in_training = testing_dataframe[testing_dataframe['Trained']==True]['Mean reward'].astype(float).to_frame()

        mean_reward_ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(df_mean_reward_in_training.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_mean_reward_in_training.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_add_NMs_in_training = testing_dataframe[testing_dataframe['Trained'] == True]['Add. NMs'].astype(float).to_frame()

        add_NMs_ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==True]['Add. NMs'].sum(), ".2f")) + "/-/" + \
            str(format(df_add_NMs_in_training.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_add_NMs_in_training.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

    testing_dataframe.loc[len(testing_dataframe.index)+2] = \
        ["Total/Percent/Mean in training",
         total_conflicts_,
         conflicts_solved_,
         conflicts_in_groups_solved_,
         conflict_in_groups_resolution_duration_,
         different_pairs_of_flights_in_conflict_,
         total_losses_,
         different_pairs_of_flights_in_loss_,
         losses_without_loss_conflict_before_,
         alerts_,
         ATC_instructions_,
         mean_reward_,
         add_NMs_,
         " "]

    if not with_mean_and_std:
        total_conflicts__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False][-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False][-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        conflicts_solved__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]['Conflicts solved'].sum()) + "/" + \
            str(format(0.00 if testing_dataframe[testing_dataframe['Trained']==False]
                                                [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[testing_dataframe['Trained']==False][-num_of_the_last_scenarios_only_for_testing:]['Conflicts solved'].sum() /
                         testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Total conflicts'].sum()) * 100), ".2f")) + "%/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Conflicts solved'].sum() / num_of_the_last_scenarios_only_for_testing, ".2f"))

        conflicts_in_groups_solved__ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained'] == False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Conflicts in groups solved'].sum(), str) \
                    else (str(testing_dataframe[testing_dataframe['Trained'] == False]
                                               [-num_of_the_last_scenarios_only_for_testing:]
                                               ['Conflicts in groups solved'].sum()) + "/" +
                          str(format(0.00 if testing_dataframe[testing_dataframe['Trained'] == False]
                                                              [-num_of_the_last_scenarios_only_for_testing:]
                                                              ['Total conflicts'].sum() == 0.0
                                           else
                                     ((testing_dataframe[testing_dataframe['Trained'] == False]
                                                        [-num_of_the_last_scenarios_only_for_testing:]
                                                        ['Conflicts in groups solved'].sum() /
                                       testing_dataframe[testing_dataframe['Trained'] == False]
                                                        [-num_of_the_last_scenarios_only_for_testing:]
                                                        ['Total conflicts'].sum()) * 100), ".2f")) + "%/" +
                          str(format(testing_dataframe[testing_dataframe['Trained'] == False]
                                                      [-num_of_the_last_scenarios_only_for_testing:]
                                                      ['Conflicts in groups solved'].sum() /
                                     num_of_the_last_scenarios_only_for_testing, ".2f")))

        conflict_in_groups_resolution_duration__ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Conflict (in groups) resolution duration'].sum(), str) \
                     else \
            ((str(testing_dataframe[testing_dataframe['Trained']==False]
                                   [-num_of_the_last_scenarios_only_for_testing:]
                                   ['Conflict (in groups) resolution duration'].sum())
                  if testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      ['Conflicts in groups solved'].sum() > 0
                  else '-') + "/-/" +
             ('-' if testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      ['Conflicts in groups solved'].sum() == 0
                  else
             str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                         [-num_of_the_last_scenarios_only_for_testing:]
                                         ['Conflict (in groups) resolution duration'].sum() /
                        (testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Conflicts in groups solved'] > 0).sum(), ".2f"))))

        different_pairs_of_flights_in_conflict__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in conflict'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        total_losses__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Total losses'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Total losses'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        different_pairs_of_flights_in_loss__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in loss'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        losses_without_loss_conflict_before__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Losses without loss/conflict before'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        alerts__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Alerts'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        ATC_instructions__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['ATC instructions'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['ATC instructions'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        mean_reward__ = \
            str(format(testing_dataframe[testing_dataframe['Trained'] == False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained'] == False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Mean reward'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

        add_NMs__ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Add. NMs'].sum(), ".2f")) + "/-/" +\
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Add. NMs'].sum() /
                       num_of_the_last_scenarios_only_for_testing, ".2f"))

    else:
        assert num_of_the_last_scenarios_only_for_testing == len(testing_dataframe[testing_dataframe['Trained']==False]
                                                                                  [-num_of_the_last_scenarios_only_for_testing:]
                                                                                  ['Total conflicts'].index), \
            'Problem, case 3!'

        total_conflicts__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        conflicts_solved__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]['Conflicts solved'].sum()) + "/" + \
            str(format(0 if testing_dataframe[testing_dataframe['Trained']==False]
                                             [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum() == 0.0 else
                       ((testing_dataframe[testing_dataframe['Trained']==False][-num_of_the_last_scenarios_only_for_testing:]['Conflicts solved'] /
                         testing_dataframe[testing_dataframe['Trained']==False]
                         [-num_of_the_last_scenarios_only_for_testing:]
                         ['Total conflicts'].
                         where(testing_dataframe[testing_dataframe['Trained']==False]
                                                [-num_of_the_last_scenarios_only_for_testing:]
                                                ['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==False]
                                                     [-num_of_the_last_scenarios_only_for_testing:]
                                                     ['Conflicts solved'] == 0)), np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(0.00 if testing_dataframe[testing_dataframe['Trained']==False]
                                                [-num_of_the_last_scenarios_only_for_testing:]['Total conflicts'].sum() == 0.0
                            else
                       ((testing_dataframe[testing_dataframe['Trained']==False][-num_of_the_last_scenarios_only_for_testing:]['Conflicts solved'] /
                         testing_dataframe[testing_dataframe['Trained']==False]
                         [-num_of_the_last_scenarios_only_for_testing:]
                         ['Total conflicts'].
                         where(testing_dataframe[testing_dataframe['Trained']==False]
                                                [-num_of_the_last_scenarios_only_for_testing:]
                                                ['Total conflicts'] != 0, 1).
                         where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==False]
                                                     [-num_of_the_last_scenarios_only_for_testing:]
                                                     ['Conflicts solved'] == 0)), np.nan)) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + \
            "%/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Conflicts solved'].
                         where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==False]
                                                     [-num_of_the_last_scenarios_only_for_testing:]
                                                     ['Conflicts solved'] == 0)), np.nan).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Conflicts solved'].
                         where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                   [-num_of_the_last_scenarios_only_for_testing:]
                                                   ['Total conflicts'] == 0)
                                 & (testing_dataframe[testing_dataframe['Trained']==False]
                                                     [-num_of_the_last_scenarios_only_for_testing:]
                                                     ['Conflicts solved'] == 0)), np.nan).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_conflicts_in_groups_solved_in_testing_w_nans = \
            (0.00 if testing_dataframe[testing_dataframe['Trained']==False]
                                     [-num_of_the_last_scenarios_only_for_testing:]
                                     ['Total conflicts'].sum() == 0.0
                  else
             ((testing_dataframe[testing_dataframe['Trained']==False]
                                [-num_of_the_last_scenarios_only_for_testing:]
                                ['Conflicts in groups solved'] /
               testing_dataframe[testing_dataframe['Trained']==False]
                                [-num_of_the_last_scenarios_only_for_testing:]
                                ['Total conflicts'].
               where(testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      ['Total conflicts'] != 0, 1).
              where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Total conflicts'] == 0)
                      & (testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Conflicts in groups solved'] == 0)), np.nan)) * 100).astype(float).to_frame())

        conflicts_in_groups_solved__ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==False]
                                  [-num_of_the_last_scenarios_only_for_testing:]
                                  ['Conflicts in groups solved'].sum(), str) \
                else (str(testing_dataframe[testing_dataframe['Trained']==False]
                          [-num_of_the_last_scenarios_only_for_testing:]
                          ['Conflicts in groups solved'].sum()) + "/" +
                      str(format(df_conflicts_in_groups_solved_in_testing_w_nans.
                                 apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" +
                      str(format(df_conflicts_in_groups_solved_in_testing_w_nans.
                                 apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else
                                                 stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + "%/" +
                      str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                                  [-num_of_the_last_scenarios_only_for_testing:]
                                                  ['Conflicts in groups solved'].
                                 where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                           [-num_of_the_last_scenarios_only_for_testing:]
                                                           ['Total conflicts'] == 0)
                                         & (testing_dataframe[testing_dataframe['Trained']==False]
                                                             [-num_of_the_last_scenarios_only_for_testing:]
                                                             ['Conflicts in groups solved'] == 0)), np.nan).astype(float).to_frame().
                                 apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                      str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                                  [-num_of_the_last_scenarios_only_for_testing:]
                                                  ['Conflicts in groups solved'].
                                 where(~((testing_dataframe[testing_dataframe['Trained']==False]
                                                           [-num_of_the_last_scenarios_only_for_testing:]
                                                           ['Total conflicts'] == 0)
                                         & (testing_dataframe[testing_dataframe['Trained']==False]
                                                             [-num_of_the_last_scenarios_only_for_testing:]
                                                             ['Conflicts in groups solved'] == 0)), np.nan).astype(float).to_frame().
                                 apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")))

        df_conflict_in_groups_resolution_duration_in_testing = \
            (np.nan if testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Conflicts in groups solved'].sum() == 0
                    else
             testing_dataframe[testing_dataframe['Trained']==False]
                              [-num_of_the_last_scenarios_only_for_testing:]
                              [testing_dataframe[testing_dataframe['Trained']==False]
                                                [-num_of_the_last_scenarios_only_for_testing:]
                                                ['Conflicts in groups solved'] > 0]
                              ['Conflict (in groups) resolution duration'].astype(float).to_frame())

        conflict_in_groups_resolution_duration__ = \
            '-/-/-' if isinstance(testing_dataframe[testing_dataframe['Trained']==False]
                                  [-num_of_the_last_scenarios_only_for_testing:]
                                  ['Conflict (in groups) resolution duration'].sum(), str) \
                    else \
            ((str(testing_dataframe[testing_dataframe['Trained']==False]
                                   [-num_of_the_last_scenarios_only_for_testing:]
                                   ['Conflict (in groups) resolution duration'].sum())
                  if testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      ['Conflicts in groups solved'].sum() > 0
                  else '-') + "/-/" +
                 ('-' if testing_dataframe[testing_dataframe['Trained']==False]
                         [-num_of_the_last_scenarios_only_for_testing:]
                         ['Conflicts in groups solved'].sum() == 0
                      else
                  str(format(df_conflict_in_groups_resolution_duration_in_testing.
                             apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' +
                  str(format(df_conflict_in_groups_resolution_duration_in_testing.
                             apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))))

        different_pairs_of_flights_in_conflict__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Different pairs of flights in conflict'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in conflict'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        total_losses__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Total losses'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained'] == False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Total losses'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        different_pairs_of_flights_in_loss__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Different pairs of flights in loss'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Different pairs of flights in loss'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        losses_without_loss_conflict_before__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Losses without loss/conflict before'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Losses without loss/conflict before'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        alerts__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['Alerts'].sum()) + "/-/" + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Alerts'].astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_ATC_instructions_in_testing = \
            (testing_dataframe[testing_dataframe['Trained']==False]
                              [-num_of_the_last_scenarios_only_for_testing:]
                              ['ATC instructions'].astype(float).to_frame())

        ATC_instructions__ = \
            str(testing_dataframe[testing_dataframe['Trained']==False]
                                 [-num_of_the_last_scenarios_only_for_testing:]
                                 ['ATC instructions'].sum()) + "/-/" + \
            str(format(df_ATC_instructions_in_testing.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_ATC_instructions_in_testing.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_mean_reward_in_testing = \
            (testing_dataframe[testing_dataframe['Trained']==False]
                              [-num_of_the_last_scenarios_only_for_testing:]
                              ['Mean reward'].astype(float).to_frame())

        mean_reward__ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Mean reward'].sum(), ".2f")) + "/-/" + \
            str(format(df_mean_reward_in_testing.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_mean_reward_in_testing.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

        df_add_NMs_in_testing = \
            (testing_dataframe[testing_dataframe['Trained']==False]
                              [-num_of_the_last_scenarios_only_for_testing:]
                              ['Add. NMs'].astype(float).to_frame())

        add_NMs__ = \
            str(format(testing_dataframe[testing_dataframe['Trained']==False]
                                        [-num_of_the_last_scenarios_only_for_testing:]
                                        ['Add. NMs'].sum(), ".2f")) + "/-/" + \
            str(format(df_add_NMs_in_testing.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + '\u00b1' + \
            str(format(df_add_NMs_in_testing.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f"))

    testing_dataframe.loc[len(testing_dataframe.index)+3] = \
        ["Total/Percent/Mean in testing",
         total_conflicts__,
         conflicts_solved__,
         conflicts_in_groups_solved__,
         conflict_in_groups_resolution_duration__,
         different_pairs_of_flights_in_conflict__,
         total_losses__,
         different_pairs_of_flights_in_loss__,
         losses_without_loss_conflict_before__,
         alerts__,
         ATC_instructions__,
         mean_reward__,
         add_NMs__,
         " "]

    if not with_mean_and_std:
        total_conflicts___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Total conflicts'].sum() -
                         all_scenarios_details['Total conflicts'].sum()) /
                        (all_scenarios_details['Total conflicts'].sum() if all_scenarios_details['Total conflicts'].sum() != 0.0
                                                                        else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_conflict___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Different pairs of flights in conflict'].sum() -
                         all_scenarios_details['Different pairs of flights in conflict'].sum()) /
                        (all_scenarios_details['Different pairs of flights in conflict'].sum() if
                         all_scenarios_details['Different pairs of flights in conflict'].sum() != 0.0 else 1)) * 100, ".2f")) + '%'

        total_losses___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Total losses'].sum() -
                         all_scenarios_details['Total losses'].sum()) /
                        (all_scenarios_details['Total losses'].sum() if all_scenarios_details['Total losses'].sum() != 0.0
                                                                     else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_loss___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Different pairs of flights in loss'].sum() -
                         all_scenarios_details['Different pairs of flights in loss'].sum()) /
                        (all_scenarios_details['Different pairs of flights in loss'].sum() if
                         all_scenarios_details['Different pairs of flights in loss'].sum() != 0.0
                                                                                           else 1)) * 100, ".2f")) + '%'

        losses_without_loss_conflict_before___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Losses without loss/conflict before'].sum() -
                         all_scenarios_details['Losses without loss/conflict before'].sum()) /
                        (all_scenarios_details['Losses without loss/conflict before'].sum() if
                         all_scenarios_details['Losses without loss/conflict before'].sum() != 0.0
                                                                                            else 1)) * 100, ".2f")) + '%'

        alerts___ = \
            str(format(((testing_dataframe[(testing_dataframe['Trained'] == True) | (testing_dataframe['Trained'] == False)]
                                          [-total_num_of_scenarios_for_testing:]
                                          ['Alerts'].sum() -
                         all_scenarios_details['Alerts'].sum()) /
                        (all_scenarios_details['Alerts'].sum() if all_scenarios_details['Alerts'].sum() != 0.0
                                                               else 1)) * 100, ".2f")) + '%'

    else:
        #Total conflicts
        df_total_conflicts___ = pd.merge(testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                                          [-total_num_of_scenarios_for_testing:]
                                                          [['Scenario', 'Total conflicts']].copy(),
                                         all_scenarios_details[['Scenario', 'Total conflicts']].copy(),
                                         how='left',
                                         left_on=['Scenario'],
                                         right_on=['Scenario'],
                                         suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_total_conflicts___['diff'] = \
            df_total_conflicts___['Total conflicts_testing_dataframe'] - df_total_conflicts___['Total conflicts_all_scenarios_details']

        assert (len(all_scenarios_details['Total conflicts'].index) ==
                len(df_total_conflicts___['Total conflicts_all_scenarios_details'].index)) and \
               (all_scenarios_details['Total conflicts'].sum() ==
                df_total_conflicts___['Total conflicts_all_scenarios_details'].sum()), \
               'Problem, case 4!'

        total_conflicts___ = \
            str(format(((df_total_conflicts___['diff'].div(df_total_conflicts___['Total conflicts_all_scenarios_details'].
                                                           where(df_total_conflicts___['Total conflicts_all_scenarios_details'] != 0,
                                                                 1.0).where(~((df_total_conflicts___
                                                                                ['Total conflicts_all_scenarios_details'] == 0)
                                                                               & (df_total_conflicts___['diff'] == 0)),
                                                                             np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_total_conflicts___['diff'].div(df_total_conflicts___['Total conflicts_all_scenarios_details'].
                                                           where(df_total_conflicts___['Total conflicts_all_scenarios_details'] != 0,
                                                                 1.0).where(~((df_total_conflicts___
                                                                                ['Total conflicts_all_scenarios_details'] == 0)
                                                                               & (df_total_conflicts___['diff'] == 0)),
                                                                             np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in conflict
        df_different_pairs_of_flights_in_conflict___ = \
            pd.merge(testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                      [-total_num_of_scenarios_for_testing:]
                                      [['Scenario', 'Different pairs of flights in conflict']].copy(),
                     all_scenarios_details[['Scenario', 'Different pairs of flights in conflict']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_different_pairs_of_flights_in_conflict___['diff'] = \
            df_different_pairs_of_flights_in_conflict___['Different pairs of flights in conflict_testing_dataframe'] - \
            df_different_pairs_of_flights_in_conflict___['Different pairs of flights in conflict_all_scenarios_details']

        different_pairs_of_flights_in_conflict___ = \
            str(format(((df_different_pairs_of_flights_in_conflict___['diff'].
                         div(df_different_pairs_of_flights_in_conflict___
                             ['Different pairs of flights in conflict_all_scenarios_details'].
                             where(df_different_pairs_of_flights_in_conflict___
                                   ['Different pairs of flights in conflict_all_scenarios_details'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict___
                                                 ['Different pairs of flights in conflict_all_scenarios_details'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0],  ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_conflict___['diff'].
                         div(df_different_pairs_of_flights_in_conflict___
                             ['Different pairs of flights in conflict_all_scenarios_details'].
                             where(df_different_pairs_of_flights_in_conflict___
                                   ['Different pairs of flights in conflict_all_scenarios_details'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict___
                                                 ['Different pairs of flights in conflict_all_scenarios_details'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Total losses
        df_total_losses___ = \
            pd.merge(testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                      [-total_num_of_scenarios_for_testing:]
                                      [['Scenario', 'Total losses']].copy(),
                     all_scenarios_details[['Scenario', 'Total losses']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_total_losses___['diff'] = \
            df_total_losses___['Total losses_testing_dataframe'] - \
            df_total_losses___['Total losses_all_scenarios_details']

        total_losses___ = \
            str(format(((df_total_losses___['diff'].
                         div(df_total_losses___
                             ['Total losses_all_scenarios_details'].
                             where(df_total_losses___
                                   ['Total losses_all_scenarios_details'] != 0,
                                   1.0).where(~((df_total_losses___
                                                 ['Total losses_all_scenarios_details'] == 0)
                                                & (df_total_losses___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_total_losses___['diff'].
                         div(df_total_losses___
                             ['Total losses_all_scenarios_details'].
                             where(df_total_losses___
                                   ['Total losses_all_scenarios_details'] != 0,
                                   1.0).where(~((df_total_losses___
                                                 ['Total losses_all_scenarios_details'] == 0)
                                                & (df_total_losses___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in loss
        df_different_pairs_of_flights_in_loss___ = \
            pd.merge(testing_dataframe[(testing_dataframe['Trained'] == True) | (testing_dataframe['Trained'] == False)]
                                      [-total_num_of_scenarios_for_testing:]
                                      [['Scenario', 'Different pairs of flights in loss']].copy(),
                     all_scenarios_details[['Scenario', 'Different pairs of flights in loss']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_different_pairs_of_flights_in_loss___['diff'] = \
            df_different_pairs_of_flights_in_loss___['Different pairs of flights in loss_testing_dataframe'] - \
            df_different_pairs_of_flights_in_loss___['Different pairs of flights in loss_all_scenarios_details']

        different_pairs_of_flights_in_loss___ = \
            str(format(((df_different_pairs_of_flights_in_loss___['diff'].
                         div(df_different_pairs_of_flights_in_loss___
                             ['Different pairs of flights in loss_all_scenarios_details'].
                             where(df_different_pairs_of_flights_in_loss___
                                   ['Different pairs of flights in loss_all_scenarios_details'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss___
                                                 ['Different pairs of flights in loss_all_scenarios_details'] == 0)
                                                & (df_different_pairs_of_flights_in_loss___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_loss___['diff'].
                         div(df_different_pairs_of_flights_in_loss___
                             ['Different pairs of flights in loss_all_scenarios_details'].
                             where(df_different_pairs_of_flights_in_loss___
                                   ['Different pairs of flights in loss_all_scenarios_details'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss___
                                                 ['Different pairs of flights in loss_all_scenarios_details'] == 0)
                                                & (df_different_pairs_of_flights_in_loss___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Losses without loss conflict before
        df_losses_without_loss_conflict_before___ = \
            pd.merge(testing_dataframe[(testing_dataframe['Trained'] == True) | (testing_dataframe['Trained'] == False)]
                                      [-total_num_of_scenarios_for_testing:]
                                      [['Scenario', 'Losses without loss/conflict before']].copy(),
                     all_scenarios_details[['Scenario', 'Losses without loss/conflict before']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_losses_without_loss_conflict_before___['diff'] = \
            df_losses_without_loss_conflict_before___['Losses without loss/conflict before_testing_dataframe'] - \
            df_losses_without_loss_conflict_before___['Losses without loss/conflict before_all_scenarios_details']

        losses_without_loss_conflict_before___ = \
            str(format(((df_losses_without_loss_conflict_before___['diff'].
                         div(df_losses_without_loss_conflict_before___
                             ['Losses without loss/conflict before_all_scenarios_details'].
                             where(df_losses_without_loss_conflict_before___
                                   ['Losses without loss/conflict before_all_scenarios_details'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before___
                                                 ['Losses without loss/conflict before_all_scenarios_details'] == 0)
                                                & (df_losses_without_loss_conflict_before___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_losses_without_loss_conflict_before___['diff'].
                         div(df_losses_without_loss_conflict_before___
                             ['Losses without loss/conflict before_all_scenarios_details'].
                             where(df_losses_without_loss_conflict_before___
                                   ['Losses without loss/conflict before_all_scenarios_details'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before___
                                                 ['Losses without loss/conflict before_all_scenarios_details'] == 0)
                                                & (df_losses_without_loss_conflict_before___['diff'] == 0)),
                                              np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Alerts
        df_alerts___ = \
            pd.merge(testing_dataframe[(testing_dataframe['Trained'] == True) | (testing_dataframe['Trained'] == False)]
                     [-total_num_of_scenarios_for_testing:]
                     [['Scenario', 'Alerts']].copy(),
                     all_scenarios_details[['Scenario', 'Alerts']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_all_scenarios_details'])
        df_alerts___['diff'] = \
            df_alerts___['Alerts_testing_dataframe'] - \
            df_alerts___['Alerts_all_scenarios_details']

        alerts___ = \
            str(format(((df_alerts___['diff'].div(df_alerts___['Alerts_all_scenarios_details'].
                                                  where(df_alerts___['Alerts_all_scenarios_details'] != 0, 1.0).
                                                  where(~((df_alerts___['Alerts_all_scenarios_details'] == 0)
                                                          & (df_alerts___['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_alerts___['diff'].div(df_alerts___['Alerts_all_scenarios_details'].
                                                  where(df_alerts___['Alerts_all_scenarios_details'] != 0, 1.0).
                                                  where(~((df_alerts___['Alerts_all_scenarios_details'] == 0)
                                                          & (df_alerts___['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

    testing_dataframe.loc[len(testing_dataframe.index)+4] = \
        ["Total Incr./Decr.",
         total_conflicts___,
           "-",
           "-",
           "-",
           different_pairs_of_flights_in_conflict___,
           total_losses___,
           different_pairs_of_flights_in_loss___,
           losses_without_loss_conflict_before___,
           alerts___,
           "-",
           "-",
           "-",
           " "]

    if not with_mean_and_std:
        total_conflicts____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==True]['Total conflicts'].sum() -
                         scenarios_details_trained_with['Total conflicts'].sum()) /
                        (scenarios_details_trained_with['Total conflicts'].sum() if
                         scenarios_details_trained_with['Total conflicts'].sum() != 0.0
                                                                                 else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_conflict____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == True]['Different pairs of flights in conflict'].sum() -
                         scenarios_details_trained_with['Different pairs of flights in conflict'].sum()) /
                        (scenarios_details_trained_with['Different pairs of flights in conflict'].sum() if
                         scenarios_details_trained_with['Different pairs of flights in conflict'].sum() != 0.0
                                                                                                        else 1)) * 100, ".2f")) + '%'

        total_losses____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == True]['Total losses'].sum() -
                         scenarios_details_trained_with['Total losses'].sum()) /
                        (scenarios_details_trained_with['Total losses'].sum() if
                         scenarios_details_trained_with['Total losses'].sum() != 0.0
                                                                              else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_loss____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == True]['Different pairs of flights in loss'].sum() -
                         scenarios_details_trained_with['Different pairs of flights in loss'].sum()) /
                        (scenarios_details_trained_with['Different pairs of flights in loss'].sum() if
                         scenarios_details_trained_with['Different pairs of flights in loss'].sum() != 0.0
                                                                                                    else 1)) * 100, ".2f")) + '%'

        losses_without_loss_conflict_before____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == True]['Losses without loss/conflict before'].sum() -
                         scenarios_details_trained_with['Losses without loss/conflict before'].sum()) /
                        (scenarios_details_trained_with['Losses without loss/conflict before'].sum() if
                         scenarios_details_trained_with['Losses without loss/conflict before'].sum() != 0.0
                                                                                                     else 1)) * 100, ".2f")) + '%'

        alerts____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == True]['Alerts'].sum() -
                         scenarios_details_trained_with['Alerts'].sum()) /
                        (scenarios_details_trained_with['Alerts'].sum() if scenarios_details_trained_with['Alerts'].sum() != 0
                                                                        else 1)) * 100, ".2f")) + '%'
    else:
        #Total conflicts
        df_total_conflicts____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained']==True][['Scenario', 'Total conflicts']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Total conflicts']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_total_conflicts____['diff'] = \
            df_total_conflicts____['Total conflicts_testing_dataframe'] - df_total_conflicts____['Total conflicts_scenarios_details_trained_with']

        assert (len(scenarios_details_trained_with['Total conflicts'].index) ==
                len(df_total_conflicts____['Total conflicts_scenarios_details_trained_with'].index)) and \
               (scenarios_details_trained_with['Total conflicts'].sum() ==
                df_total_conflicts____['Total conflicts_scenarios_details_trained_with'].sum()), \
               'Problem, case 5!'

        df_total_conflicts_inc_decr_in_training_w_nans = \
            ((df_total_conflicts____['diff'].div(df_total_conflicts____['Total conflicts_scenarios_details_trained_with'].
                                                 where(df_total_conflicts____['Total conflicts_scenarios_details_trained_with'] != 0,
                                                       1.0).where(~((df_total_conflicts____
                                                                     ['Total conflicts_scenarios_details_trained_with'] == 0)
                                                                    & (df_total_conflicts____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame()

        total_conflicts____ = \
            str(format(df_total_conflicts_inc_decr_in_training_w_nans.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(df_total_conflicts_inc_decr_in_training_w_nans.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in conflict
        df_different_pairs_of_flights_in_conflict____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == True][['Scenario', 'Different pairs of flights in conflict']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Different pairs of flights in conflict']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_different_pairs_of_flights_in_conflict____['diff'] = \
            df_different_pairs_of_flights_in_conflict____['Different pairs of flights in conflict_testing_dataframe'] - \
            df_different_pairs_of_flights_in_conflict____['Different pairs of flights in conflict_scenarios_details_trained_with']

        different_pairs_of_flights_in_conflict____ = \
            str(format(((df_different_pairs_of_flights_in_conflict____['diff'].
                         div(df_different_pairs_of_flights_in_conflict____['Different pairs of flights in conflict_scenarios_details_trained_with'].
                             where(df_different_pairs_of_flights_in_conflict____
                                   ['Different pairs of flights in conflict_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict____
                                                 ['Different pairs of flights in conflict_scenarios_details_trained_with'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_conflict____['diff'].
                         div(df_different_pairs_of_flights_in_conflict____['Different pairs of flights in conflict_scenarios_details_trained_with'].
                             where(df_different_pairs_of_flights_in_conflict____
                                   ['Different pairs of flights in conflict_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict____
                                                 ['Different pairs of flights in conflict_scenarios_details_trained_with'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Total losses
        df_total_losses____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == True]
                                      [['Scenario', 'Total losses']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Total losses']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_total_losses____['diff'] = \
            df_total_losses____['Total losses_testing_dataframe'] - \
            df_total_losses____['Total losses_scenarios_details_trained_with']

        total_losses____ = \
            str(format(((df_total_losses____['diff'].
                         div(df_total_losses____['Total losses_scenarios_details_trained_with'].
                             where(df_total_losses____['Total losses_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_total_losses____['Total losses_scenarios_details_trained_with'] == 0)
                                                & (df_total_losses____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_total_losses____['diff'].
                         div(df_total_losses____['Total losses_scenarios_details_trained_with'].
                             where(df_total_losses____['Total losses_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_total_losses____['Total losses_scenarios_details_trained_with'] == 0)
                                                & (df_total_losses____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in loss
        df_different_pairs_of_flights_in_loss____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == True]
                                      [['Scenario', 'Different pairs of flights in loss']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Different pairs of flights in loss']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_different_pairs_of_flights_in_loss____['diff'] = \
            df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_testing_dataframe'] - \
            df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_scenarios_details_trained_with']

        different_pairs_of_flights_in_loss____ = \
            str(format(((df_different_pairs_of_flights_in_loss____['diff'].
                         div(df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_scenarios_details_trained_with'].
                             where(df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss____
                                                 ['Different pairs of flights in loss_scenarios_details_trained_with'] == 0)
                                                & (df_different_pairs_of_flights_in_loss____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_loss____['diff'].
                         div(df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_scenarios_details_trained_with'].
                             where(df_different_pairs_of_flights_in_loss____['Different pairs of flights in loss_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss____
                                                 ['Different pairs of flights in loss_scenarios_details_trained_with'] == 0)
                                                & (df_different_pairs_of_flights_in_loss____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Losses without loss conflict before
        df_losses_without_loss_conflict_before____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == True]
                                      [['Scenario', 'Losses without loss/conflict before']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Losses without loss/conflict before']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_losses_without_loss_conflict_before____['diff'] = \
            df_losses_without_loss_conflict_before____['Losses without loss/conflict before_testing_dataframe'] - \
            df_losses_without_loss_conflict_before____['Losses without loss/conflict before_scenarios_details_trained_with']

        losses_without_loss_conflict_before____ = \
            str(format(((df_losses_without_loss_conflict_before____['diff'].
                         div(df_losses_without_loss_conflict_before____['Losses without loss/conflict before_scenarios_details_trained_with'].
                             where(df_losses_without_loss_conflict_before____['Losses without loss/conflict before_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before____
                                                 ['Losses without loss/conflict before_scenarios_details_trained_with'] == 0)
                                                & (df_losses_without_loss_conflict_before____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_losses_without_loss_conflict_before____['diff'].
                         div(df_losses_without_loss_conflict_before____['Losses without loss/conflict before_scenarios_details_trained_with'].
                             where(df_losses_without_loss_conflict_before____['Losses without loss/conflict before_scenarios_details_trained_with'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before____
                                                 ['Losses without loss/conflict before_scenarios_details_trained_with'] == 0)
                                                & (df_losses_without_loss_conflict_before____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Total alerts
        df_alerts____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == True]
                                      [['Scenario', 'Alerts']].copy(),
                     scenarios_details_trained_with[['Scenario', 'Alerts']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_trained_with'])
        df_alerts____['diff'] = \
            df_alerts____['Alerts_testing_dataframe'] - \
            df_alerts____['Alerts_scenarios_details_trained_with']

        df_total_alerts_inc_decr_in_training_w_nans = \
            ((df_alerts____['diff'].
              div(df_alerts____['Alerts_scenarios_details_trained_with'].
                  where(df_alerts____['Alerts_scenarios_details_trained_with'] != 0,
                        1.0).where(~((df_alerts____
                                      ['Alerts_scenarios_details_trained_with'] == 0)
                                     & (df_alerts____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame()
        alerts____ = \
            str(format(df_total_alerts_inc_decr_in_training_w_nans.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(df_total_alerts_inc_decr_in_training_w_nans.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

    testing_dataframe.loc[len(testing_dataframe.index)+5] = \
        ["Incr./Decr. in training",
         total_conflicts____,
         "-",
         "-",
         "-",
         different_pairs_of_flights_in_conflict____,
         total_losses____,
         different_pairs_of_flights_in_loss____,
         losses_without_loss_conflict_before____,
         alerts____,
         "-",
         "-",
         "-",
         " "]

    if not with_mean_and_std:
        total_conflicts_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Total conflicts'].sum() -
                         scenarios_details_testing_only['Total conflicts'].sum()) /
                        (scenarios_details_testing_only['Total conflicts'].sum() if
                         scenarios_details_testing_only['Total conflicts'].sum() != 0.0
                                                                                 else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_conflict_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Different pairs of flights in conflict'].sum() -
                         scenarios_details_testing_only['Different pairs of flights in conflict'].sum()) /
                        (scenarios_details_testing_only['Different pairs of flights in conflict'].sum() if
                         scenarios_details_testing_only['Different pairs of flights in conflict'].sum() != 0.0
                                                                                                        else 1)) * 100, ".2f")) + '%'

        total_losses_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]['Total losses'].sum() -
                         scenarios_details_testing_only['Total losses'].sum()) /
                        (scenarios_details_testing_only['Total losses'].sum() if
                         scenarios_details_testing_only['Total losses'].sum() != 0.0
                                                                              else 1)) * 100, ".2f")) + '%'

        different_pairs_of_flights_in_loss_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Different pairs of flights in loss'].sum() -
                         scenarios_details_testing_only['Different pairs of flights in loss'].sum()) /
                        (scenarios_details_testing_only['Different pairs of flights in loss'].sum() if
                         scenarios_details_testing_only['Different pairs of flights in loss'].sum() != 0.0
                                                                                                    else 1)) * 100, ".2f")) + '%'

        losses_without_loss_conflict_before_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained']==False]
                                          [-num_of_the_last_scenarios_only_for_testing:]
                                          ['Losses without loss/conflict before'].sum() -
                         scenarios_details_testing_only['Losses without loss/conflict before'].sum()) /
                        (scenarios_details_testing_only['Losses without loss/conflict before'].sum() if
                         scenarios_details_testing_only['Losses without loss/conflict before'].sum() != 0.0
                                                                                                     else 1)) * 100, ".2f")) + '%'

        alerts_____ = \
            str(format(((testing_dataframe[testing_dataframe['Trained'] == False]
                                          [-num_of_the_last_scenarios_only_for_testing:]['Alerts'].sum() -
                         scenarios_details_testing_only['Alerts'].sum()) /
                        (scenarios_details_testing_only['Alerts'].sum() if scenarios_details_testing_only['Alerts'].sum() != 0.0
                                                                        else 1)) * 100, ".2f")) + '%'
    else:
        #Total conflicts
        df_total_conflicts_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Total conflicts']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Total conflicts']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_total_conflicts_____['diff'] = \
            df_total_conflicts_____['Total conflicts_testing_dataframe'] - \
            df_total_conflicts_____['Total conflicts_scenarios_details_testing_only']

        assert (len(scenarios_details_testing_only['Total conflicts'].index) ==
                len(df_total_conflicts_____['Total conflicts_scenarios_details_testing_only'].index)) and \
               (scenarios_details_testing_only['Total conflicts'].sum() ==
                df_total_conflicts_____['Total conflicts_scenarios_details_testing_only'].sum()), \
               'Problem, case 6!'

        df_total_conflicts_inc_decr_in_testing_w_nans = \
            ((df_total_conflicts_____['diff'].
              div(df_total_conflicts_____['Total conflicts_scenarios_details_testing_only'].
                  where(df_total_conflicts_____['Total conflicts_scenarios_details_testing_only'] != 0,
                        1.0).where(~((df_total_conflicts_____['Total conflicts_scenarios_details_testing_only'] == 0)
                                     & (df_total_conflicts_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame()

        total_conflicts_____ = \
            str(format(df_total_conflicts_inc_decr_in_testing_w_nans.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(df_total_conflicts_inc_decr_in_testing_w_nans.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in conflict
        df_different_pairs_of_flights_in_conflict_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Different pairs of flights in conflict']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Different pairs of flights in conflict']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_different_pairs_of_flights_in_conflict_____['diff'] = \
            df_different_pairs_of_flights_in_conflict_____['Different pairs of flights in conflict_testing_dataframe'] - \
            df_different_pairs_of_flights_in_conflict_____['Different pairs of flights in conflict_scenarios_details_testing_only']

        different_pairs_of_flights_in_conflict_____ = \
            str(format(((df_different_pairs_of_flights_in_conflict_____['diff'].
                         div(df_different_pairs_of_flights_in_conflict_____['Different pairs of flights in conflict_scenarios_details_testing_only'].
                             where(df_different_pairs_of_flights_in_conflict_____
                                   ['Different pairs of flights in conflict_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict_____
                                                 ['Different pairs of flights in conflict_scenarios_details_testing_only'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_conflict_____['diff'].
                         div(df_different_pairs_of_flights_in_conflict_____['Different pairs of flights in conflict_scenarios_details_testing_only'].
                             where(df_different_pairs_of_flights_in_conflict_____
                                   ['Different pairs of flights in conflict_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_conflict_____
                                                 ['Different pairs of flights in conflict_scenarios_details_testing_only'] == 0)
                                                & (df_different_pairs_of_flights_in_conflict_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Total losses
        df_total_losses_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Total losses']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Total losses']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_total_losses_____['diff'] = \
            df_total_losses_____['Total losses_testing_dataframe'] - \
            df_total_losses_____['Total losses_scenarios_details_testing_only']

        total_losses_____ = \
            str(format(((df_total_losses_____['diff'].
                         div(df_total_losses_____['Total losses_scenarios_details_testing_only'].
                             where(df_total_losses_____['Total losses_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_total_losses_____['Total losses_scenarios_details_testing_only'] == 0)
                                                & (df_total_losses_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_total_losses_____['diff'].
                         div(df_total_losses_____['Total losses_scenarios_details_testing_only'].
                             where(df_total_losses_____['Total losses_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_total_losses_____['Total losses_scenarios_details_testing_only'] == 0)
                                                & (df_total_losses_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Different pairs of flights in loss
        df_different_pairs_of_flights_in_loss_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Different pairs of flights in loss']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Different pairs of flights in loss']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_different_pairs_of_flights_in_loss_____['diff'] = \
            df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_testing_dataframe'] - \
            df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_scenarios_details_testing_only']

        different_pairs_of_flights_in_loss_____ = \
            str(format(((df_different_pairs_of_flights_in_loss_____['diff'].
                         div(df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_scenarios_details_testing_only'].
                             where(df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss_____
                                                 ['Different pairs of flights in loss_scenarios_details_testing_only'] == 0)
                                                & (df_different_pairs_of_flights_in_loss_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_different_pairs_of_flights_in_loss_____['diff'].
                         div(df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_scenarios_details_testing_only'].
                             where(df_different_pairs_of_flights_in_loss_____['Different pairs of flights in loss_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_different_pairs_of_flights_in_loss_____
                                                 ['Different pairs of flights in loss_scenarios_details_testing_only'] == 0)
                                                & (df_different_pairs_of_flights_in_loss_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Losses without loss conflict before
        df_losses_without_loss_conflict_before_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained']==False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Losses without loss/conflict before']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Losses without loss/conflict before']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_losses_without_loss_conflict_before_____['diff'] = \
            df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_testing_dataframe'] - \
            df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_scenarios_details_testing_only']

        losses_without_loss_conflict_before_____ = \
            str(format(((df_losses_without_loss_conflict_before_____['diff'].
                         div(df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_scenarios_details_testing_only'].
                             where(df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before_____
                                                 ['Losses without loss/conflict before_scenarios_details_testing_only'] == 0)
                                                & (df_losses_without_loss_conflict_before_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(((df_losses_without_loss_conflict_before_____['diff'].
                         div(df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_scenarios_details_testing_only'].
                             where(df_losses_without_loss_conflict_before_____['Losses without loss/conflict before_scenarios_details_testing_only'] != 0,
                                   1.0).where(~((df_losses_without_loss_conflict_before_____
                                                 ['Losses without loss/conflict before_scenarios_details_testing_only'] == 0)
                                                & (df_losses_without_loss_conflict_before_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame().
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

        #Total alerts
        df_alerts_____ = \
            pd.merge(testing_dataframe[testing_dataframe['Trained'] == False]
                                      [-num_of_the_last_scenarios_only_for_testing:]
                                      [['Scenario', 'Alerts']].copy(),
                     scenarios_details_testing_only[['Scenario', 'Alerts']].copy(),
                     how='left',
                     left_on=['Scenario'],
                     right_on=['Scenario'],
                     suffixes=['_testing_dataframe', '_scenarios_details_testing_only'])
        df_alerts_____['diff'] = \
            df_alerts_____['Alerts_testing_dataframe'] - \
            df_alerts_____['Alerts_scenarios_details_testing_only']

        df_total_alerts_inc_decr_in_testing_w_nans = \
            ((df_alerts_____['diff'].
              div(df_alerts_____['Alerts_scenarios_details_testing_only'].
                  where(df_alerts_____['Alerts_scenarios_details_testing_only'] != 0,
                        1.0).where(~((df_alerts_____['Alerts_scenarios_details_testing_only'] == 0)
                                     & (df_alerts_____['diff'] == 0)), np.nan))) * 100).astype(float).to_frame()

        alerts_____ = \
            str(format(df_total_alerts_inc_decr_in_testing_w_nans.
                       apply(lambda x: np.mean(x) if not with_median_instead_of_mean else np.nanmedian(x)).values[0], ".2f")) + "%\u00b1" + \
            str(format(df_total_alerts_inc_decr_in_testing_w_nans.
                       apply(lambda x: np.nanstd(x, ddof=1) if not with_iqr_instead_of_std else stats.iqr(x, nan_policy='omit')).values[0], ".2f")) + '%'

    testing_dataframe.loc[len(testing_dataframe.index)+6] = \
       ["Incr./Decr. in testing",
        total_conflicts_____,
        "-",
        "-",
        "-",
        different_pairs_of_flights_in_conflict_____,
        total_losses_____,
        different_pairs_of_flights_in_loss_____,
        losses_without_loss_conflict_before_____,
        alerts_____,
        "-",
        "-",
        "-",
        " "]

    testing_dataframe.loc[len(testing_dataframe.index)+7] = \
        ["Total scenarios resolved: {0:.2f}%".format((testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                                                       [testing_dataframe['Total losses']==0]
                                                                       ['Total losses'].count()/total_num_of_scenarios_for_testing)*100),
         "Total scenarios without losses: {}".format(testing_dataframe[(testing_dataframe['Trained']==True) | (testing_dataframe['Trained']==False)]
                                                                      [testing_dataframe['Total losses']==0]
                                                                      ['Total losses'].count()),
         "Total scenarios: {}".format(total_num_of_scenarios_for_testing),
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " "]

    testing_dataframe.loc[len(testing_dataframe.index)+8] = \
        ["Total training scenarios resolved: {0:.2f}%".format((testing_dataframe[testing_dataframe['Total losses']==0]
                                                               [testing_dataframe['Trained']==True]['Total losses'].count() /
                                                                num_of_scenarios_trained_with)*100),
         "Total training scenarios without losses: {}".format(testing_dataframe[testing_dataframe['Total losses'] == 0]
                                                              [testing_dataframe['Trained']==True]['Total losses'].count()),
         "Total training scenarios: {}".format(num_of_scenarios_trained_with),
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " "]

    testing_dataframe.loc[len(testing_dataframe.index)+9] = \
        ["Total testing scenarios resolved: {0:.2f}%".format((testing_dataframe[testing_dataframe['Trained']==False]
                                                              [-num_of_the_last_scenarios_only_for_testing:][testing_dataframe['Total losses']==0]
                                                              ['Total losses'].count() /
                                                              num_of_the_last_scenarios_only_for_testing)*100),
         "Total testing scenarios without losses: {}".format(testing_dataframe[testing_dataframe['Trained']==False]
                                                             [-num_of_the_last_scenarios_only_for_testing:][testing_dataframe['Total losses']==0]
                                                             ['Total losses'].count()),
         "Total testing scenarios: {}".format(num_of_the_last_scenarios_only_for_testing),
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " ",
         " "]

    # Save the dataframe as a csv file
    testing_dataframe.to_csv("./testing_scenarios/testing_dataframe.csv", index=False)

    if boxplots or save_data_for_boxplots:

        #Define a function to construct boxplots
        def construct_boxplot(font_size_=15, linewidth_=1.5, specify_figsize_=True, figsize_tuple_=(15, 5),
                              mean_markers_size_=5, outlier_markers_size_=5,
                              df_data_=None, x_=None, y_=None, hue_=None,
                              xlabel_='', ylabel_='', xlabelpad_=1, ylabelpad_=7, boxplot_style_='darkgrid',
                              w_legend=True, legend_position_='upper_left', w_plot_title_=True, plot_title_='',
                              ax_=None):

            sns.set(style=boxplot_style_, rc={"font.size": font_size_, "axes.titlesize": font_size_, "axes.labelsize": font_size_,
                                              "xtick.labelsize": font_size_, "ytick.labelsize": font_size_, "legend.fontsize": font_size_})

            if specify_figsize_:
                plt.figure(figsize=figsize_tuple_)  # figsize default: [6.4, 4.8]
            ax = sns.boxplot(x=x_, y=y_, hue=hue_, data=df_data_,
                             palette=['whitesmoke', 'darkgray'], showmeans=True,
                             meanprops={'marker': 'o', 'markeredgecolor': 'c', 'markerfacecolor': 'c', 'markersize': mean_markers_size_},
                             boxprops={'edgecolor': 'black', "linewidth": linewidth_},
                             whiskerprops={'color': 'black', "linewidth": linewidth_},
                             capprops={'color': 'black', "linewidth": linewidth_},
                             medianprops={"color": "r", "linewidth": linewidth_},
                             flierprops={'markersize': outlier_markers_size_},
                             ax=ax_)

            if w_plot_title_:
                ax.set(title=plot_title_)
            ax.set_xlabel(xlabel_, labelpad=xlabelpad_)
            ax.set_ylabel(ylabel_, labelpad=ylabelpad_)

            if w_legend:
                handles, labels = ax.get_legend_handles_labels()
                for handle in handles:
                    handle.set_edgecolor('black')
                    handle.set_linewidth(linewidth_)

                legend = ax.legend()
                for text in legend.get_texts():
                    text.set_color('black')
                plt.legend(handles=handles, loc=legend_position_)
            else:
                ax.legend([], [], frameon=False)

        def transform_data(list_of_data, final_dataframe_column_name):
            """
            A function to transform the data according to the requirements of seaborn boxplot.

            :param list_of_data: list of tuples of the form
                   tuple(previous_column_name, new_column_name, dataframe, metric_name, train_test_label)
            :param final_dataframe_column_name: Column name of the final dataframe
            :return: pandas Dataframe with columns: [final_dataframe_column_name, 'Metric', 'Train/Test']
            """

            dfs_ = []
            metric_ = []
            df_train_test_ = []
            for data_tuple in list_of_data:
                dfs_.append(data_tuple[2].rename(columns={data_tuple[0]: data_tuple[1]}, errors="raise")[data_tuple[1]])
                metric_.extend([data_tuple[3]] * len(data_tuple[2].index))
                df_train_test_.extend([data_tuple[4]] * len(data_tuple[2].index))

            final_df_ = pd.DataFrame({final_dataframe_column_name: pd.concat(dfs_, ignore_index=True, axis=0)})
            final_df_['Metric'] = metric_
            final_df_['Train/Test'] = df_train_test_

            return final_df_

        ################################################
        # Transform the results of 'Conflicts solved in groups', 'Incr./Decr. Conflicts', and 'Incr./Decr. Alerts' in order to create a boxplot
        df_for_boxplots_1 = \
            transform_data([(0, 'CR %', df_conflicts_in_groups_solved_in_training_w_nans, 'CR', 'Train'),
                            (0, 'CR %', df_conflicts_in_groups_solved_in_testing_w_nans, 'CR', 'Test'),
                            (0, '\u00b1Conflicts %', df_total_conflicts_inc_decr_in_training_w_nans, '\u00b1Conflicts', 'Train'),
                            (0, '\u00b1Conflicts %', df_total_conflicts_inc_decr_in_testing_w_nans, '\u00b1Conflicts', 'Test'),
                            (0, '\u00b1Alerts %', df_total_alerts_inc_decr_in_training_w_nans, '\u00b1Alerts', 'Train'),
                            (0, '\u00b1Alerts %', df_total_alerts_inc_decr_in_testing_w_nans, '\u00b1Alerts', 'Test')], 'Percentage')

        if boxplots:
            # Construct boxplot for 'Conflicts solved in groups', 'Incr./Decr. Conflicts', and 'Incr./Decr. Alerts'
            construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, figsize_tuple_=(20, 7),
                              mean_markers_size_=boxplots_mean_markers_size, outlier_markers_size_=boxplots_outlier_markers_size,
                              df_data_=df_for_boxplots_1,
                              x_='Metric', y_='Percentage', hue_='Train/Test', ylabel_='Percentage (%)', ylabelpad_=7,
                              w_legend=True, legend_position_='upper left', plot_title_=boxplots_title)
            plt.savefig('./testing_scenarios/CR_IncrDecrConflicts_IncrDecrAlerts.png')
            plt.clf()
        ################################################

        ################################################
        # Transform the results of 'ATC instructions', 'Conflicts in groups resolution duration', Additional Nautical Miles,
        # and 'Mean Reward' in order to create a boxplot
        df_for_boxplots_ATC_instructions = \
            transform_data([('ATC instructions', 'RA', df_ATC_instructions_in_training, 'RA', 'Train'),
                            ('ATC instructions', 'RA', df_ATC_instructions_in_testing, 'RA', 'Test')], '#')

        df_for_boxplots_conflict_in_groups_resolution_duration = \
            transform_data([('Conflict (in groups) resolution duration', 'RAD', df_conflict_in_groups_resolution_duration_in_training, 'RAD', 'Train'),
                            ('Conflict (in groups) resolution duration', 'RAD', df_conflict_in_groups_resolution_duration_in_testing, 'RAD', 'Test')],
                           'Seconds')

        df_for_boxplots_add_NMs = \
            transform_data([('Add. NMs', 'ANMs', df_add_NMs_in_training, 'ANMs', 'Train'),
                            ('Add. NMs', 'ANMs', df_add_NMs_in_testing, 'ANMs', 'Test')],
                           'Nautical Miles')

        df_for_boxplots_mean_reward = \
            transform_data([('Mean reward', 'RW', df_mean_reward_in_training, 'RW', 'Train'),
                            ('Mean reward', 'RW', df_mean_reward_in_testing, 'RW', 'Test')],
                           'Mean Reward')

        if boxplots:
            #Construct a plot with subplots for 'ATC instructions', 'Conflicts in groups resolution duration', 'Additional Nautical Miles', and 'Mean Reward'
            f, axes = plt.subplots(2, 2, figsize=(17, 10))

            # Construct boxplot for 'ATC instructions'
            construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                              mean_markers_size_=boxplots_mean_markers_size, outlier_markers_size_=boxplots_outlier_markers_size,
                              df_data_=df_for_boxplots_ATC_instructions,
                              x_='Metric', y_='#', hue_='Train/Test', ylabel_='#', ylabelpad_=7, w_legend=False,
                              w_plot_title_=False, ax_=axes[0, 0])

            # Construct boxplot for 'Conflicts in groups resolution duration'
            construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                              mean_markers_size_=boxplots_mean_markers_size, outlier_markers_size_=boxplots_outlier_markers_size,
                              df_data_=df_for_boxplots_conflict_in_groups_resolution_duration,
                              x_='Metric', y_='Seconds', hue_='Train/Test', ylabel_='Seconds', ylabelpad_=7, w_legend=False,
                              w_plot_title_=False, ax_=axes[0, 1])

            # Construct boxplot for 'Additional Nautical Miles'
            construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                              mean_markers_size_=boxplots_mean_markers_size, outlier_markers_size_=boxplots_outlier_markers_size,
                              df_data_=df_for_boxplots_add_NMs,
                              x_='Metric', y_='Nautical Miles', hue_='Train/Test', ylabel_='Nautical Miles', ylabelpad_=7,
                              w_legend=False, w_plot_title_=False, ax_=axes[1, 0])

            # Construct boxplot for 'Mean Reward'
            construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                              mean_markers_size_=boxplots_mean_markers_size, outlier_markers_size_=boxplots_outlier_markers_size,
                              df_data_=df_for_boxplots_mean_reward,
                              x_='Metric', y_='Mean Reward', hue_='Train/Test', ylabel_='Mean Reward', ylabelpad_=7,
                              w_legend=True, legend_position_='lower right', w_plot_title_=False, ax_=axes[1, 1])

            # set the spacing between subplots
            f.tight_layout()
            f.subplots_adjust(left=0.066,
                              bottom=0.051,
                              right=0.986,
                              top=0.909,
                              wspace=0.194,
                              hspace=0.202)
            f.suptitle(boxplots_title, fontsize=boxplots_font_size)
            plt.savefig('./testing_scenarios/RA_RAD_ANMs_RW.png')
            plt.clf()
        ################################################

        # Save data for boxplots to pickle files
        if save_data_for_boxplots:
            df_for_boxplots_1.to_pickle('./testing_scenarios/df_for_boxplots_1.pkl', compression=None)
            df_for_boxplots_ATC_instructions.to_pickle('./testing_scenarios/df_for_boxplots_ATC_instructions.pkl', compression=None)
            df_for_boxplots_conflict_in_groups_resolution_duration.\
                to_pickle('./testing_scenarios/df_for_boxplots_conflict_in_groups_resolution_duration.pkl', compression=None)
            df_for_boxplots_add_NMs.to_pickle('./testing_scenarios/df_for_boxplots_add_NMs.pkl', compression=None)
            df_for_boxplots_mean_reward.to_pickle('./testing_scenarios/df_for_boxplots_mean_reward.pkl', compression=None)

