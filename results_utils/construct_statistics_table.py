import os
import pandas as pd
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_mean_and_std", type=bool, default=False)
    tt = parser.parse_args()
    with_mean_and_std = tt.with_mean_and_std

    pd.set_option('max_columns', None)

    paths_models_dict = {
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario/36_scenarios/testing/testing_scenarios_2/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': 'All36',
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario/4th/testing/testing_scenarios_4/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': '4Sec6',
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario/6th/testing/testing_scenarios_4/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': '6Sec6',
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario_SSE/36_scenarios/testing/testing_scenarios_2/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': 'All36+SSE',
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario_SSE/4th/testing/testing_scenarios_2/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': '4Seq6+SSE',
                         '/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario_SSE/6th/testing/testing_scenarios_2/testing_dataframe_24_testing_scenarios_w_mean_std_excluding_NAN.csv': '6Seq6+SSE'}
    testing_dataframes = [pd.read_csv(path) for path in list(paths_models_dict.keys())]

    #The above dictionary maps the desired high-level columns of the dataframe which is going to be created to the indices/columns of the testing dataframe(s).
    #The first integer of the tuple determines the row of the testing dataframe(s) for the training set of the corresponding high-level column.
    #The second index determines the corresponding row for the testing set (for the same high-level column and the same testing dataframe).
    #The third element (string) determines the column of the testing dataframe(s) for the corresponding high-level column.
    #If there is a forth element (integer), this is the index after splitting the text of the specific cell
    # (determined by the column name and the row index) using split('/') or split(' ').
    high_level_columns_indices_in_test_df_dict = {'Increase/Decrease Conflicts %': (-5, -4, 'Total conflicts'),
                                                  'Increase/Decrease LoSs %': (-5, -4, 'Total losses'),
                                                  'Increase/Decrease Alerts %': (-5, -4, 'Alerts'),
                                                  'Scenario Resolved %': (-2, -1, 'Scenario', 4),
                                                  'Conflicts solved %': (-8, -7, 'Conflicts solved', 1),
                                                  'Conflicts in groups solved %': (-8, -7, 'Conflicts in groups solved', 1),
                                                  'Average conflict (in groups) resolution Duration': (-8, -7, 'Conflict (in groups) resolution duration', 2),
                                                  'Average number of ATCo Instructions': (-8, -7, 'ATC instructions', 2),
                                                  'Average mean Reward': (-8, -7, 'Mean reward', 2),
                                                  'Average additional NMs': (-8, -7, 'Add. NMs', 2)}
    sub_columns = ['Training', 'Testing']

    all_dataframes = dict(zip(list(high_level_columns_indices_in_test_df_dict.keys()),
                              [pd.DataFrame(columns=sub_columns, index=list(paths_models_dict.values()))
                               for _ in range(len(list(high_level_columns_indices_in_test_df_dict.keys())))]))
    for ind, model in enumerate(list(paths_models_dict.values())):
        for col, indices in high_level_columns_indices_in_test_df_dict.items():

            #Training set
            if len(indices) == 3:
                temp_value = testing_dataframes[ind].iloc[indices[0]][indices[2]]
            elif indices[2] not in testing_dataframes[ind]:
                temp_value = '-'
            elif len(testing_dataframes[ind].iloc[indices[0]][indices[2]].split("/")) > 1:
                temp_value = testing_dataframes[ind].iloc[indices[0]][indices[2]].split("/")[indices[3]]
            elif len(testing_dataframes[ind].iloc[indices[0]][indices[2]].split(" ")) > 1:
                temp_value = testing_dataframes[ind].iloc[indices[0]][indices[2]].split(" ")[indices[3]]
            else:
                print("\n None of the options is true for the training set")
            all_dataframes[col].loc[model][sub_columns[0]] = temp_value

            #Testing set
            if len(indices) == 3:
                temp_value = testing_dataframes[ind].iloc[indices[1]][indices[2]]
            elif indices[2] not in testing_dataframes[ind]:
                temp_value = '-'
            elif len(testing_dataframes[ind].iloc[indices[1]][indices[2]].split("/")) > 1:
                temp_value = testing_dataframes[ind].iloc[indices[1]][indices[2]].split("/")[indices[3]]
            elif len(testing_dataframes[ind].iloc[indices[1]][indices[2]].split(" ")) > 1:
                temp_value = testing_dataframes[ind].iloc[indices[1]][indices[2]].split(" ")[indices[3]]
            else:
                print("\n None of the options is true for the training set")
            all_dataframes[col].loc[model][sub_columns[1]] = temp_value

    final_dataframe = pd.concat(all_dataframes, axis=1)

    # Create the workbook to save the data within
    workbook = pd.ExcelWriter('./statistics_table.xlsx', engine='xlsxwriter')
    # Create sheets in Excel for data
    final_dataframe.to_excel(workbook, sheet_name='statistics_table')
    # Auto-adjust columns' width
    for column in final_dataframe:
        column_width = max(len(final_dataframe[column].name[0]), len(column))
        col_idx = final_dataframe.columns.get_loc(column)
        workbook.sheets['statistics_table'].set_column(col_idx+1, col_idx+1, (column_width/2)+(5.5 if with_mean_and_std else 0.0))
    # save the changes
    workbook.save()
