import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def get_data(exps_dirs, file_path, count_exps, line_index, episode_length_data=False):
    data = [[] for _ in range(count_exps)]
    for directory in exps_dirs:
        exp_number = directory[0]
        with open(file_path + '/' + directory + '/' + 'episodes_log_' + exp_number + '.txt') as f:
            lines = f.readlines()
        for line_id, line in enumerate(lines):
            if line_id >= skip_num_episodes_before_train:
                splitted_line = line.split(" ")
                if episode_length_data:
                    double_splitted_line = splitted_line[line_index].split("(")[1]
                    triple_splitted_line = double_splitted_line.split(")")[0]
                    data[int(exp_number) - 1].append(float(triple_splitted_line))
                else:
                    data[int(exp_number) - 1].append(float(splitted_line[line_index]))
        if int(exp_number) == count_exps:
            break
    return data

def get_mean_min_max(np_array_data):
    if len(np_array_data.shape) > 1:
        mean_ = np.mean(np_array_data, axis=0)
        min_ = np.min(np_array_data, axis=0)
        max_ = np.max(np_array_data, axis=0)
    else:
        mean_ = np_array_data.copy()
        min_ = np_array_data.copy()
        max_ = np_array_data.copy()
    return mean_, min_, max_

def get_mean_std_sliding_window(np_array_data):
    if len(np_array_data.shape) > 1:
        mean_ = np.mean(sliding_window_view(np_array_data[0], num_episodes_of_window_to_get_average), axis=1)
        std_lower = np.min(sliding_window_view(np_array_data[0], num_episodes_of_window_to_get_average), axis=1)#mean_-np.std(sliding_window_view(np_array_data[0], num_episodes_of_window_to_get_average), axis=1)
        std_upper = np.max(sliding_window_view(np_array_data[0], num_episodes_of_window_to_get_average), axis=1)#mean_+np.std(sliding_window_view(np_array_data[0], num_episodes_of_window_to_get_average), axis=1)
    else:
        mean_ = np_array_data.copy()
        std_lower = np_array_data.copy()
        std_upper = np_array_data.copy()
    return mean_, std_lower, std_upper

def get_rewards(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    rewards = get_data(exps_dirs, file_path, count_exps, 3)
    arr_rewards = np.asarray(rewards)
    mean_, min_, max_ = get_mean_min_max(arr_rewards)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_rewards)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_losses_of_separation(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    losses_of_separ = get_data(exps_dirs, file_path, count_exps, 9)
    arr_losses_of_separ = np.asarray(losses_of_separ)
    mean_, min_, max_ = get_mean_min_max(arr_losses_of_separ)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_losses_of_separ)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_alerts(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    alerts = get_data(exps_dirs, file_path, count_exps, 19)
    arr_alerts = np.asarray(alerts)
    mean_, min_, max_ = get_mean_min_max(arr_alerts)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_alerts)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_episodes_length(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    episodes_length = get_data(exps_dirs, file_path, count_exps, 53, episode_length_data=True)
    arr_episodes_length = np.asarray(episodes_length)
    mean_, min_, max_ = get_mean_min_max(arr_episodes_length)
    return mean_, min_, max_

def get_ATC_instruction_number(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    ATC_instruction_number = get_data(exps_dirs, file_path, count_exps, 30)
    arr_ATC_instruction_number = np.asarray(ATC_instruction_number)
    mean_, min_, max_ = get_mean_min_max(arr_ATC_instruction_number)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_ATC_instruction_number)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_additional_NM(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    additional_NM = get_data(exps_dirs, file_path, count_exps, 34)
    arr_additional_NM = np.asarray(additional_NM)
    mean_, min_, max_ = get_mean_min_max(arr_additional_NM)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_additional_NM)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_total_conflicts_not_alerts_with_positive_tcpa(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    conflicts_not_alerts_with_positive_tcpa = get_data(exps_dirs, file_path, count_exps, 42)
    arr_conflicts_not_alerts_with_positive_tcpa = np.asarray(conflicts_not_alerts_with_positive_tcpa)
    mean_, min_, max_ = get_mean_min_max(arr_conflicts_not_alerts_with_positive_tcpa)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_conflicts_not_alerts_with_positive_tcpa)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows

def get_loss(file_path, used_exper):
    exps_dirs = os.listdir(file_path)
    if used_exper is None:
        count_exps = len(exps_dirs)
    else:
        count_exps = used_exper
    loss = get_data(exps_dirs, file_path, count_exps, 54)
    arr_loss = np.asarray(loss)
    mean_, min_, max_ = get_mean_min_max(arr_loss)
    mean_windows, std_lower_windows, std_upper_windows = get_mean_std_sliding_window(arr_loss)
    return mean_, min_, max_, mean_windows, std_lower_windows, std_upper_windows


def plot_data(num_plots, data, labels_list_, y_axis_label_list_, save_pl=False, legend_position='lower right', y_lim=None, x_lim=None):

    #Plot results
    plt.xlabel('Episode')
    plt.ylabel(y_axis_label_list_[0])
    for plot in range(num_plots):
        plt.fill_between(range(data[plot][0].shape[-1]), data[plot][1], data[plot][2], alpha=0.5)
        plt.plot(data[plot][0], label=labels_list_[plot])
        if y_lim is not None:
            plt.ylim(y_lim)
        if x_lim is not None:
            plt.xlim(x_lim)
    plt.legend(loc=legend_position)
    if save_pl:
        plt.savefig(y_axis_label_list_[0])
    plt.show()

    # Plot results in sliding windows
    if len(y_axis_label_list_) > 1:
        plt.xlabel('Episode')
        plt.ylabel(y_axis_label_list_[1])
        for plot in range(num_plots):
            plt.fill_between(range(data[plot][3].shape[-1]), data[plot][4], data[plot][5], alpha=0.5)
            plt.plot(data[plot][3], label=labels_list_[plot])
            if y_lim is not None:
                plt.ylim(y_lim)
            if x_lim is not None:
                plt.xlim(x_lim)
        plt.legend(loc=legend_position)
        if save_pl:
            plt.savefig(y_axis_label_list_[1])
        plt.show()


if __name__ == '__main__':

    reward_plot = True #Set this value according to whether you want to plot the mean reward of training
    losses_of_separation_plot = True #Set this value according to whether you want to plot the absolute episodes losses of separation of training
    alerts_plot = True #Set this value according to whether you want to plot the absolute episodes alerts of training
    episode_length_plot = True #Set this value according to whether you want to plot the length of training episodes
    ATC_instruction_plot = True #Set this value according to whether you want to plot the number of ATC instruction in each training episode
    loss_plot = True #Set this value according to whether you want to plot the loss of each training episode
    additional_NM_plot = True #Set this value according to whether you want to plot the additional NM of each training episode
    conflicts_not_alerts_with_positive_tcpa_plot = True #Set this value according to whether you want to plot the conflicts (not alerts) with pocitive tcpa of each training episode
    #table_results = False ##Set this value according to whether you want to create the table results for evaluation

    save_plot = True #Set this value to true if you want to save the plots, or False otherwise. Note that the plots will be saved in the same directory as this script

    used_exp = None #Set this value to N, where N is the number of the experiments that you want to be used,
                    #if you want to employ the mean, min and max of N experiments,
                    #or to None if you want to use the mean, min and max of the all the available
                    #experiments.

    skip_num_episodes_before_train = 0  # Number of episodes before train to be skipped
    num_episodes_of_window_to_get_average = 100  # Specify the size of sliding window
    axes_font_size = 10 #15  # Specify font size of axes x-y
    plt.rc('axes', labelsize=axes_font_size)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=axes_font_size)

    # Set these paths according to where your experiments results are located.
    # In the provided directory, the experiments should be ordered as: '1st exp', '2nd exp', '3rd exp' etc,
    # and the results should be named in each experiment folder as: 'episodes_log_1.txt', 'episodes_log_2.txt', 'episodes_log_3.txt' etc.
    file_path_list = ['/home/georgepap/PycharmProjects/DGN_project/TAPAS_implementation/v_1.7/TAPAS_environment_final/testing_results/multiscenario/4th_alt_1/batch_size=256/r_norm=20/drift_alt_norm=6.15/lr=0.00001/with_ROC/with_prior_buffer/without_reg/training']
    labels_list = ['4Seq6Alt']
    data_dict = {'rewards': [], 'losses_of_separation': [], 'alerts': [], 'episodes_length': [], 'ATC_instruction': [], 'loss': [], 'additional_NM': [], 'conflicts_not_alerts': []}
    for file in file_path_list:
        data_dict['rewards'].append(get_rewards(file, used_exp))
        data_dict['losses_of_separation'].append(get_losses_of_separation(file, used_exp))
        data_dict['alerts'].append(get_alerts(file, used_exp))
        data_dict['episodes_length'].append(get_episodes_length(file, used_exp))
        data_dict['ATC_instruction'].append(get_ATC_instruction_number(file, used_exp))
        data_dict['loss'].append(get_loss(file, used_exp))
        data_dict['additional_NM'].append(get_additional_NM(file, used_exp))
        data_dict['conflicts_not_alerts'].append(get_total_conflicts_not_alerts_with_positive_tcpa(file, used_exp))

    if reward_plot:
        y_axis_label_list = ['Mean reward', 'Mean reward in sliding window']
        plot_data(len(file_path_list), data_dict['rewards'], labels_list, y_axis_label_list,
                  save_pl=save_plot, legend_position='lower right')

    if losses_of_separation_plot:
        y_axis_label_list = ['Total losses of separation', 'Total losses of separation in sliding window']
        plot_data(len(file_path_list), data_dict['losses_of_separation'], labels_list, y_axis_label_list,
                  save_pl=save_plot, legend_position='upper right')

        if alerts_plot:
            y_axis_label_list = ['Total alerts', 'Total alerts in sliding window']
            plot_data(len(file_path_list), data_dict['alerts'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='upper right')

        if episode_length_plot:
            y_axis_label_list = ['Episodes length']
            plot_data(len(file_path_list), data_dict['episodes_length'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='upper right')

        if ATC_instruction_plot:
            y_axis_label_list = ['Number of ATC instructions', 'Number of ATC instructions in sliding window']
            plot_data(len(file_path_list), data_dict['ATC_instruction'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='upper right')

        if loss_plot:
            y_axis_label_list = ['Training Loss', 'Training Loss in sliding window']
            plot_data(len(file_path_list), data_dict['loss'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='upper right')

        if additional_NM_plot:
            y_axis_label_list = ['Additional NM', 'Additional NM in sliding window']
            plot_data(len(file_path_list), data_dict['additional_NM'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='lower right')#, y_lim=(-50, 50))

        if conflicts_not_alerts_with_positive_tcpa_plot:
            y_axis_label_list = ['Total conflicts (not alerts) with positive tcpa', 'Total conflicts (not alerts) with positive tcpa in sliding window']
            plot_data(len(file_path_list), data_dict['conflicts_not_alerts'], labels_list, y_axis_label_list,
                      save_pl=save_plot, legend_position='upper right')
