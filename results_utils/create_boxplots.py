"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def construct_boxplot(font_size_=15, linewidth_=1.5, specify_figsize_=True, figsize_tuple_=(15, 5),
                      mean_markers_size_=5, outlier_markers_size_=5,
                      df_data_=None, x_=None, y_=None, hue_=None,
                      xlabel_='', ylabel_='', xlabelpad_=1, ylabelpad_=7, boxplot_style_='darkgrid',
                      w_legend=True, w_two_levels_x_labels=False, legend_position_='upper_left', w_plot_title_=True, plot_title_='',
                      ax_=None, ticks_labelsize_for_two_levels_x_labels_=13, xtick_major_pad_for_two_levels_x_labels_=33,
                      factor_xticks_first_level_pad_single_plot_=0.04, factor_xticks_first_level_pad_multiple_plots_=0.06):

    assert (w_legend and not w_two_levels_x_labels) or (not w_legend and w_two_levels_x_labels) or (not w_legend and not w_two_levels_x_labels), \
        "Arguments 'w_legend' and 'w_two_levels_x_labels' cannot be both True !!!"

    rc_params_ = {"font.size": font_size_, "axes.titlesize": font_size_, "axes.labelsize": font_size_,
                  "xtick.labelsize": font_size_, "ytick.labelsize": font_size_, "legend.fontsize": font_size_}

    if w_two_levels_x_labels:
        rc_params_["ytick.labelsize"] = ticks_labelsize_for_two_levels_x_labels_
        rc_params_["xtick.major.pad"] = xtick_major_pad_for_two_levels_x_labels_

    sns.set(style=boxplot_style_, rc=rc_params_)

    if specify_figsize_:
        plt.figure(figsize=figsize_tuple_)  # figsize default: [6.4, 4.8]
    ax = sns.boxplot(x=x_, y=y_, hue=hue_, data=df_data_,
                     showmeans=True,
                     meanprops={'marker': 'o', 'markeredgecolor': 'c', 'markerfacecolor': 'c',
                                'markersize': mean_markers_size_},
                     boxprops={'edgecolor': 'black', "linewidth": linewidth_},
                     whiskerprops={'color': 'black', "linewidth": linewidth_},
                     capprops={'color': 'black', "linewidth": linewidth_},
                     medianprops={"color": "r", "linewidth": linewidth_},
                     flierprops={'markersize': outlier_markers_size_},
                     ax=ax_) #palette=['whitesmoke', 'darkgray']

    if w_plot_title_:
        ax.set(title=plot_title_)
    ax.set_xlabel(xlabel_, labelpad=xlabelpad_)
    ax.set_ylabel(ylabel_, labelpad=ylabelpad_)

    if w_two_levels_x_labels:

        lines = ax.get_lines()
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        legend_texts = []
        legend = ax.legend()
        for text in legend.get_texts():
            legend_texts.append(text.get_text())
        text_id = 0
        for median in lines[4:len(lines):lines_per_box]:
            x, _ = (data.mean() for data in median.get_data())
            ax.text(x,
                    ax.get_ylim()[0] - ((ax.get_ylim()[1] - ax.get_ylim()[0])*(factor_xticks_first_level_pad_single_plot_ if ax_ is None else
                                                                               factor_xticks_first_level_pad_multiple_plots_)),
                    legend_texts[text_id], ha='center', va='center',
                    color='.15', fontsize=ticks_labelsize_for_two_levels_x_labels_, fontfamily='sans-serif', fontweight='normal') # rotation=45,

            if (text_id+1) % len(legend_texts) == 0:
                text_id = 0
            else:
                text_id += 1

        ax.legend([], [], frameon=False)

    elif w_legend:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--boxplots_font_size", type=int, default=15)
    parser.add_argument("--boxplots_linewidth", type=float, default=1.5)
    parser.add_argument("--boxplots_mean_markers_size", type=int, default=15)
    parser.add_argument("--boxplots_outlier_markets_size", type=int, default=15)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--w_two_levels_x_labels_instead_of_legend", type=bool, default=False)
    parser.add_argument("--ticks_labelsize_for_two_levels_x_labels", type=int, default=13)
    parser.add_argument("--xtick_major_pad_for_two_levels_x_labels", type=int, default=33)
    parser.add_argument("--factor_xticks_first_level_pad_single_plot", type=float, default=0.04)
    parser.add_argument("--factor_xticks_first_level_pad_multiple_plots", type=float, default=0.06)

    tt = parser.parse_args()

    boxplots_font_size = tt.boxplots_font_size
    boxplots_linewidth = tt.boxplots_linewidth
    boxplots_mean_markers_size = tt.boxplots_mean_markers_size
    boxplots_outlier_markers_size = tt.boxplots_mean_markers_size
    train = tt.train
    w_two_levels_x_labels_instead_of_legend = tt.w_two_levels_x_labels_instead_of_legend
    ticks_labelsize_for_two_levels_x_labels = tt.ticks_labelsize_for_two_levels_x_labels
    xtick_major_pad_for_two_levels_x_labels = tt.xtick_major_pad_for_two_levels_x_labels
    factor_xticks_first_level_pad_single_plot = tt.factor_xticks_first_level_pad_single_plot
    factor_xticks_first_level_pad_multiple_plots = tt.factor_xticks_first_level_pad_multiple_plots

    nested_lists_model_names_lists_paths_dataLabels_data = [
        ['All36', [
            ['........./df_for_boxplots_1.pkl', 'CR_IncrDecrConflicts_IncrDecrAlerts'],
            ['........./df_for_boxplots_add_NMs.pkl', 'RA'],
            ['........./df_for_boxplots_ATC_instructions.pkl', 'ANMs'],
            ['........./df_for_boxplots_conflict_in_groups_resolution_duration.pkl', 'RAD'],
            ['........./df_for_boxplots_mean_reward.pkl', 'RW']
                 ]
         ],
        ['6Seq6', [
            ['........./df_for_boxplots_1.pkl', 'CR_IncrDecrConflicts_IncrDecrAlerts'],
            ['........./df_for_boxplots_add_NMs.pkl', 'RA'],
            ['........./df_for_boxplots_ATC_instructions.pkl', 'ANMs'],
            ['........./df_for_boxplots_conflict_in_groups_resolution_duration.pkl', 'RAD'],
            ['........./df_for_boxplots_mean_reward.pkl', 'RW']
                ]
         ],
        ['All36-DGN', [
            ['........./df_for_boxplots_1.pkl', 'CR_IncrDecrConflicts_IncrDecrAlerts'],
            ['........./df_for_boxplots_add_NMs.pkl', 'RA'],
            ['........./df_for_boxplots_ATC_instructions.pkl', 'ANMs'],
            ['........./df_for_boxplots_conflict_in_groups_resolution_duration.pkl', 'RAD'],
            ['........./df_for_boxplots_mean_reward.pkl', 'RW']
                    ]
         ],
        ['6Seq6-DGN', [
            ['........./df_for_boxplots_1.pkl', 'CR_IncrDecrConflicts_IncrDecrAlerts'],
            ['........./df_for_boxplots_add_NMs.pkl', 'RA'],
            ['........./df_for_boxplots_ATC_instructions.pkl', 'ANMs'],
            ['........./df_for_boxplots_conflict_in_groups_resolution_duration.pkl', 'RAD'],
            ['........./df_for_boxplots_mean_reward.pkl', 'RW']
                    ]
         ],
    ]

    num_of_different_dataframes_per_model = len(nested_lists_model_names_lists_paths_dataLabels_data[0][1])
    different_dataframeNames_per_model = \
        [list_paths_dataLabels_data[1] for nested_list in nested_lists_model_names_lists_paths_dataLabels_data
                                       for list_paths_dataLabels_data in nested_list[1]]
    total_df = pd.DataFrame([], columns=['DataValues', 'Metric', 'Train/Test', 'Model', 'DataframeName'])
    for nested_list in nested_lists_model_names_lists_paths_dataLabels_data:
        for list_paths_dataLabels_data in nested_list[1]:
            temp_df = pd.read_pickle(list_paths_dataLabels_data[0])
            list_paths_dataLabels_data.append(temp_df.
                                              rename(columns={temp_df.columns.values.tolist()[0]: 'DataValues'}, errors="raise"))
            list_paths_dataLabels_data[-1]['Model'] = [nested_list[0]] * len(list_paths_dataLabels_data[-1].index)
            list_paths_dataLabels_data[-1]['DataframeName'] = [list_paths_dataLabels_data[1]] * len(list_paths_dataLabels_data[-1].index)
            total_df = total_df.append(list_paths_dataLabels_data[-1], ignore_index=True)

    #############################################################################################################
    ## Construct boxplot for 'Conflicts solved in groups', 'Incr./Decr. Conflicts', and 'Incr./Decr. Alerts'
    # Test
    construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, figsize_tuple_=(20, 7),
                      mean_markers_size_=boxplots_mean_markers_size,
                      outlier_markers_size_=boxplots_outlier_markers_size,
                      df_data_=total_df[((total_df['Metric'] == 'CR') | (total_df['Metric'] == '\u00b1Conflicts') | (total_df['Metric'] == '\u00b1Alerts'))
                                        & (total_df['Train/Test'] == 'Test')],
                      x_='Metric', y_='DataValues', hue_='Model', ylabel_='Percentage (%)', ylabelpad_=7,
                      w_legend=not w_two_levels_x_labels_instead_of_legend,
                      w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, legend_position_='upper left', plot_title_='',
                      ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                      xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                      factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                      factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)
    plt.savefig('./CR_IncrDecrConflicts_IncrDecrAlerts_test.png')
    plt.clf()

    # Train
    if train:
        construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, figsize_tuple_=(20, 7),
                          mean_markers_size_=boxplots_mean_markers_size,
                          outlier_markers_size_=boxplots_outlier_markers_size,
                          df_data_=total_df[((total_df['Metric'] == 'CR') | (
                                      total_df['Metric'] == '\u00b1Conflicts') | (total_df['Metric'] == '\u00b1Alerts'))
                                            & (total_df['Train/Test'] == 'Train')],
                          x_='Metric', y_='DataValues', hue_='Model', ylabel_='Percentage (%)', ylabelpad_=7,
                          w_legend=not w_two_levels_x_labels_instead_of_legend,
                          w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, legend_position_='upper left', plot_title_='',
                          ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                          xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                          factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                          factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)
        plt.savefig('./CR_IncrDecrConflicts_IncrDecrAlerts_train.png')
        plt.clf()
    #############################################################################################################

    #############################################################################################################
    ### Construct a plot with subplots for 'ATC instructions', 'Conflicts in groups resolution duration', 'Additional Nautical Miles', and 'Mean Reward'
    ## Test
    f, axes = plt.subplots(2, 2, figsize=(20, 10))

    # Construct boxplot for 'ATC instructions'
    construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                      mean_markers_size_=boxplots_mean_markers_size,
                      outlier_markers_size_=boxplots_outlier_markers_size,
                      df_data_=total_df[(total_df['Metric'] == 'RA') & (total_df['Train/Test'] == 'Test')],
                      x_='Metric', y_='DataValues', hue_='Model', ylabel_='#', ylabelpad_=7, w_legend=False,
                      w_plot_title_=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, ax_=axes[0, 0],
                      ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                      xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                      factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                      factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

    # Construct boxplot for 'Conflicts in groups resolution duration'
    construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                      mean_markers_size_=boxplots_mean_markers_size,
                      outlier_markers_size_=boxplots_outlier_markers_size,
                      df_data_=total_df[(total_df['Metric'] == 'RAD') & (total_df['Train/Test'] == 'Test')],
                      x_='Metric', y_='DataValues', hue_='Model', ylabel_='Seconds', ylabelpad_=7, w_legend=False,
                      w_plot_title_=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, ax_=axes[0, 1],
                      ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                      xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                      factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                      factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

    # Construct boxplot for 'Additional Nautical Miles'
    construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                      mean_markers_size_=boxplots_mean_markers_size,
                      outlier_markers_size_=boxplots_outlier_markers_size,
                      df_data_=total_df[(total_df['Metric'] == 'ANMs') & (total_df['Train/Test'] == 'Test')],
                      x_='Metric', y_='DataValues', hue_='Model', ylabel_='Nautical Miles', ylabelpad_=7,
                      w_legend=not w_two_levels_x_labels_instead_of_legend,
                      w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, legend_position_='lower left', w_plot_title_=False, ax_=axes[1, 0],
                      ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                      xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                      factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                      factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

    # Construct boxplot for 'Mean Reward'
    construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                      mean_markers_size_=boxplots_mean_markers_size,
                      outlier_markers_size_=boxplots_outlier_markers_size,
                      df_data_=total_df[(total_df['Metric'] == 'RW') & (total_df['Train/Test'] == 'Test')],
                      x_='Metric', y_='DataValues', hue_='Model', ylabel_='Mean Reward', ylabelpad_=7,
                      w_legend=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, w_plot_title_=False, ax_=axes[1, 1],
                      ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                      xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                      factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                      factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

    # set the spacing between subplots
    f.subplots_adjust(left=0.052,
                      bottom=0.097,
                      right=0.989,
                      top=0.978,
                      wspace=0.126,
                      hspace=0.332)
    f.suptitle('', fontsize=boxplots_font_size)
    #f.tight_layout()

    plt.savefig('./RA_RAD_ANMs_RW_test.png')
    plt.clf()

    ## Train
    if train:
        f, axes = plt.subplots(2, 2, figsize=(20, 10))

        # Construct boxplot for 'ATC instructions'
        construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                          mean_markers_size_=boxplots_mean_markers_size,
                          outlier_markers_size_=boxplots_outlier_markers_size,
                          df_data_=total_df[(total_df['Metric'] == 'RA') & (total_df['Train/Test'] == 'Train')],
                          x_='Metric', y_='DataValues', hue_='Model', ylabel_='#', ylabelpad_=7, w_legend=False,
                          w_plot_title_=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, ax_=axes[0, 0],
                          ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                          xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                          factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                          factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

        # Construct boxplot for 'Conflicts in groups resolution duration'
        construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                          mean_markers_size_=boxplots_mean_markers_size,
                          outlier_markers_size_=boxplots_outlier_markers_size,
                          df_data_=total_df[(total_df['Metric'] == 'RAD') & (total_df['Train/Test'] == 'Train')],
                          x_='Metric', y_='DataValues', hue_='Model', ylabel_='Seconds', ylabelpad_=7, w_legend=False,
                          w_plot_title_=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, ax_=axes[0, 1],
                          ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                          xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                          factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                          factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

        # Construct boxplot for 'Additional Nautical Miles'
        construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                          mean_markers_size_=boxplots_mean_markers_size,
                          outlier_markers_size_=boxplots_outlier_markers_size,
                          df_data_=total_df[(total_df['Metric'] == 'ANMs') & (total_df['Train/Test'] == 'Train')],
                          x_='Metric', y_='DataValues', hue_='Model', ylabel_='Nautical Miles', ylabelpad_=7,
                          w_legend=not w_two_levels_x_labels_instead_of_legend, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend,
                          legend_position_='lower left', w_plot_title_=False, ax_=axes[1, 0],
                          ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                          xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                          factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                          factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

        # Construct boxplot for 'Mean Reward'
        construct_boxplot(font_size_=boxplots_font_size, linewidth_=boxplots_linewidth, specify_figsize_=False,
                          mean_markers_size_=boxplots_mean_markers_size,
                          outlier_markers_size_=boxplots_outlier_markers_size,
                          df_data_=total_df[(total_df['Metric'] == 'RW') & (total_df['Train/Test'] == 'Train')],
                          x_='Metric', y_='DataValues', hue_='Model', ylabel_='Mean Reward', ylabelpad_=7,
                          w_legend=False, w_two_levels_x_labels=w_two_levels_x_labels_instead_of_legend, w_plot_title_=False, ax_=axes[1, 1],
                          ticks_labelsize_for_two_levels_x_labels_=ticks_labelsize_for_two_levels_x_labels,
                          xtick_major_pad_for_two_levels_x_labels_=xtick_major_pad_for_two_levels_x_labels,
                          factor_xticks_first_level_pad_single_plot_=factor_xticks_first_level_pad_single_plot,
                          factor_xticks_first_level_pad_multiple_plots_=factor_xticks_first_level_pad_multiple_plots)

        # set the spacing between subplots
        f.subplots_adjust(left=0.052,
                          bottom=0.097,
                          right=0.989,
                          top=0.978,
                          wspace=0.126,
                          hspace=0.332)
        f.suptitle('', fontsize=boxplots_font_size)
        #f.tight_layout()
        plt.savefig('./RA_RAD_ANMs_RW_train.png')
        plt.clf()
    #############################################################################################################

