# CDR_DGN

This repository constitutes the implementation (code and dataset) of our paper:                                                                                        
*Deep Reinforcement Learning in service of Air Traffic Controllers to resolve tactical conflicts*

This repository is publicly available under a *GPL-3.0 licence* found in the *LICENSE.md* file in the root directory of this source tree.                              

All data contained in this publicly available repository is under a *CC BY-NC-ND 4.0 licence* found in the *data_LICENSE.md* file in the root directory of this source tree. As data we consider the followings:
- all files included in the folders: *./env/SectorGeom*, *./env/fplans*, *./env/training_data*, *./env/environment/bins_median_dict.pickle*, 
  *./env/environment/dangles_dspeeds_dict.txt*, *./env/environment/mbr_spain_2.csv*, *./env/environment/sectors_over_spain.csv*, 
  *./env/environment/dependencies/bins_median_dict.pickle*, *./env/environment/states_out_0.csv*, *./env/environment/states_out_1.csv*, 
  *./env/environment/exit_wp_0.csv*, and *./env/environment/exit_wp_1.csv* 
- all files included in the folders: *./trained_model*, and *./selected_scenarios_files*                                                                               

NOTE: as *./* we consider the root directory of this repository.

## Contents

* 1\. [Introduction](#introduction)
* 2\. [Code dependencies](#code-dependencies)
* 3\. [Real World Dataset](#real-world-dataset)
* 4\. [Trained models](#trained-models)
* 5\. [Use of code](#use-of-code)
	* 5.1\. [One scenario testing](#one-scenario-testing)
 	* 5.2\. [Multiple scenarios testing](#multiple-scenarios-testing)
		* 5.2.1\. [Statistics table](#statistics-table)
		* 5.2.2\. [Boxplots](#boxplots)
		* 5.2.3\. [Extract details of scenarios](#extract-details-of-scenarios)
	* 5.3\. [Training](#training)
		* 5.3.1\. [Train with only 1 scenario](#train-with-only-1-scenario)
  		* 5.3.2\. [Train with more than 1 scenario](#train-with-more-than-1-scenario)
  		* 5.3.3\. [Sequential training](#sequential-training)
		* 5.3.4\. [Plot learning curves](#plot-learning-curves)
* 6\. [Citation](#citation)

## Introduction

This repository consists of: 

* code of a virtual environment which simulates the air traffic of a predefined area given a Real World Dataset, detects conficts among the flights, and controls the flights according to the resolution actions provided in a specific form,

* code of [DGN](https://github.com/PKU-RL/DGN) algorithm enhanced to receive as input edges-features and provides resolution actions to conflicting flights. In the original DGN, the agents form a graph where they are represented as the vertices, and the edges are only utilized to create a matrix of adjacency indicators, e.g., 1 in the *i,j* position of the matrix if agents *i* and *j* are neighbors, otherwise 0. In our impementation, the enhanced DGN is fed with the vertices-features (current speed of flights, altitude, etc) as well as the edges-feature (interdependencies between two conflicting flight, like their current distance, their distance to their Closest Point of Approach, etc). In addition, there are further modifications such as the use of a *Prioritized Replay Buffer* instead of a conventional Replay Buffer, and replacing the reward with a *discounted reward* based on *history of rewards*. For more details, please read our paper.

* a Real World Dataset which is comprised of: *(a)* flight trajectories, *(b)* flight plans, and *(c)* radar tracks. This dataset (along with the trained models) allows the replication of the results of our paper. Note that for privacy reasons, we have altered the real dates.

* Trained models.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

## Code dependencies

To run the code (for testing or training), you need the followings:

1. Ubuntu 18.04, or 20.04, or 22.04,
2. Conda environment with *python 3.6*. You can execute the commands below to install conda and create the required conda virtual environment:
   - ```mkdir tmp```
   - ```cd tmp```
   - ```curl https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh --output anaconda.sh```
   - ```sha256sum anaconda.sh```
   - ```bash anaconda.sh```
   - ```source ~/.bashrc```
   - ```conda install python=3.6```
   - ```conda create --name conda_env python=3.6```
   - ```conda activate conda_env```
3. After installing and activating the conda environment, you should install the requirements using the following commands:
   - ```conda install pandas=1.1.5```
   - ```conda install numba=0.50```
   - ```conda install pyproj=2.6.1.post1```
   - ```conda install cython=0.29.24```
   - ```pip install geo-py==0.4```
   - ```conda install shapely=1.7.1```
   - ```conda install matplotlib=3.1.1```
   - ```conda install scipy=1.5```
   - ```conda install pymongo=3.12.0```
   - ```conda install tensorflow-gpu=1.14```
   - ```pip install Keras==2.1.2```
   - ```conda install h5py=2.10.0```
   - ```conda install numpy==1.19.2```
   - ```conda install psutil==5.8.0```

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

## Real World Dataset

The provided dataset is about flights operating in the airspace of Spain. 
There are 25 scenarios in total: 
* 24 scenarios which we have used for testing. 
* 1 for training. Note that we do not provide the entire training dataset used to train the models.

You can find the IDs of the scenarios in the file: <br />
*./selected_scenarios_files/selected_scenarios_for_testing.txt* <br />
where the first one has been used for training and the rest (24) for testing.

Additionally, you can find the details of these 25 scenarios in the file: <br />
*./selected_scenarios_files/selected_scenarios_details.csv* <br />
which are the same as those reported in our paper. 

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

## Trained models

We provide trained parameters for all models reported in our paper. Specifically:

- model All36: *./trained_model/All36/gdn_1.h5*
- model 6Seq6: *./trained_model/6Seq6/gdn_1.h5*
- model 4Seq6: *./trained_model/4Seq6/gdn_1.h5*
- model All36-DGN (ablation): *./trained_model/All36_DGN/gdn_1.h5*
- model 6Seq6-DGN (ablation): *./trained_model/6Seq6_DGN/gdn_1.h5*
- model 4Seq6-DGN (ablation): *./trained_model/4Seq6_DGN/gdn_1.h5*

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

## Use of code

### One scenario testing

In order to a perform a test using only 1 scenario (whichever of the provided) and the model of your preference, you should execute the following command:

```python runexp.py --DGN_model_path=<model_path> --evaluation=True --with_RoC_term_reward=True --scenario=<scenario_ID> ```
<br />
where ```<model_path>``` should be replaced by one of the paths referred above (e.g, *./trained_model/All36/gdn_1.h5*, which is the best model!),
and ```<scenario_ID>``` should be replaced by one of the scenario IDs reported in the file: <br />
*./selected_scenarios_files/selected_scenarios_for_testing.txt* . <br /> 
If the selected model path corresponds to a model of the ablation study (All36-DGN, 6Seq6-DGN, 4Seq6-DGN) you should also use the argument ```--conc_observations_edges=True```.

When run is complete, the following files should be created in current directory:

- *episodes_log.txt* which will include information about the *mean reward*, *number of alerts*, *number of ATC instructions*, and *total additional NM*.
- *./logs/solved_confl_and_further_info.txt* which will include information about the *number of conflicts*, *number of LoSs*, *number of conflicts in groups solved*, 
  *total conflict resolution duration*. Note that in our paper, the *percentage of Conflicts Resolved* is computed based on the *number of conflicts in groups solved*   since *number of conflicts solved* is misleading (each conflict is considered unresolved if there is another conflict for the same pair of conflicting flights in a
  subsequent timestamp, while we consider a conflict unresolved only if there is a LoS for the same pair of conflicting flights in a subsequent timestamp).
- *./logs/resolution_actions_episode_1* which will include details about the suggested resolution actions.
- other files in the *./logs* folder as well as the directories *./log* and *./heatmaps_by_agent* with more details.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

### Multiple scenarios testing

To run tests with more than one scenario, you should execute the following command:

- ```python run_multiple_tests.py --DGN_model_path=<model_path>```

where by default:

- *micro* average results are calculated. You have the following choices for calculating *macro* average results:
  - use the argument ```--with_mean_and_std=True``` to compute the *mean* and *std* of the macro average results.
  - use the argument ```--with_median_instead_of_mean=True``` along with ```--with_mean_and_std=True``` to compute the *median* and *std* of the macro average results.
  - use the argument ```--with_iqr_instead_of_std``` along with ```--with_mean_and_std=True``` and ```--with_median_instead_of_mean=True``` to compute the *median* 
    and *iqr* (Interquantile Range) of the macro average results.

- the file *./selected_scenarios_files/selected_scenarios_for_testing.txt* is loaded to determine the scenarios to be used for testing. <br />
  Note that the argument ```--file_path_selected_scenarios_for_testing``` controls the path of this file, and       
  ```--num_of_the_last_scenarios_only_for_testing``` controls the number of testing scenarios. This file should include at least 1 training scenario at the top and
  then the testing scenarios should be listed.

- the file *./selected_scenarios_files/selected_scenarios_of_training_for_testing.txt* is loaded to determine the scenarios used for training and which are going to be
  used for testing as well (since the results are separated into train-and-test and test-only). <br />
  Note that the argument ```--file_path_selected_scenarios_of_training_for_testing``` controls the path of this file. Also, note that this file should include at least 
  1 scenario which should also be included in the file *./selected_scenarios_files/selected_scenarios_for_testing.txt*.

- the file *./selected_scenarios_files/selected_scenarios_details.csv* is loaded to determine the details (like the *number of conflicts*, *number of alerts*, etc) of 
  the scenarios used for testing (specifically, the details of the scenarios without applying any resolution action in order to compared them with the
  corresponding details after applying the resolution actions proposed by the selected model).
  
NOTE: If you want to run the test using a model of the ablation study, you should use the argument ```--conc_observations_edges=True``` . 

When run is complete, a folder named *testing* should be created in the current directory, which should include:

- A folder for each of the testing scenarios named *scenario=<scenario_ID>*, which will contain the corresponding files as described in [One scenario testing](#one-scenario-testing).

- A file named *testing_dataframe.csv* which includes the statistics (like the *Increase/Decrease of alerts*, *Increase/Decrease of LoS*, etc) reported in our paper.
  It should be noted again that *percentage of Conflicts Resolved* reported in the paper is computed based on the *Number of conflicts in groups solved*, 
  as well as *Resolution Action Duration* corresponds to *Conflict (in groups) resolution duration*.
  
<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>
  
#### Statistics table
In order to summarize the statistics of the different models (after running *run_multiple_tests.py* for the 6 models), you can create a table of statistics by running
the file *construct_statistics_table.py* located in the folder *results_utils* using the following command:

```python construct_statistics_table.py```

NOTE: Before running the above command, you should change the paths according to the location of each file *testing_dataframe.csv* (one for each model). Specifically, you should change the paths in lines 15-20 of *construct_statistics_table.py* file.

When run is complete, a file named *statistics_table.xlsx* should be created in the current directory. 

If the statistics to be summarized include *std* or *iqr*, you can use the argument ```--with_mean_median_and_std_iqr=True``` to adjust the length of the cells of the .xlsx file.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

#### Boxplots
To create boxplots for macro average results (for all models), you should first save the corresponding dataframes. For this, you should run *run_multiple_tests.py* using the argument ```--save_data_for_boxplots=True``` along with ```--with_mean_and_std=True``` (and obviously with   
```--DGN_model_path=<model_path>```). When run is complete, the following files should be created:

- *./testing_scenarios/df_for_boxplots_1.pkl*
- *./testing_scenarios/df_for_boxplots_ATC_instructions.pkl*
- *./testing_scenarios/df_for_boxplots_conflict_in_groups_resolution_duration.pkl*
- *./testing_scenarios/df_for_boxplots_add_NMs.pkl*
- *./testing_scenarios/df_for_boxplots_mean_reward.pkl*

After performing this step for each model and storing the corresponding files, you should change the paths in lines 113-142 of the file *construct_statistics_table.py* 
(located in the folder *results_utils*) according to the locations of the stored files (30 file paths = 5 files X 6 models). Finally, you should execute the command below:

```python create_boxplots.py```

When run is complete, the following files should be created in the current directory:

- *CR_IncrDecrConflicts_IncrDecrAlerts_test.png*
- *RA_RAD_ANMs_RW_test.png*

If you want to create the corresponding boxplots for the training scenarios (in case that you have train a model with other training scenarios than the one provided)
you should use the argument ```--train=True```.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

### Extract details of scenarios

To reproduce the results of the file *selected_scenarios_details.csv*, you should run the following command:

```python extract_scenarios_details.py```

where by default the file *./selected_scenarios_files/selected_scenarios_for_testing.txt* is loaded to determine the scenarios for which the details will be extracted.

After run is complete, the file *scenarios_details.csv* should be created in the current directory.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

### Training

The hyperparameters of the enhanced DGN can be controlled through the file *./enhanced_DGN/DGN_config.py* .

#### Train with only 1 scenario

To train a model using samples from a specific scenario, you should execute the following command:

```python runexp.py --scenario=<scenario_ID>```

Furthermore, you can control the following hyperparameters using only arguments (NOT through the file *./enhanced_DGN/DGN_config.py*):

- batch size, argument: ```--batch_size=<integer_number>```
- learning rate, argument: ```--LRA=<float_number>```
- total number of training episodes (exploration + exploitation), argument: ```--train_episodes=<integer_number>```
- total number of exploration episodes (it should be lower than the total number of training episodes),                                                                 
  argument: ```--exploration_episodes=<integer_number>```
- specify if want to use priorized replay buffer, argument: ```--prioritized_replay_buffer=<True_or_False>```
- specify if want to use the *Rate of Closure* in the reward function, argument: ```--with_RoC_term_reward=<True_or_False>```

After training is finished, the trained model will be stored in a file named *gdn.h5* in the current directory. Also, a file named *episodes_log.txt* will be created in the same directory, which will contain details about the progress of training and can be used to plot the learning curves.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

#### Train with more than 1 scenario

To train a model using samples from many scenarios, you should run the *runexp.py* file with the following arguments:

- ```--multi_scenario_training=True``` instead of ```--scenario=<scenario_ID>```. 
- ```--selected_scenarios_path=<path_to_the_file>```, where the *file* should be a .txt file containing the IDs of the selected scenarios to be used (like the 
  *selected_scenarios_for_testing.txt*) and named *selected_scenarios.txt*.

The way to control the hyperparameters is the same as in the single-scenario training.

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

#### Sequential training

You can train a model in a sequential fashion, as we did in the cases of 4Seq6 and 6Seq6 models (please read the paper for details). When a model is already trained with a batch of scenarios, you can further train it using another batch (or even the same) by running the *runexp.py* file with the following arguments:

- ```--continue_train=True```
- ```--DGN_model_path=<path_to_trained_model>``` . NOTE: The name of the trained model should be different from *gdn.h5* (such as *gdn_1.h5*).

#### Plot learning curves

After training a model you can plot the learning curves by performing the following steps:

- create a folder named *training* inside the folder *results_utils*.
- create a folder named *1st exp* inside the folder *training*.
- move the generated file *episodes_log.txt* in the folder *1st exp*.
- rename the *episodes_log.txt* file to *episodes_log_1.txt*.
- run the command ```python plot_results.py```.

NOTE: The file *plot_results.py* is located in the folder *results_utils*. To run the above command you need to create a different conda environment with python version >= 3.7, numpy version >= 1.21.5, and matplotlib version >= 3.5.1 .

<p align="right">(<a href="#cdr_dgn">back to top</a>)</p>

## Citation
``` . ```
