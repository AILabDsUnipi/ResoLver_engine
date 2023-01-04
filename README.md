# CDR_DGN

This repository constitutes the implementation (code and dataset) of our paper: *Deep Reinforcement Learning in service of Air Traffic Controllers to resolve tactical conflicts*

## Contents

* 1\. [Introduction](#introduction)
* 2\. [Code dependencies](#code_dependencies)
* 3\. [Real World Dataset](#real_world_dataset)
* 4\. [Trained models](#trained_models)
* 5\. [Use of code](#use_of_code)
	* 5.1\. [One scenario testing](#one_scenario_testing)
 	* 5.2\. [Multiple scenarios testing](#multiple_scenarios_testing)
		* 5.2.1\. [Statistics table](##statistics_table)
		* 5.2.2\. [Boxplots](#boxplots)
		* 5.2.3\. [Extract details of scenarios](#extract_details_of_scenarios)
	* 5.3\. [Training](#training)
		* 5.3.1\. [Train with only 1 scenario](#extract_details_of_scenarios)
  		* 5.3.2\. [Train with more than 1 scenario](#train_with_more_than_1_scenario)
  		* 5.3.3\. [Sequential training](#sequential_training)
		* 5.3.4\. [Plot learning curves](#plot_learning_curves)
* 6\. [Citation](#citation)

## Introduction

This repository consists of: 

* code of a virtual environment which simulates the air traffic of a predefined area given a Real World Dataset, detects conficts among the flights, and controls the flights according to the resolution actions provided in a specific form,

* code of [DGN](https://github.com/PKU-RL/DGN) algorithm enhanced to receive as input edges-features and provides resolution actions to conflicting flights. In the original DGN, the agents form a graph where they are represented as the vertices, and the edges are only utilized to create a matrix of adjacency indicators, e.g., 1 in the *i,j* position of the matrix if agents *i* and *j* are neighbors, otherwise 0. In our impementation, the enhanced DGN is fed with the vertices-features (current speed of flights, altitude, etc) as well as the edges-feature (interdependencies between two conflicting flight, like their current distance, their distance to their Closest Point of Approach, etc). In addition, there are further modifications such as the use of a *Prioritized Replay Buffer* instead of a conventional Replay Buffer, and replacing the reward with a *discounted reward* based on *history of rewards*. For more details, please read our paper.

* a Real World Dataset which is comprised of: *(a)* flight trajectories, *(b)* flight plans, and *(c)* radar tracks. This dataset (along with the trained models) allows the replication of the results of our paper. Note that for privacy reasons, we have altered the real dates.

* Trained models.

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

## Trained models

We provide trained parameters for all models reported in our paper. Specifically:

- model All36: *./trained_model/All36/gdn_1.h5*
- model 6Seq6: *./trained_model/6Seq6/gdn_1.h5*
- model 4Seq6: *./trained_model/4Seq6/gdn_1.h5*
- model All36-DGN (ablation): *./trained_model/All36_DGN/gdn_1.h5*
- model 6Seq6-DGN (ablation): *./trained_model/6Seq6_DGN/gdn_1.h5*
- model 4Seq6-DGN (ablation): *./trained_model/4Seq6_DGN/gdn_1.h5*


