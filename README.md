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


