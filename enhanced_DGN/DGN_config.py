"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

dgn_config = {
    'num_eval_episodes': 1,
    'capacity': 200000,
    'TAU': 0.01, # In the paper of DGN this parameter is denoted as 'beta'
    'alpha': 0.6,
    'min_alpha': 0.01,
    'GAMMA': 0.96,
    'episode_before_train': 200,
    'train_step_per_episode': 80,
    'neighbors_observed': 3,
    'n_heads': 8,
    'first_MLP_layer_neurons': 512,
    'second_MLP_layer_neurons': 128,
    'out_dim': 128,
}