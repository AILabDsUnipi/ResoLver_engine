"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

import random
import os
import sys

import numpy as np

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from sum_tree import SumTree as ST
from PER_config import per_config

class Memory(object):

    def __init__(self, capacity, DGN_m, exploration_episodes):
        self.capacity = capacity
        self.memory = ST(self.capacity)
        self.pr_scale = per_config['per_alpha']
        self.max_pr = 0
        self.DGN_model = DGN_m
        self.beta = per_config['starting_beta']
        self.e = per_config['per_epsilon']
        self.beta_step = (per_config['beta_max'] - per_config['starting_beta']) / \
                         (exploration_episodes * per_config['per_beta_step_decay_rate'])
        self.beta_max = per_config['beta_max']

    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, samples, errors):
        p = self.get_priority(errors)

        self_max = p.clip(min=self.max_pr)
        for i in range(len(samples)):
            self.memory.add(self_max[i], samples[i])

    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        num_segments = self.memory.total() / n

        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            sample_batch.append(data)
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_indices, errors):
        for i in range(len(batch_indices)):
            p = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], p)

    def find_targets_per(self,
                         batch=None,
                         valid_max_target_q_values=None,
                         n_agent=None,
                         train=True,
                         current_states_predictions=None,
                         q_values=None,
                         current_states=None,
                         GAMMA=None,
                         different_target_q_values_mask=None,
                         default_action_q_value_mask_=None,
                         np_mask_determ_dir_to_wp_after_maneuv_b=None,
                         np_mask_climb_descend_dir_to_wp_determ_with_confl_loss_b=None,
                         not_use_max_next_q_mask_=None,
                         next_flight_phases_b=None):
        if not train:

            q_values = \
                self.DGN_model.update_q_values(len(batch),
                                                np.asarray([batch[i][4] for i in range(len(batch))]), #dones
                                                n_agent,
                                                np.asarray([batch[i][6] for i in range(len(batch))]), #active_flights_m
                                                [batch[i][9] for i in range(len(batch))], #fls_with_loss_of_separation_m
                                                [batch[i][11] for i in range(len(batch))], #next_fls_with_loss_of_separation_m
                                                [batch[i][10] for i in range(len(batch))], #fls_with_conflicts_m
                                                [batch[i][12] for i in range(len(batch))], #next_fls_with_conflicts
                                                [batch[i][13] for i in range(len(batch))], #history_loss_confl_m
                                                current_states_predictions,
                                                np.asarray([batch[i][1] for i in range(len(batch))]), #actions
                                                np.asarray([batch[i][3] for i in range(len(batch))]), #last_reward
                                                [batch[i][15] for i in range(len(batch))], #reward_hist
                                                [batch[i][14] for i in range(len(batch))], #next_active_flights_m,
                                                GAMMA,
                                                valid_max_target_q_values,
                                                [batch[i][16] for i in range(len(batch))], #dur_of_acts
                                                [batch[i][19] for i in range(len(batch))], #data_needed_for_delayed_update__
                                                [batch[i][17] for i in range(len(batch))], #next_timestamp__
                                                different_target_q_values_mask,
                                                default_action_q_value_mask_,
                                                np_mask_determ_dir_to_wp_after_maneuv_b,
                                                np_mask_climb_descend_dir_to_wp_determ_with_confl_loss_b,
                                                not_use_max_next_q_mask_,
                                                next_flight_phases_b)

        eps = 1e-5
        sum_of_diff_over_all_actions_of_an_agent = np.sum(q_values - current_states_predictions, axis=2)
        errors = np.abs(np.sum(sum_of_diff_over_all_actions_of_an_agent, axis=1)/(np.count_nonzero(sum_of_diff_over_all_actions_of_an_agent) + eps))

        return [current_states, q_values, errors]

    def observe(self,
                sample,
                valid_max_target_q_values,
                n_agent,
                GAMMA,
                current_states_predictions,
                different_target_q_values_mask,
                default_action_q_value_mask_,
                np_mask_determ_dir_to_wp_after_maneuv_b,
                np_mask_climb_descend_dir_to_wp_determ_with_confl_loss_b,
                not_use_max_next_q_mask_,
                next_flight_phases_b):

        _, _, errors = self.find_targets_per(sample,
                                             valid_max_target_q_values,
                                             n_agent,
                                             train=False,
                                             current_states_predictions=current_states_predictions,
                                             GAMMA=GAMMA,
                                             different_target_q_values_mask=different_target_q_values_mask,
                                             default_action_q_value_mask_=default_action_q_value_mask_,
                                             np_mask_determ_dir_to_wp_after_maneuv_b=np_mask_determ_dir_to_wp_after_maneuv_b,
                                             np_mask_climb_descend_dir_to_wp_determ_with_confl_loss_b=np_mask_climb_descend_dir_to_wp_determ_with_confl_loss_b,
                                             not_use_max_next_q_mask_=not_use_max_next_q_mask_,
                                             next_flight_phases_b=next_flight_phases_b)

        self.remember(sample, errors)

    def calculate_sample_weights(self, batch_priorities, batch_size, current_states_predictions, q_values):
        _, _, errors = self.find_targets_per(current_states_predictions=current_states_predictions, q_values=q_values)
        batch_priorities_np = np.asarray(batch_priorities)
        normalized_batch_priorities_np = batch_priorities_np / sum(batch_priorities_np)
        importance_sampling_weights_np = (batch_size * normalized_batch_priorities_np) ** (-1 * self.beta)
        normalized_importance_sampling_weights_np = importance_sampling_weights_np / max(importance_sampling_weights_np)
        sample_weights_np = errors * normalized_importance_sampling_weights_np

        return sample_weights_np, errors

    def update_beta(self):
        self.beta += self.beta_step
        if self.beta > self.beta_max:
            self.beta = self.beta_max



