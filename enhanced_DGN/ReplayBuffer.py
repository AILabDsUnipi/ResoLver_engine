"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

from collections import deque
import random
import copy
import numpy as np

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from DGN_config import dgn_config

class ReplayBuffer(object):

    def __init__(self):

        self.buffer_size = dgn_config['capacity']
        self.num_experiences = 0
        self.buffer = deque()
        self.temp_episode_buffer = list()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add_in_temp_episode_buff(self,
                                 state,
                                 action,
                                 new_state,
                                 reward,
                                 done,
                                 adjacency,
                                 active_flights_mask,
                                 norm_edges_feats,
                                 next_norm_edges_feats_based_on_previous_adj_matrix,
                                 fls_with_loss_of_separation,
                                 fls_with_conflicts,
                                 next_fls_with_loss_of_separation,
                                 next_fls_with_conflicts,
                                 history_loss_confl,
                                 next_active_flights_mask,
                                 reward_history,
                                 durations_of_actions,
                                 next_timestamp,
                                 timestamp,
                                 unfixed_acts,
                                 np_mask_climb_descend_res_fplan_with_confl_loss,
                                 np_mask_res_fplan_after_maneuv,
                                 duration_dir_to_wp_ch_FL_res_fplan_,
                                 next_executing_FL_change,
                                 next_executing_direct_to,
                                 next_flight_phases,
                                 executing_FL_change,
                                 executing_direct_to,
                                 next_available_wps,
                                 next_executing_resume_to_fplan):

        # Store in temporal episode buffer which might be updated at some subsequent timesteps due to the duration of the actions
        # Data needed for delayed update are the following:
        # "new_state", "adjacency", "next_norm_edges_feats_based_on_previous_adj_matrix", "next_fls_with_loss_of_separation",
        # "next_fls_with_conflicts", "next_active_flights_mask"
        data_needed_for_delayed_update = [None for _ in range(state.shape[0])]

        experience = (state.copy(),
                      action.copy(),
                      new_state.copy(),
                      reward.copy(),
                      done,
                      adjacency.copy(),
                      active_flights_mask.copy(),
                      norm_edges_feats.copy(),
                      next_norm_edges_feats_based_on_previous_adj_matrix.copy(),
                      fls_with_loss_of_separation.copy(),
                      fls_with_conflicts.copy(),
                      next_fls_with_loss_of_separation.copy(),
                      next_fls_with_conflicts.copy(),
                      history_loss_confl.copy(),
                      next_active_flights_mask.copy(),
                      reward_history.copy(),
                      copy.deepcopy(durations_of_actions),
                      next_timestamp,
                      timestamp,
                      data_needed_for_delayed_update.copy(),
                      unfixed_acts.copy(),
                      next_executing_direct_to.copy(),
                      next_executing_FL_change.copy(),
                      np_mask_res_fplan_after_maneuv.copy(),
                      np_mask_climb_descend_res_fplan_with_confl_loss.copy(),
                      next_flight_phases.copy(),
                      executing_FL_change.copy(),
                      executing_direct_to.copy(),
                      next_available_wps.copy(),
                      next_executing_resume_to_fplan.copy())

        self.temp_episode_buffer.append(experience)

        # Check if there are actions with duration and if any of these durations have just expired, in order to update the
        # corresponding sample based on the timestep
        timestep_duration = next_timestamp - timestamp
        for agent in range(len(durations_of_actions)):

            # If an action (not for actions 'continue', 'direct to (any) way point' and 'change FL') has duration and this duration has just expired,
            # then update the corresponding history reward
            # or if the duration of an action has not been reached but the flight has just become inactive or the episode has just
            # ended, then store the history of rewards for the executed maneuver until the current timestep.
            case_1 = (durations_of_actions[agent][1] != 0
                      and next_timestamp - durations_of_actions[agent][1] == durations_of_actions[agent][0]) \
                     or (durations_of_actions[agent][1] != 0 and durations_of_actions[agent][0] != np.inf and
                         next_timestamp - durations_of_actions[agent][1] < durations_of_actions[agent][0]
                         and (not next_active_flights_mask[agent] or done))

            # If a non-deterministic action 'direct to (any) way point' or 'change FL' has just finished (or the agent became inactive or
            # the episode has just ended), or the deterministic action 'resume to fplan' has just finished (or the agent became inactive or
            # the episode has just ended) given that 'np_mask_climb_descend_res_fplan_with_confl_loss' or 'np_mask_res_fplan_after_maneuv'
            # is True (in the case of 'np_mask_climb_descend_res_fplan_with_confl_loss' we also check if the phase has changed at the next state
            # and there is a conflict/loss), then store the history of rewards for the executed maneuver until the current timestep.
            case_2 = (durations_of_actions[agent][0] == np.inf and
                      (not (next_executing_FL_change[agent] or next_executing_direct_to[agent])
                       or done or not next_active_flights_mask[agent])) or \
                     (durations_of_actions[agent][0] == 0 and
                      ((np_mask_climb_descend_res_fplan_with_confl_loss[agent] and
                        (not next_executing_resume_to_fplan[agent] or
                         (not (next_flight_phases[agent] == 'climbing' or next_flight_phases[agent] == 'descending') and
                          (agent in next_fls_with_loss_of_separation or agent in next_fls_with_conflicts)) or done or
                         not next_active_flights_mask[agent]))
                       or (np_mask_res_fplan_after_maneuv[agent] and
                           (not next_executing_resume_to_fplan[agent] or
                            (next_executing_resume_to_fplan[agent] and (agent in next_fls_with_loss_of_separation or agent in next_fls_with_conflicts) and
                             not (next_flight_phases[agent] == 'climbing' or next_flight_phases[agent] == 'descending'))
                           or done or not next_active_flights_mask[agent]))))

            if case_1 or case_2:

                if case_1:
                    # If the flight became inactive or the episode was ended before the completeness of the action
                    # (i.e. the duration has not expired), then change the duration accordingly
                    duration = durations_of_actions[agent][0] if next_timestamp - durations_of_actions[agent][1] == \
                                                                 durations_of_actions[agent][0] \
                                                                 else next_timestamp - durations_of_actions[agent][1]
                elif case_2:
                    duration = duration_dir_to_wp_ch_FL_res_fplan_[agent]

                for timestep in range(int(duration/timestep_duration)):
                    # We use "timestep+1" because timestep starts from 0 but the last element of the list (and the second
                    # element before the end of the list, the third etc) can be gotten by -1 (-2, -3, etc, respectively) index.
                    # Also, the index 15 is the stored reward history which should be updated.
                    self.temp_episode_buffer[-(timestep+1)][15][agent] = reward_history[agent][-(timestep+1):].copy()
                    self.temp_episode_buffer[-(timestep+1)][19][agent] = (new_state.copy(),
                                                                          adjacency.copy(),
                                                                          next_norm_edges_feats_based_on_previous_adj_matrix.copy(),
                                                                          next_fls_with_loss_of_separation.copy(),
                                                                          next_fls_with_conflicts.copy(),
                                                                          next_active_flights_mask.copy(),
                                                                          done,
                                                                          next_flight_phases.copy(),
                                                                          next_available_wps.copy(),
                                                                          duration_dir_to_wp_ch_FL_res_fplan_.copy(),
                                                                          next_timestamp) #Index 19 refers to "data_needed_for_delayed_update"

    def store_episode_samples(self):
        for sample in self.temp_episode_buffer:
            if self.num_experiences < self.buffer_size:
                self.buffer.append(tuple(list(sample).copy()))
                self.num_experiences += 1
            else:
                self.buffer.popleft()
                self.buffer.append(tuple(list(sample).copy()))

        self.reset_temp_episode_buff()

    def get_temp_episode_samples(self):
        list_of_samples_to_get = []
        for sample in self.temp_episode_buffer:
            list_of_samples_to_get.append(tuple(list(sample).copy()))

        return list_of_samples_to_get

    def reset_temp_episode_buff(self):
        self.temp_episode_buffer = list()

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
