"""
AILabDsUnipi/CDR_DGN Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

#Code references:
# https://github.com/PKU-RL/DGN
# https://github.com/wingsweihua/colight

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["KMP_WARNINGS"] = "FALSE"

import random
import numpy as np
from numpy.polynomial.polynomial import polyval

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import backend as K
from keras.optimizers import Adam
import keras.models
from keras.layers import Dense, Input, Lambda, Flatten, merge
from keras.layers import Reshape, Concatenate
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils, to_categorical
from keras.engine.topology import Layer
import keras.backend.tensorflow_backend as KTF

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from DGN_config import dgn_config

class Slice(Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.size[2], input_shape[3]

    def get_config(self):

        config = {
            'begin': self.begin,
            'size': self.size,
                }

        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return K.tf.slice(inputs, self.begin, self.size)

class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        #[batch,agent,dim]->[batch,1,agent,dim]
        #[batch,1,agent,dim]->[batch,agent,agent,dim]

        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DGN_model:
    def __init__(self,
                 n_agent,
                 observ_dim,
                 edges_dim,
                 n_actions,
                 evaluation,
                 DGN_model_path,
                 LRA,
                 continue_train,
                 conc_observations_edges):

        self.n_neighbors = dgn_config['neighbors_observed'] + 1
        self.n_agent = n_agent
        self.observ_dim = observ_dim
        self.edges_dim = edges_dim
        self.first_MLP_layer_neurons = dgn_config['first_MLP_layer_neurons']
        self.second_MLP_layer_neurons = dgn_config['second_MLP_layer_neurons']
        self.out_dim = dgn_config['out_dim']
        self.n_heads = dgn_config['n_heads']
        self.n_actions = n_actions
        self.evaluation = evaluation
        self.DGN_model_path = DGN_model_path
        self.LRA = LRA
        self.TAU = dgn_config['TAU']
        self.dv = int(self.out_dim/self.n_heads)
        self.continue_train = continue_train
        self.conc_observations_edges = conc_observations_edges #If True, each agent observations includes its edges features

        ######build the model#########
        # Graph and session definition is needed when DGN is used side-by-side with another model,
        # e.g., in case that we predict the duration of each action with another model.
        # Even if we do not use another model here, we have created the code in this way to ease a future work.
        self.graph = K.tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            KTF.set_session(self.session)
            with self.session.as_default():

                In = list()
                In.append(Input(shape=(self.n_agent, self.observ_dim)))  # Feature vector of each agent. shape: [batch, agent, features]
                In.append(Input(shape=(self.n_agent, self.n_neighbors, self.n_agent)))  # Adjacency matrix of each agent. shape: [batch, agent, neighbors, agent]
                In.append(Input(shape=(self.n_agent, self.n_neighbors, self.edges_dim))) # Edges feature vector of each agent. shape: [batch, agent, neighbors, edges_features]

                In.append(Input(shape=[1])) #Sample weights for prioritized buffer. shape: [batch, 1]

                if conc_observations_edges:
                    feature = self.MLP_conc_obs_and_edges(self.n_agent, self.n_neighbors, first_MLP_layer_neurons, second_MLP_layer_neurons, Inp_ag=In[0], Inp_edg=In[2]) #shape: [batch, agent, 256]
                    edges_features = None

                else:
                    feature = self.MLP(self.first_MLP_layer_neurons, self.second_MLP_layer_neurons, In_0=In[0]) #shape: [batch, agent, 128]
                    edges_features = self.MLP_edges(self.first_MLP_layer_neurons, self.second_MLP_layer_neurons, In_0=In[2]) #shape: [batch, agent, neighbors, 128]

                if evaluation is False:
                    # shape: [batch, agent, 128]
                    relation1 = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges,
                                                        ev=self.evaluation, Inp_ag=feature, Inp_edg=edges_features, Inp_adj_matrix=In[1])
                else:
                    # shape: [batch, agent, 128], [batch, agent, 1, n_heads, neighbor]
                    relation1, att1 = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges,
                                                              ev=self.evaluation, Inp_ag=feature, Inp_edg=edges_features, Inp_adj_matrix=In[1])

                # shape: [batch, agent, neighbors, 128]
                edges_features1 = edges_features

                if evaluation is False:
                    # shape: [batch, agent, 128]
                    relation2 = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges,
                                                        ev=self.evaluation, Inp_ag=relation1, Inp_edg=edges_features1, Inp_adj_matrix=In[1])
                else:
                    # shape: [batch, agent, 128], [batch, agent, 1, n_heads, neighbor]
                    relation2, att2 = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges,
                                                              ev=self.evaluation, Inp_ag=relation1, Inp_edg=edges_features1, Inp_adj_matrix=In[1])
                    all_att = Concatenate(axis=2)([att1, att2]) #shape: [batch, agent, 2, n_heads, neighbor]

                conc_q_net_input = Concatenate(axis=2)([feature, relation1, relation2]) #shape [batch, agent, 128*3(=384)] or [batch, agent, 256+(128*2)=512] in case of conc_observations_edges
                q_net_out = self.Q_Net(self.n_actions, Inp=conc_q_net_input) #shape [batch, agent, n_action]

                def MSE_loss(y_true, y_pred):

                    eps = 1e-5

                    diff = y_true - y_pred
                    squared_diff = K.sum(K.square(diff), axis=2)  # Sum over actions because only one of the difference of the predicted with true q-values will not be zero.
                    mean_over_agents_for_non_zero_values = K.sum(squared_diff, axis=1) / (K.cast(K.tf.count_nonzero(squared_diff, axis=1), K.tf.float32) + eps)
                    weights = K.sum(In[3], axis=1) #This is a dummy sum just to succeed the shape of weights to be [batch]. This is necessary because the 1-D numpy array is automatically converted to a 2-D tensor. On the other hand, a numpy array with shape () is not recognized as a 1-D tensor.
                    weights_mean_over_agents_for_non_zero_values = mean_over_agents_for_non_zero_values * weights
                    mean_over_batch_for_non_zero_values = K.sum(weights_mean_over_agents_for_non_zero_values, axis=0) / (K.cast(K.tf.count_nonzero(weights_mean_over_agents_for_non_zero_values, axis=0), K.tf.float32) + eps)

                    return mean_over_batch_for_non_zero_values

                if self.evaluation is False:
                    self.model = Model(input=In, output=q_net_out)
                    self.model.compile(optimizer=Adam(lr=self.LRA), loss=MSE_loss)
                    if self.continue_train:
                        temp_model = keras.models.load_model(self.DGN_model_path + '/gdn_1.h5',
                                                             custom_objects={'RepeatVector3D': RepeatVector3D,
                                                                             'Slice': Slice,
                                                                             'MSE_loss': MSE_loss})
                        trained_weights = temp_model.get_weights()
                        self.model.set_weights(trained_weights)
                        print('A pre-trained model is loaded and its training is about to be continued!')
                    else:
                        print('A new model is about to be trained!')

                    self.model.summary()

                else:
                    print('A pre-trained model is loaded!')
                    self.model = Model(input=In, output=[q_net_out, all_att])
                    temp_model = keras.models.load_model(self.DGN_model_path + '/gdn_1.h5',
                                                         custom_objects={'RepeatVector3D': RepeatVector3D,
                                                                         'Slice': Slice,
                                                                         'MSE_loss': MSE_loss})
                    trained_weights = temp_model.get_weights()
                    self.model.set_weights(trained_weights)

                ######build the target model#########
                if not evaluation:
                    In_t = list()
                    In_t.append(Input(shape=(self.n_agent, self.observ_dim)))  # Feature vector of each agent. shape: [batch, agent, features]
                    In_t.append(Input(shape=(self.n_agent, self.n_neighbors, self.n_agent)))  # Adjacency matrix of each agent. shape: [batch, agent, neighbors, agent]
                    In_t.append(Input(shape=(self.n_agent, self.n_neighbors, self.edges_dim)))  # Edges feature vector of each agent. shape: [batch, agent, neighbors, edges_features]

                    if conc_observations_edges:
                        feature_t = self.MLP_conc_obs_and_edges(self.n_agent, self.n_neighbors, first_MLP_layer_neurons, second_MLP_layer_neurons, Inp_ag=In_t[0], Inp_edg=In_t[2])  # shape: [batch, agent, 256]
                        edges_features_t = None
                    else:
                        feature_t = self.MLP(self.first_MLP_layer_neurons, self.second_MLP_layer_neurons, In_0=In_t[0])  # shape: [batch, agent, 128]
                        edges_features_t = self.MLP_edges(self.first_MLP_layer_neurons, self.second_MLP_layer_neurons, In_0=In_t[2])  # shape: [batch, agent, neighbors, 128]

                    relation1_t = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges, Inp_ag=feature_t, Inp_edg=edges_features_t, Inp_adj_matrix=In_t[1])  # shape: [batch, agent, 128]

                    # shape: [batch, agent, neighbors, 128]
                    edges_features1_t = edges_features_t

                    # shape: [batch, agent, 128]
                    relation2_t = self.MultiHeadsAttModel(self.n_agent, self.n_neighbors, self.out_dim, self.n_heads, self.dv, self.conc_observations_edges,
                                                          Inp_ag=relation1_t, Inp_edg=edges_features1_t, Inp_adj_matrix=In_t[1])
                    q_net_out_t = self.Q_Net(self.n_actions, Inp=Concatenate(axis=2)([feature_t, relation1_t, relation2_t]))  # shape [batch, agent, n_action]
                    self.model_t = Model(input=In_t, output=q_net_out_t)
                    if self.continue_train:
                        temp_model = keras.models.load_model(self.DGN_model_path + '/gdn_1.h5',
                                                             custom_objects={'RepeatVector3D': RepeatVector3D,
                                                                             'Slice': Slice,
                                                                             'MSE_loss': MSE_loss})
                        trained_weights = temp_model.get_weights()
                        self.model_t.set_weights(trained_weights)

    @staticmethod #It does not need to be the whole method static, just the arguments and the same is true for the rest of methods
    def MLP(first_MLP_layer_neurons, second_MLP_layer_neurons, In_0=None):
        #Input shape: [batch, agent, feature]
        #Output shape: #shape: [batch, agent, 128]

        h = Dense(first_MLP_layer_neurons, activation='relu', kernel_initializer='random_normal')(In_0) #shape: [batch, agent, 512]
        h_ = Dense(second_MLP_layer_neurons, activation='relu', kernel_initializer='random_normal')(h) #shape: [batch, agent, 128]

        return h_

    @staticmethod
    def MLP_edges(first_MLP_layer_neurons, second_MLP_layer_neurons, In_0=None):
        # Input shape: [batch, agent, neighbors, edges_features]
        # Output shape: #shape: [batch, agent, neighbors, 128]

        h = Dense(first_MLP_layer_neurons, activation='relu', kernel_initializer='random_normal')(In_0) #shape [batch, agent, neighbors, 512]
        h_ = Dense(second_MLP_layer_neurons, activation='relu', kernel_initializer='random_normal')(h) #shape [batch, agent, neighbors, 128]

        return h_

    @staticmethod
    def MLP_conc_obs_and_edges(n_ag, n_neibs, first_MLP_layer_neurons, second_MLP_layer_neurons, Inp_ag=None, Inp_edg=None):
        # Inp_ag shape: [batch, agent, feature]
        # Inp_edg shape: [batch, agent, neighbor, edges_features]

        Inp_edg_shape_for_conc = Inp_edg.get_shape().as_list()  # List=[batch, agent, neighbor, edges_features]

        # shape: [batch, agent, neighbor-1, edges_features]
        sliced_Inp_edge_neighbors_except_i = \
            Slice(begin=[0, 0, 1, 0], size=[-1, Inp_edg_shape_for_conc[1], Inp_edg_shape_for_conc[2]-1, Inp_edg_shape_for_conc[3]])(Inp_edg)

        # shape: [batch, agent, (neighbor-1)*edges_features]
        flat_edges = Reshape((n_ag, (n_neibs - 1) * Inp_edg_shape_for_conc[3]))(sliced_Inp_edge_neighbors_except_i)

        # shape: [batch, agent, dim=(features+((neighbor-1)*edges_features))]
        obs_edges_conc = Concatenate(axis=2)([Inp_ag, flat_edges])

        h = Dense(first_MLP_layer_neurons, activation='relu', kernel_initializer='random_normal')(obs_edges_conc)  # shape [batch, agent, neighbors, 512]
        h_ = Dense(second_MLP_layer_neurons*2, activation='relu', kernel_initializer='random_normal')(h)  # shape [batch, agent, neighbors, 128*2=256]

        return h_

    @staticmethod
    def MultiHeadsAttModel(n_ag, n_neibs, out_dim, n_heads, dv, conc_observations_edges,
                           ev=False, Inp_ag=None, Inp_adj_matrix=None, Inp_edg=None):

        # Inp_ag shape: [batch, agent, 128] or [batch, agent, 256] (in case of conc_observations_edges in the first convolution)
        # Inp_adj_matrix shape: [batch, agent, neighbor, agent]
        # Inp_edg shape: [batch, agent, neighbor, 128] or None (in case of conc_observations_edges)

        Inp_ag_shape = Inp_ag.get_shape().as_list()  # List=[batch, agent, 128 or 256]
        self_obs_agent_repr = Reshape((n_ag, 1, Inp_ag_shape[2]))(Inp_ag)  # shape: [batch, agent, 1, 128 or 256].

        # shape: [batch, agent, 128 or 256]->(reshape)[batch, 1, agent, 128 or 256]->(tile)[batch, agent, agent, 128 or 256]
        neighbor_obs_repr_all = RepeatVector3D(n_ag)(Inp_ag)

        # shape: [batch, agent, neighbor, agent]x[batch, agent, agent, 128 or 256]->[batch, agent, neighbor, 128 or 256]
        neighbor_obs_repr = Lambda(lambda x: K.batch_dot(x[0], x[1]))([Inp_adj_matrix, neighbor_obs_repr_all])

        if conc_observations_edges:
            # shape: [batch, agent, neighbor, 256]
            neighbor_repr = neighbor_obs_repr

            # shape: [batch, agent, 1, 256].
            self_agent_repr = self_obs_agent_repr

        else:
            #shape: [batch, agent, neighbor, dim=(128*2)=256]
            neighbor_repr = Concatenate(axis=3)([neighbor_obs_repr, Inp_edg])

            Inp_edg_shape = Inp_edg.get_shape().as_list() #List=[batch, agent, neighbor, 128]

            #shape: slice shape: [batch, agent, 1, 128] -> (Concatenate) [batch, agent, 1, dim=(128*2)=256]
            sliced_Inp_edge = Slice(begin=[0, 0, 0, 0], size=[-1, Inp_edg_shape[1], 1, Inp_edg_shape[3]])(Inp_edg)
            self_agent_repr = Concatenate(axis=3)([self_obs_agent_repr, sliced_Inp_edge])

        q = Dense(dv * n_heads, activation="relu", kernel_initializer='random_normal')(self_agent_repr)  # shape: [batch, agent, 1, dim (or dv*n_heads)]. This is query, just one row for each agent, instead of neighbs+1 which is applied by DGN original code.
        k = Dense(dv * n_heads, activation="relu", kernel_initializer='random_normal')(neighbor_repr)  # shape: [batch, agent, neighbor, dim (or dv*n_heads)]. This is key.
        v = Dense(dv * n_heads, activation="relu", kernel_initializer='random_normal')(neighbor_repr)  # shape: [batch, agent, neighbor, dim (or dv*n_heads)]. This is value.

        q_ = Reshape((n_ag, 1, dv, n_heads))(q) #shape: [batch, agent, 1, dim (or dv*n_heads)]->(reshape)[batch, agent, 1, dv, n_heads]
        k_ = Reshape((n_ag, n_neibs, dv, n_heads))(k) #shape: [batch, agent, neighbor, dim (or dv*n_heads)]->(reshape)[batch, agent, neighbor, dv, n_heads)]
        v_ = Reshape((n_ag, n_neibs, dv, n_heads))(v)  # shape: [batch, agent, neighbor, dim (or dv*n_heads)]->(reshape)[batch, agent, neighbor, dv, n_heads)]

        q__ = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(q_) #shape: [batch, agent, 1, dv, n_heads]->(permute)[batch, agent, n_heads, 1, dv]
        k__ = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(k_)  # shape: [batch, agent, neighbor, dv, n_heads)]->(permute)[batch, agent, n_heads, neighbor, dv]
        v__ = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(v_) #shape: [batch, agent, neighbor, dv, n_heads)]->(permute)[batch, agent, n_heads, neighbor, dv]

        att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 4]) / np.sqrt(dv))([q__, k__]) #shape: [batch, agent, n_heads, 1, dv]x[batch, agent, n_heads, neighbor, dv]->[batch, agent, n_heads, 1, neighbor]

        att_ = Lambda(lambda x: K.softmax(x))(att)  # shape: [batch, agent, n_heads, 1, neighbor]
        if ev:
            att__ = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2, 4)))(att_)  # shape: [batch, agent, n_heads, 1, neighbor]->(permute)[batch, agent, 1, n_heads, neighbor]

        out = Lambda(lambda x: K.batch_dot(x[0], x[1]))([att_, v__])  # shape: [batch, agent, n_heads, 1, neighbor]x[batch, agent, n_heads, neighbor, dv]->[batch, agent, n_heads, 1, dv]
        out_ = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 4, 2)))(out)  # shape: [batch, agent, n_heads, 1, dv]->(permute)[batch, agent, 1, dv, n_heads]
        out__ = Reshape((n_ag, dv, n_heads))(out_)  # shape: [batch, agent, 1, dv, n_heads]->(reshape)[batch, agent, dv, n_heads]
        out___ = Reshape((n_ag, dv * n_heads))(out__)  # shape: [batch, agent, dv, n_heads]->(reshape)[batch, agent, 128]

        out____ = Dense(out_dim, activation="relu", kernel_initializer='random_normal')(out___)  # shape: [batch, agent, 128]

        if ev is False:
            return out____
        else:
            return out____, att__

    @staticmethod
    def Q_Net(action_dim, Inp=None):
        # Inp shape: [batch, agent, 128*3(=384)] or [batch, agent, 256+(128*2)=512] in case of conc_observations_edges
        # Output shape: [batch, agent, n_action]

        V = Dense(action_dim, kernel_initializer='random_normal')(Inp)  # shape: [batch, agent, action_dim]

        return V

    def train_target_model(self):
        with self.graph.as_default():
            with self.session.as_default():
                weights = self.model.get_weights()
                target_weights = self.model_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = self.TAU * weights[w] + (1 - self.TAU) * target_weights[w]
                self.model_t.set_weights(target_weights)

    def save_model(self, path_to_save_model):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save(path_to_save_model)

    def model_predict(self, obs_):
        with self.graph.as_default():
            with self.session.as_default():
                obs = obs_.copy()
                obs.append(np.ones(obs[0].shape[0]))  # Append dummy sample weights which are needed
                preds = self.model.predict(obs)
        return preds

    def model_qnet_input_predict(self, obs_):
        with self.graph.as_default():
            with self.session.as_default():
                obs = obs_.copy()
                obs.append(np.ones(obs[0].shape[0]))  # Append dummy sample weights which are needed
                preds = self.qnet_input.predict(obs)
        return preds

    def model_fit(self, x_, y=None, sample_weight=None, epochs=1, batch_size=10, verbose=0):
        with self.graph.as_default():
            with self.session.as_default():
                x = x_.copy()
                x.append(sample_weight) #Pass sample weights in the model in this indirect way
                history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return history

    def model_t_predict(self, obs):
        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model_t.predict(obs)
        return preds

    def model_r_predict(self, obs):
        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model_r.predict(obs)
        return preds

    def model_r_fit(self, x, y, epochs=1, verbose=0):
        with self.graph.as_default():
            with self.session.as_default():
                self.model_r.fit(x, y, epochs=epochs, verbose=verbose)

    def update_q_values(self,
                        len_batch,
                        dones,
                        n_agent,
                        active_flights_m,
                        fls_with_loss_of_separation_m,
                        next_fls_with_loss_of_separation_m,
                        fls_with_conflicts_m,
                        next_fls_with_conflicts_m,
                        history_loss_confl_m,
                        q_values,
                        actions,
                        last_reward,
                        reward_hist,
                        next_active_flights_m,
                        GAMMA,
                        valid_max_target_q_values,
                        dur_of_acts,
                        data_needed_for_delayed_update__,
                        next_timestamp__,
                        different_target_q_values_mask_,
                        default_action_q_value_mask__,
                        np_mask_determ_res_fplan_after_maneuv_m,
                        np_mask_climb_descend_res_fplan_determ_with_confl_loss_m,
                        not_use_max_next_q_mask_,
                        next_flight_phases_m):

        for k in range(len_batch):

            ######Update q_values only for the selected actions######

            for j in range(n_agent):
                # We should update the q-value of an agent if it is active and:
                #   - it is in conflict/loss, or
                #   - it is executing a non-deterministic action (based on dur_of_acts[k][j][0] != 0), or
                #   - np_mask_determ_res_fplan_after_maneuv_m[k][j] or np_mask_climb_descend_res_fplan_determ_with_confl_loss_m[k][j]
                #     is True.
                if active_flights_m[k, j] and \
                        ((j in fls_with_loss_of_separation_m[k] or j in fls_with_conflicts_m[k])
                         or dur_of_acts[k][j][0] != 0 or
                         np_mask_determ_res_fplan_after_maneuv_m[k][j] or
                         np_mask_climb_descend_res_fplan_determ_with_confl_loss_m[k][j]):

                    disc_reward = self.compute_discounted_reward(reward_hist[k][j].copy(),
                                                                  GAMMA,
                                                                  valid_max_target_q_values[k][j],
                                                                  not_use_max_next_q_mask_[k][j])

                    q_values[k][j][actions[k][j]] = disc_reward

        return q_values

    @staticmethod
    def compute_discounted_reward(reward_hist,
                                  GAMMA,
                                  valid_max_target_q_values_,
                                  agent_not_use_max_next_q_mask_):

        rewards_list = []

        # In the above conditions we assume (because of the conditions of "update_q_values" function) that the discounted reward
        # will be calculated only for the flights which are active at the next timepoint, and:
        # - participate in a conflict/loss of separation, or
        # - they are executing a non-deterministic action (based on dur_of_acts[k][j] != 0), or
        # - np_mask_determ_res_fplan_after_maneuv_m[k][j] or np_mask_climb_descend_res_fplan_determ_with_confl_loss_m[k][j]
        #   is True.

        # We use 'agent_not_use_max_next_q_mask_' in order to decide if we should use the calculated max_next_q or not (not means that there is not
        # calculated max_next_q. For more details about how this mask is calculated, see the function 'get_maxq_valid_actions')
        if agent_not_use_max_next_q_mask_:
            rewards_list = reward_hist.copy()

        else:
            last_reward_value_plus_max_next_q = reward_hist[-1] + (GAMMA * valid_max_target_q_values_)
            reward_hist[-1] = last_reward_value_plus_max_next_q
            rewards_list = reward_hist.copy()

        # The above command gets the parameter "GAMMA" and the list of the rewards which are referred to a specific action,
        # and it computes the discounted reward. For instance:
        # inputs: reward_list = [1,2,3], GAMMA=2
        # output: 17 , that is 1 + (2 * (2^1)) + (3 * (2^2)) = r0 + (r1 * (GAMMA^1)) + (r2 * (GAMMA^2))
        # Note that if an action has duration equal to the interval between two steps, the reward list will contain only
        # one reward value. Also, when 'agent_not_use_max_next_q_mask_' is True, the last value of the list
        # will be just the reward and not the sum reward+GAMMA*max_net_Q.
        computed_disc_r = polyval(GAMMA, rewards_list)

        return computed_disc_r
