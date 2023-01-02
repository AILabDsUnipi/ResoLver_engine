import pandas as pd
import flight_utils as flight_utils
from datetime import datetime, timezone
from env_config import env_config
from math import cos, sin, radians, degrees, sqrt
import numpy as np
from sector_utils import SectorData
import utils
import faulthandler; faulthandler.enable()

class Environment(object):

    def __init__(self):
        self.dt = 5
        self.flight_index = {}
        self.flight_entry_index = {}
        self.active_flights_mask = None
        self.active_flights = None
        self.flight_arr = None
        self.init_timestamp = 1498763735
        self.timestamp = self.init_timestamp
        self.read_flights(self.init_timestamp)
        self.features = ['flightKey', 'longitude', 'latitude',
                                 'x', 'y', 'altitude',
                                 'speed_x_component', 'speed_y_component',
                                 'course', 'speed', 'alt_speed']
        self.feature_size = len(self.features)
        self.flight_num = len(self.flight_index)
        self.initialize(self.init_timestamp)
        bins_median_dict = utils.equibins()
        # d_angles 0 and d_speeds 0 must always be first in the list
        utils.utils_global['d_angles'] = np.array([0.] + bins_median_dict['d_angles_transformed'].tolist())
        utils.utils_global['d_speeds'] = np.array([0.] + bins_median_dict['d_h_speeds_transformed'].tolist())
        self.d_angles = np.array([0.] + bins_median_dict['d_angles_transformed'].tolist())
        self.d_speeds = np.array([0.] + bins_median_dict['d_h_speeds_transformed'].tolist())
        self.sector_data = SectorData()

    # [0: RTkey, 1: x, 2: y, 3: x_speed, 4: y_speed, 5: altitudeInFeets, 6: alt_speed, 7: max_altitude]
    def edges_between_active_flights(self):

        active_flights_idxs = np.where(self.active_flights_mask == True)[0]
        active_flights = self.flight_arr[active_flights_idxs]

        print(active_flights.shape)
        edges = utils.compute_edges(active_flights[:, np.array([0, 3, 4, 6, 7, 5, 10])],
                                    active_flights_idxs, self.d_angles, self.d_speeds, 15*1852, 20*60)
        print('edges')
        print(edges)
        exit(0)

    def out_of_sector(self):
        'LECSCEN'
        print(self.active_flights_mask)
        print(self.flight_arr.tolist())
        print(np.apply_along_axis(self.sector_data.point_in_sector, 1, self.flight_arr[:, 3:6],
                                  timestamp=self.timestamp,
                                  sectorID='LECSCEN', transformed=True))
        exit(0)

    def new_active_flights(self):
        try:
            for flightKey in self.flight_entry_index[self.timestamp]:
                dict_entry = self.flight_index[int(flightKey)]
                self.flight_arr[dict_entry['idx'], :] = dict_entry['df'][self.features].iloc[0].values
                self.active_flights_mask[dict_entry['idx']] = True
        except KeyError:
            pass

    def initialize(self, timestamp):

        self.flight_arr = np.zeros([self.flight_num, self.feature_size])
        self.active_flights_mask = np.zeros(self.flight_num).astype(bool)

        for flightKey in self.flight_entry_index[timestamp]:
            dict_entry = self.flight_index[int(flightKey)]
            self.flight_arr[dict_entry['idx'], :] = dict_entry['df'][self.features].iloc[0].values
            self.active_flights_mask[dict_entry['idx']] = True

        self.active_flights = self.flight_arr[self.active_flights_mask]

    def move_flights(self, actions):
        self.flight_arr[:, -3:] += actions
        self.flight_arr[:, -5] = (np.sin(np.radians(self.flight_arr[:, -3]))) * self.flight_arr[:, -2]
        self.flight_arr[:, -4] = (np.cos(np.radians(self.flight_arr[:, -3]))) * self.flight_arr[:, -2]

        self.flight_arr[:, 3] += self.flight_arr[:, -5] * self.dt
        self.flight_arr[:, 4] += self.flight_arr[:, -4] * self.dt
        self.flight_arr[:, 5] += actions[:, 2] * self.dt

    def step(self, actions):
        """
        Applies actions and returns s' and rewards.
        :param actions: N x [dcourse,dspeed,d_alt_speed] matrix where N is the number of agents.
        :return: Next state s' and rewards.
        """

        actions = self.active_flights_mask.astype(int)[:, np.newaxis]*actions
        self.move_flights(actions)
        self.timestamp += self.dt
        self.new_active_flights()
        self.edges_between_active_flights()

    def read_flights(self, init_timestamp):

        own_columns = ['flightKey',	'FPkey', 'callsign', 'adep', 'ades', 'aircraft', 'latitude', 'longitude',
                       'altitude', 'timestamp', 'cruiseAltitudeInFeet', 'cruiseSpeed', 'aircraftWake', 'flightType',
                       'timestamp_start', 'timestamp_end', 'atco_event_time_annotation', 'mwm_codigo', 'EntryExitKey_x',
                       'myID', 'flightKeyRels', 'EntryTimeRels', 'ExitTimeRels', 'EntryExitKey_y', 'SectorID',
                       'flightKeyRel', 'entryTime', 'exitTime']
        rel_columns = ['RTkey', 'FPkey', 'callsign', 'departureAerodrome', 'arrivalAerodrome', 'cruiseAltitudeInFeets',
                       'cruiseSpeed', 'flightType', 'aircraftType', 'aircraftWake', 'timestamp', 'longitude', 'latitude',
                       'altitudeInFeets', 'heading', 'moduleSpeedInKnots', 'xSpeedInKnots', 'ySpeedInKnots',
                       'verticalSpeedInFeetsPerSecond']

        own_df = pd.read_csv('/media/hdd/Documents/TP4CFT_Datasets/enriched_w_sectors'
                             '/LEMG_EGKK/1000392_enriched_w_sectors.csv', sep='\t', names=own_columns).iloc[:10]

        flight_idx = 0
        for idx, neighbour_flight in own_df[['flightKeyRels', 'EntryTimeRels', 'ExitTimeRels']].iterrows():
            flight_key_rels = list(map(int, neighbour_flight['flightKeyRels'].split(',')))
            entry_time_rels = neighbour_flight['EntryTimeRels'].split(',')
            exit_time_rels = neighbour_flight['ExitTimeRels'].split(',')
            for i, flight_key in enumerate(flight_key_rels):
                if flight_key not in self.flight_index:
                    flight_df = pd.read_csv('/media/hdd/Documents/TP4CFT_Datasets/datasets/TAPAS_env_samples/' +
                                            str(flight_key)+'.csv', sep=',', names=rel_columns)
                    flight_df = flight_df.rename(columns={'RTkey': 'flightKey', 'altitudeInFeets': 'altitude'})

                    flight_df['timestamp'] = flight_df['timestamp'].\
                        apply(lambda x: (datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')).
                              replace(tzinfo=timezone.utc).timestamp())
                    flight_df = flight_df[(flight_df['timestamp'] >= int(entry_time_rels[i])) &
                                          (flight_df['timestamp'] <= int(exit_time_rels[i]))]
                    if not env_config['historical_transition']:
                        flight_df = flight_df[(flight_df['timestamp'] >= int(entry_time_rels[i])) &
                                              (flight_df['timestamp'] <= int(exit_time_rels[i])) &
                                              (flight_df['timestamp'] >= int(init_timestamp))].sort_values('timestamp')
                        flight_df = flight_df.iloc[[0, 1]]

                    flight_df = flight_utils.transform_compute_speed_bearing(flight_df)
                    flight_df['timestamp'] = flight_df['timestamp'].astype(int)
                    if not env_config['historical_transition']:
                        flight_df = flight_df.iloc[[0]]
                    first_row = flight_df.iloc[0]
                    entry_timestamp = first_row['timestamp']
                    flightKey = int(first_row['flightKey'])
                    if entry_timestamp not in self.flight_entry_index:
                        self.flight_entry_index[entry_timestamp] = set([flightKey])
                    else:
                        self.flight_entry_index[entry_timestamp].add(flightKey)

                    flight_df = flight_df[['flightKey', 'longitude', 'latitude', 'x', 'y', 'altitude', 'timestamp',
                                           'speed_x_component', 'speed_y_component', 'course', 'speed', 'alt_speed']]

                    self.flight_index[int(flight_key)] = {'idx': flight_idx,
                                                          'entryTime': int(entry_time_rels[i]),
                                                           'exitTime': int(exit_time_rels[i]),
                                                           'df': flight_df}
                    flight_idx += 1

        if not env_config['historical_transition']:
            own_df = own_df[(own_df['timestamp'] >= int(init_timestamp))].sort_values('timestamp')
            own_df = own_df.iloc[[0, 1]]
        own_first_row = own_df.iloc[0]
        own_df = flight_utils.transform_compute_speed_bearing(own_df)[['flightKey', 'longitude', 'latitude',
                                                                       'x', 'y', 'altitude', 'timestamp',
                                                                       'speed_x_component', 'speed_y_component',
                                                                       'course', 'speed', 'alt_speed']]
        own_df = own_df.iloc[[0]]

        ownKey = int(own_first_row['flightKey'])
        self.flight_index[ownKey] = {'idx': flight_idx,
                                     'entryTime': own_first_row['entryTime'],
                                     'exitTime': own_first_row['exitTime'],
                                     'df': own_df}

        entry_timestamp = own_first_row['timestamp']

        if entry_timestamp not in self.flight_entry_index:
            self.flight_entry_index[entry_timestamp] = set([ownKey])
        else:
            self.flight_entry_index[entry_timestamp].add(ownKey)

def step_unit_test():
    env = Environment()
    actions = np.array([[i]*3 for i in range(env.flight_arr.shape[0])])
    print(env.flight_arr.tolist())
    for i in range(8):
        env.step(actions)
        if i == 0:
            assert np.allclose(env.flight_arr[:, 3:5], [[545807.0789846644, 214874.27334798896],
                                                        [532890.7009236065, 192151.8927377278],
                                                        [612471.4269904875, 297356.6363871085],
                                                        [571239.7695710933, 297770.8094159008],
                                                        [574768.1931770298, 263866.9761404899],
                                                        [0, 0],
                                                        [533817.8616083944, 283813.2858827754]])
            print(env.flight_arr.tolist())

    print(env.flight_arr.tolist())

    print(env.flight_entry_index)


if __name__ == '__main__':
    env = Environment()
    env.out_of_sector()
