import pandas as pd
import numpy as np
import glob
from shapely import wkt
from datetime import datetime, timezone
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import Transformer
import os
from numba import njit

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from utils import utils_global

class SectorData:

    def __init__(self):

        if os.path.exists('./env/SectorGeom/'):
            path = './env/SectorGeom/'
        elif os.path.exists('./../SectorGeom/'):
            path = './../SectorGeom/'
        else:
            raise ValueError('None of the options for the path of the sector geometries ("./../SectorGeom/" or '
                             '"./env/SectorGeom/") is valid !!!')

        if len(glob.glob(path + '*2023*')) > 0:
            year = '2023'
            self.years = ['2022', '2023']
        else:
            ValueError('None of the options (2023, ..) for the path of the sector geometries is valid !!!')

        self.project = Transformer.from_crs("EPSG:4326", "EPSG:2062", always_xy=True).transform

        self.inverse_project = Transformer.from_crs("EPSG:2062", "EPSG:4326", always_xy=True).transform

        self.sectors_dict = self.read_sectors(path)
        self.airblocks_dict = self.read_airblocks(path)
        self.airblocks_dict_transformed = self.transform_geom_v2(self.airblocks_dict, '')

        self.sectors_geom_dict = self.read_sectors_geom(path)
        self.transform_geom(self.sectors_geom_dict, 'geometry(WKT)')

        self.activation_dict = self.read_activation(path)

    def transform_geom_v2(self, geom_dict, col_name='airblockWKT'):
        new_dict = {}
        if col_name == '':
            for key in geom_dict:

                new_dict[key] = transform(self.project, geom_dict[key])

        else:
            for key in geom_dict:
                new_dict[key][col_name] = transform(self.project, geom_dict[key][col_name])

        return new_dict

    def read_sectors_geom(self, path):
        dfs_dict = {}
        dfs = []
        for i, file in enumerate(glob.glob(path+'*'+self.years[0]+'*') + glob.glob(path+'*'+self.years[1]+'*')):
            for j, geom_file in enumerate(glob.glob(file+'/sectors_geom.tsv')):
                df = pd.read_csv(geom_file, sep='\t')
                dup_df = df[df.duplicated(['AIRAC', 'sectorID'])]
                if dup_df.shape[0] != 0:
                    print('duplicate found')
                    print(dup_df)
                df = df.set_index(['AIRAC', 'sectorID'])
                dfs.append((df))

        df = pd.concat(dfs, axis=0)
        df.to_csv('sec_geoms.csv' if os.path.exists('./../environment') else './env/environment/sec_geoms.csv')
        sectors_geom_dict = df.to_dict('index')
        sectors_geom_dict = self.convert_to_shapely_objects(sectors_geom_dict, 'geometry(WKT)')

        return sectors_geom_dict

    def read_airblocks(self, path):
        airblock_dict = {}
        AIRACS = []
        for i, file in enumerate(glob.glob(path+'*'+self.years[0]+'*')+glob.glob(path+'*'+self.years[1]+'*')):
            f_split = file.split('/')

            dt = datetime.strptime(f_split[-1], '%Y%m%d')
            airac_ts = dt.replace(tzinfo=timezone.utc).timestamp()
            AIRACS.append(airac_ts)
            for j, sector_file in enumerate(glob.glob(file+'/airblocks.tsv')):
                df = pd.read_csv(sector_file, sep='\t')

                for g_name, group in df.groupby(['airblockID', 'AIRAC']):
                    name = (g_name[0], airac_ts)
                    assert group.shape[0] == 1

                    airblock_dict[name] = wkt.loads(group['geometry(WKT)'].values[0])

        AIRACS_np = np.array(AIRACS)
        AIRACS_np.sort()

        airac_diffs_np = AIRACS_np[1:] - AIRACS_np[:-1]
        self.airacs = AIRACS_np
        self.airac_t_start = AIRACS_np[0]
        self.airac_cell_size = airac_diffs_np[0]

        return airblock_dict

    def convert_to_shapely_objects(self, airblock_dict, col_name='airblockWKT'):

        if os.path.exists('./env/environment/'):
            csv_path = './env/environment/'
        elif os.path.exists('./../environment/'):
            csv_path = './../environment/'
        else:
            raise ValueError('None of the options for the path of the environment'
                             ' ("./env/environment/" or "./../environment/") is valid!!!')

        mbr_spain_df = pd.read_csv(csv_path + 'mbr_spain_2.csv')
        mbr_spain = wkt.loads(mbr_spain_df.values[0][0])

        sector_geom_list = []
        keys_to_del = []
        for key in airblock_dict:

            airblock_dict[key][col_name] = wkt.loads(airblock_dict[key][col_name])
            if not (mbr_spain.intersects(airblock_dict[key][col_name])):
                keys_to_del.append(key)
            else:
                if key[1] == 'LPPCW365':
                    print(key)
                sector_geom_list.append([key[0],key[1],airblock_dict[key][col_name].wkt])

        for key in keys_to_del:

            del airblock_dict[key]

        pd.DataFrame(sector_geom_list, columns=['AIRAC', 'sectorID', 'geometry(WKT)']).\
            to_csv('sectors_over_spain.csv' if os.path.exists('./../environment')
                                            else './env/environment/sectors_over_spain.csv',
                   index=False)

        return airblock_dict

    def transform_geom(self, geom_dict, col_name='airblockWKT'):
        for key in geom_dict:
            geom_dict[key][col_name+'_transformed'] = transform(self.project, geom_dict[key][col_name])

    def area_stats(self, geom_dict, col_name='geometry(WKT)_transformed'):
        areas = []

        for key in geom_dict:
            areas.append([geom_dict[key][col_name].area, key])
        areas = np.array(areas)
        print('mean')
        print(np.mean(areas[:, 0]))
        print('sqrt mean')
        print(np.sqrt(np.mean(areas[:, 0])))
        print('std')
        print(np.std(areas[:, 0]))
        print('sqrt std')
        print(np.sqrt(np.std(areas[:, 0])))
        print('median')
        print(np.median(areas[:, 0]))
        print('sqrt median')
        print(np.sqrt(np.median(areas[:, 0])))
        print('max median')
        print(np.max(areas[:, 0]))
        print('sqrt max median')
        print(np.sqrt(np.max(areas[:, 0])))
        max_idx = np.argmax(areas[:, 0])
        print(max_idx)
        print(areas[max_idx])

    def read_sectors_geom(self, path):
        dfs_dict = {}
        dfs = []
        for i, file in enumerate(glob.glob(path+'*'+self.years[0]+'*')+glob.glob(path+'*'+self.years[1]+'*')):
            for j, geom_file in enumerate(glob.glob(file+'/sectors_geom.tsv')):
                df = pd.read_csv(geom_file, sep='\t')
                dup_df = df[df.duplicated(['AIRAC', 'sectorID'])]
                if dup_df.shape[0] != 0:
                    print('duplicate found')
                    print(dup_df)
                df = df.set_index(['AIRAC', 'sectorID'])
                dfs.append((df))

        df = pd.concat(dfs, axis=0)
        df.to_csv('sec_geoms.csv' if os.path.exists('./../environment') else './env/environment/sec_geoms.csv')
        sectors_geom_dict = df.to_dict('index')
        sectors_geom_dict = self.convert_to_shapely_objects(sectors_geom_dict, 'geometry(WKT)')

        return sectors_geom_dict

    def read_activation(self, path):
        sector_activation_dict = {}

        for i, file in enumerate(glob.glob(path + '*' + self.years[0] + '*') +
                                 glob.glob(path + '*' + self.years[1] + '*')):
            f_split = file.split('/')

            dt = datetime.strptime(f_split[-1], '%Y%m%d')
            AIRAC_timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
            for j, sector_file in enumerate(glob.glob(file + '/activation.tsv')):
                df = pd.read_csv(sector_file, sep='\t')

                df = df.sort_values(by=['sectorID', 'activationStart'])
                for g_name, group in df.groupby(['sectorID', 'AIRAC']):
                    name = (g_name[0], AIRAC_timestamp)

                    sector_activation_dict[name] = group.values[:, 2:]

        return sector_activation_dict

    def sector_active(self, sectorID, AIRAC_ts, timestamp):

        for activation_interval in self.activation_dict[(sectorID, AIRAC_ts)]:
            if timestamp >= activation_interval[0] and timestamp <= activation_interval[1]:
                return True

        return False

    def read_sectors(self, path):
        sector_dict = {}

        for i, file in enumerate(glob.glob(path+'*'+self.years[0]+'*')+glob.glob(path+'*'+self.years[1]+'*')):
            f_split = file.split('/')

            dt = datetime.strptime(f_split[-1], '%Y%m%d')
            AIRAC_timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
            for j, sector_file in enumerate(glob.glob(file+'/sectors.tsv')):
                df = pd.read_csv(sector_file, sep='\t')

                df = df.sort_values(by=['sectorID', 'airblockID'])
                for g_name, group in df.groupby(['sectorID', 'AIRAC']):
                    name = (g_name[0], AIRAC_timestamp)

                    sector_dict[name] = {}
                    for airblock in group.values:
                        airblockID = airblock[2]
                        if not airblockID in sector_dict[name]:
                            sector_dict[name][airblockID] = [{'minAlt': airblock[3], 'maxAlt': airblock[4]}]
                        else:
                            sector_dict[name][airblockID].append({'minAlt': airblock[3], 'maxAlt': airblock[4]})

        return sector_dict

    def find_AIRAC(self, timestamp):
        idx = np.searchsorted(self.airacs, timestamp)-1
        airac_ts = self.airacs[idx]

        return airac_ts

    def line_segment_sector_intersection(self, line_segment, timestamp, sectorID, transformed):

        airac_ts = self.find_AIRAC(timestamp)
        keyFalse = False
        sector_active = True
        if not self.sector_active(sectorID, airac_ts, timestamp):
            sector_active = False
            return False, keyFalse, sector_active, None

        if not (sectorID, airac_ts) in self.sectors_dict:
            keyFalse = True
            return False, keyFalse, sector_active, None

        sector_airblocks = self.sectors_dict[(sectorID, airac_ts)]

        shapely_linestring = LineString(line_segment[:, :2])
        if transformed:
            airblocks_dict = self.airblocks_dict_transformed
        else:
            airblocks_dict = self.airblocks_dict
        segment_distance = np.sqrt(np.sum((line_segment[1, :2]-line_segment[0, :2])**2, axis=0))
        alt_rate = (line_segment[1, 2]-line_segment[0, 2])/segment_distance
        intersection_points = []

        for airblock_key in sector_airblocks:
            airblock_geom = airblocks_dict[(airblock_key, airac_ts)]

            intersection_p = shapely_linestring.intersection(airblock_geom.boundary)

            if intersection_p.is_empty:
                continue

            intersection_p_np = np.array(intersection_p)
            if intersection_p_np.ndim == 1:
                intersection_p_np = intersection_p_np[np.newaxis]

            if intersection_p_np.ndim > 2:
                print('too many dimenstions')
                exit(0)

            for int_p_np in intersection_p_np:
                alt_on_intersection = line_segment[0, 2]+np.sqrt(np.sum((int_p_np - line_segment[0, :2])**2,
                                                                        axis=0))*alt_rate

                for level_dict in sector_airblocks[airblock_key]:
                    if alt_on_intersection <= level_dict['maxAlt'] \
                            and alt_on_intersection >= level_dict['minAlt']:
                        distance_from_segment_end = np.sqrt(np.sum((line_segment[1, :2]-int_p_np)**2, axis=0))

                        intersection_points.append(np.append(int_p_np, [alt_on_intersection,
                                                                        distance_from_segment_end]))
                        break

        if len(intersection_points) == 0:
            return False, keyFalse, sector_active, None

        intersection_points_np = np.array(intersection_points)

        idx = np.argmin(intersection_points_np[:, -1])

        return True, keyFalse, sector_active, intersection_points_np[idx]

    def point_in_sector(self, point, timestamp, sectorID, transformed):
        """
        :param point: 3D point either [lon, lat, alt] or transformed
        :param timestamp: timestamp
        :param sectorID: the ID of the sector
        :param transformed: boolean denoting point is transformed
        :return: True if point in sector, False otherwise
        """

        altitude = point[2]
        airac_ts = self.find_AIRAC(timestamp)
        keyFalse = False

        sector_active = True
        if not self.sector_active(sectorID, airac_ts, timestamp):
            sector_active = False
            return False, keyFalse, sector_active

        if not (sectorID, airac_ts) in self.sectors_dict:
            keyFalse = True
            return False, keyFalse, sector_active

        sector_airblocks = self.sectors_dict[(sectorID, airac_ts)]

        shapely_point = Point(point[:2])
        flag = False
        if transformed:
            airblocks_dict = self.airblocks_dict_transformed
        else:
            airblocks_dict = self.airblocks_dict
        for airblock_key in sector_airblocks:
            airblock_geom = airblocks_dict[(airblock_key, airac_ts)]

            for level_dict in sector_airblocks[airblock_key]:

                if altitude <= level_dict['maxAlt']\
                    and altitude >= level_dict['minAlt']\
                    and airblock_geom.contains(shapely_point):
                    flag = True

                    break
            if flag:
                break

        return flag, keyFalse, sector_active

def unit_test_sector_active():
    sector_data = SectorData()
    assert sector_data.sector_active('BIRDES', 1544054400.0, 1544054395) == False
    assert sector_data.sector_active('BIRDES', 1544054400.0, 1544054440) == True
    assert sector_data.sector_active('BIRDES', 1544054400.0, 1546473545) == False


if __name__ == "__main__":

    unit_test_sector_active()
