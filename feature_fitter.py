'''
Created on 12/7
'''
import datetime

import numpy as np
import pandas as pd
from haversine import haversine
from datetime import datetime as dt


class FeatureFitter:
    def __init__(self,
                 column_map: dict,
                 timestamp_fmt: str = '%Y-%m-%d %H:%M:%S',
                 timestamp_column: str = 'pickup_datetime'):
        self._column_map        = column_map
        self._timestamp_fmt     = timestamp_fmt
        self._timestamp_column  = timestamp_column

    def _get_datetimes(self, datetime_str: str):
        datetime_val = dt.strptime(datetime_str, self._timestamp_fmt)
        datetime_series = pd.Series([datetime_val.hour, datetime_val.month, datetime_val.year, datetime_val.weekday()])
        return datetime_series

    def _get_coordinates(self, df: pd.DataFrame):
        dropoff_lat_colname     = self._column_map['dropoff_latitude']
        dropoff_long_colname    = self._column_map['dropoff_longitude']
        pickup_lat_colname      = self._column_map['pickup_latitude']
        pickup_long_colname     = self._column_map['pickup_longitude']
        dropoff_lat             = df[dropoff_lat_colname].reset_index(drop=True)
        dropoff_long            = df[dropoff_long_colname].reset_index(drop=True)
        pickup_lat              = df[pickup_lat_colname].reset_index(drop=True)
        pickup_long             = df[pickup_long_colname].reset_index(drop=True)
        return dropoff_lat, dropoff_long, pickup_lat, pickup_long

    def _get_datetime_df(self, df: pd.DataFrame):
        timestamp_colname = self._column_map['timestamp_column']
        datetime_series = df[timestamp_colname]
        datetime_info_df = datetime_series.apply(self._get_datetimes)
        datetime_info_df.reset_index(drop=True, inplace=True)
        datetime_info_df.columns = ['hour', 'month', 'year', 'day_of_week']
        return datetime_info_df

    def _get_directions(self, df: pd.DataFrame):
        dropoff_lat, dropoff_long, pickup_lat, pickup_long = self._get_coordinates(df)
        dir_lat                 = abs(dropoff_lat - pickup_lat) / (dropoff_lat - pickup_lat)
        dir_long                = abs(dropoff_long - pickup_long) / (dropoff_long - pickup_long)
        return pd.DataFrame.from_dict({
            'dir_lat': dir_lat,
            'dir_long': dir_long
        })

    def _get_manhatten_distance(self, df: pd.DataFrame):
        dropoff_lat, dropoff_long, pickup_lat, pickup_long = self._get_coordinates(df)
        dist_lat = abs(dropoff_lat - pickup_lat)
        dist_long = abs(dropoff_long - pickup_long)
        return pd.DataFrame.from_dict({
            'dist_lat': dist_lat,
            'dist_long': dist_long
        })

    def _get_haversine_distance(self, df: pd.DataFrame):
        dropoff_lat, dropoff_long, pickup_lat, pickup_long = self._get_coordinates(df)
        pickup_point = zip(pickup_lat, pickup_long)
        dropoff_point = zip(dropoff_lat, dropoff_long)
        combined_points = zip(pickup_point, dropoff_point)
        h_distances = pd.Series([haversine(pickup, dropoff) for pickup, dropoff in combined_points])
        h_distances.name = 'h_distance'
        return h_distances

    def _get_euclidean_distance(self, df: pd.DataFrame, include_log: bool = False):
        dropoff_lat, dropoff_long, pickup_lat, pickup_long = self._get_coordinates(df)
        eu_distances = np.sqrt((dropoff_lat - pickup_lat) ** 2 + (dropoff_long - pickup_long) ** 2)
        to_return = pd.DataFrame.from_dict({'eu_distance': eu_distances})
        if include_log:
            to_return.update({'eu_dustance_log': np.log(eu_distances + 1)})
        return to_return

    def fit(self, df: pd.DataFrame):
        dropoff_lat, dropoff_long, pickup_lat, pickup_long = self._get_coordinates(df)
        datetime_info_df    = self._get_datetime_df(df)
        n_passengers        = df[self._column_map['passenger_count']]
        dir_df              = self._get_directions(df)
        dist_df             = self._get_manhatten_distance(df)
        h_distances         = self._get_haversine_distance(df)
        eu_distance_df      = self._get_euclidean_distance(df)
        trip_duration       = df[self._column_map['trip_duration']]
        feature_df = pd.concat([
            datetime_info_df.reset_index(drop=True),
            n_passengers.reset_index(drop=True),
            dropoff_lat.reset_index(drop=True),
            dropoff_long.reset_index(drop=True),
            pickup_lat.reset_index(drop=True),
            pickup_long.reset_index(drop=True),
            dir_df.reset_index(drop=True),
            dist_df.reset_index(drop=True),
            h_distances.reset_index(drop=True),
            eu_distance_df.reset_index(drop=True),
            trip_duration.reset_index(drop=True)
        ], axis=1)
        return feature_df
