import numpy as np
import pandas as pd
import os
import xgboost as xgb
from datetime import datetime as dt
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tuner import Tuner
from misc.file_fns import read_yaml
from feature_fitter import FeatureFitter


if '__name__' == '__main__':
    my_seed = 123456
    pd.options.display.max_columns = 20
    train_data_filepath         = 'data/train.csv'
    feature_transform_filepath  = 'ref/features.yml'
    param_grid_filepath         = 'ref/param_grid.yml'

    raw_data                    = pd.read_csv(train_data_filepath)
    feature_transform_dict      = read_yaml(feature_transform_filepath)
    param_grid                  = read_yaml(param_grid_filepath)

    outlier_dict = feature_transform_dict['outliers']
    vals_to_scale = feature_transform_dict['vals_to_scale']

    train_data, non_train_data = train_test_split(raw_data, test_size=0.35, random_state=my_seed)
    val_data, test_data = train_test_split(non_train_data, test_size=0.65, random_state=my_seed)

    column_map = {
        'dropoff_latitude':     'dropoff_latitude',
        'dropoff_longitude':    'dropoff_longitude',
        'passenger_count':      'passenger_count',
        'pickup_latitude':      'pickup_latitude',
        'pickup_longitude':     'pickup_longitude',
        'timestamp_column':     'pickup_datetime',
        'trip_duration':        'trip_duration'
    }

    fitter = FeatureFitter(column_map=column_map)
    final_df = fitter.fit(train_data)

    y_col = 'trip_duration'
    train_x = final_df.drop(y_col, axis=1)
    train_y = final_df[y_col]

    model = XGBRegressor()
    model.fit(train_x, train_y)
