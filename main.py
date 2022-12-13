import numpy as np
import pandas as pd
import os
import xgboost as xgb
from datetime import datetime as dt
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

    train_data_raw, non_train_data = train_test_split(raw_data, test_size=0.35, random_state=my_seed)
    val_data_raw, test_data_raw = train_test_split(non_train_data, test_size=0.65, random_state=my_seed)
    # train_data_raw = train_data_raw.iloc[:1000]

    column_map = {
        'dropoff_latitude':     'dropoff_latitude',
        'dropoff_longitude':    'dropoff_longitude',
        'passenger_count':      'passenger_count',
        'pickup_latitude':      'pickup_latitude',
        'pickup_longitude':     'pickup_longitude',
        'timestamp_column':     'pickup_datetime',
        'trip_duration':        'trip_duration'
    }

    fitter = FeatureFitter(column_map=column_map, feature_filepath='ref/features.yml')

    # fit data, arrange columns, and drop nas #
    train_data  = fitter.fit(train_data_raw, trim_outliers=True)
    val_data    = fitter.fit(val_data_raw, trim_outliers=False)
    test_data   = fitter.fit(test_data_raw, trim_outliers=False)

    colnames    = train_data.columns
    train_data  = train_data[colnames].dropna()
    val_data    = val_data[colnames].dropna()
    test_data   = test_data[colnames].dropna()

    # split into x and y data #
    y_col       = 'trip_duration'
    train_x     = train_data.drop(y_col, axis=1)
    val_x       = val_data.drop(y_col, axis=1)
    test_x      = test_data.drop(y_col, axis=1)
    train_y     = train_data[y_col]
    val_y       = val_data[y_col]
    test_y      = test_data[y_col]

    tuner = Tuner(param_grid=param_grid, estimator=XGBRegressor)
    tuner.set_data(train_x, train_y, val_x, val_y)
    tuner.set_metric(mean_squared_error, direction='minimize')
    # tuner.set_metric(mean_absolute_error, direction='minimize')
    estimator = tuner.run_study(n_trials=10, refit=True)

    estimator.score(test_x, test_y)
    test_predict = estimator.predict(test_x)

    # submission file #
    submission_data_raw = pd.read_csv('data/test.csv')
    id_series = submission_data_raw['id']
    submission_data = fitter.fit(submission_data_raw)
    submission_data = submission_data[colnames]