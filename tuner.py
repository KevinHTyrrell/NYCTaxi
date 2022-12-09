import time

import numpy as np
import pandas as pd
import copy
import inspect
import optuna
import xgboost as xgb
from typing import Optional, Union
from misc.file_fns import read_file


class Tuner:
    def __init__(self,
                 param_grid: Union[dict, str],
                 estimator
                 ):
        self._param_grid_input  = param_grid
        self._data_dict         = {}
        self._estimator         = estimator
        self._param_grid        = self._process_param_grid(param_grid)
        self._estimator_fit     = None
        self._study             = None
        self._metric            = None
        self._direction         = None

    def reset_study(self):
        self._study = optuna.create_study()

    def _process_param_grid(self, param_grid):
        if isinstance(param_grid, dict):
            return param_grid
        return read_file(param_grid)

    def set_metric(self, metric, direction):
        self._metric = metric
        self._direction = direction

    def set_data(self,
                 train_x,
                 train_y,
                 val_x,
                 val_y):
        self._data_dict = {
            'train_x':  train_x,
            'train_y':  train_y,
            'val_x':    val_x,
            'val_y':    val_y
        }

    def _get_params(self, trial):
        suggest_dict = {
            'categorical':  trial.suggest_categorical,
            'discrete':     trial.suggest_discrete_uniform,
            'float':        trial.suggest_float,
            'int':          trial.suggest_int,
            'loguniform':   trial.suggest_loguniform,
            'uniform':      trial.suggest_uniform,
        }
        # iterate through param dict and create proper param dict #
        param_grid = {}
        for k, v in self._param_grid.items():
            fn_to_use = suggest_dict[v['type']]
            fn_params = inspect.signature(fn_to_use).parameters.keys()
            input_params = {}
            for p in fn_params:
                if v.get(p):
                    input_params[p] = v.get(p)
            param_grid[k] = fn_to_use(name=k, **input_params)
        return param_grid

    def _objective(self, trial):
        params = self._get_params(trial)

        # get data #
        train_x = self._data_dict['train_x']
        train_y = self._data_dict['train_y']
        val_x = self._data_dict['val_x']
        val_y = self._data_dict['val_y']

        # Fit the model
        optuna_model = self._estimator(**params)
        optuna_model.fit(train_x, train_y)

        # Make predictions
        y_pred = optuna_model.predict(val_x)

        # Evaluate predictions
        metric = self._metric(val_y, y_pred)
        return metric

    def run_study(self, n_trials: int, refit: bool = True):
        self._study = optuna.create_study(direction=self._direction)
        self._study.optimize(self._objective, n_trials=n_trials)
        final_estimator = None

        if refit:
            time.sleep(1)
            print('FITTING BEST MODEL')
            best_params = self._study.best_params
            final_estimator = self._estimator(**best_params)
            final_estimator.fit(self._data_dict['train_x'], self._data_dict['train_y'])
            self._estimator_fit = final_estimator
        return final_estimator