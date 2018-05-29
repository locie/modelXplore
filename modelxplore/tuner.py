#!/usr/bin/env python
# coding=utf8

import inspect
import multiprocessing as mp
import sys

from fuzzywuzzy import process
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import optunity
import optunity.metrics
from voluptuous import ALLOW_EXTRA, Any, Coerce, Schema


def get_tuner(algorithm):
    try:
        return available_tuners[algorithm]()
    except KeyError:
        err_msg = "%s tuner is not registered." % algorithm
        (suggest, score), = process.extract(algorithm,
                                            available_tuners.keys(),
                                            limit=1)
        if score > 70:
            err_msg += ("\n%s is available and seems to be close. "
                        "It may be what you are looking for !" % suggest)
        err_msg += ("\nFull list of available tuners:\n\t- %s" %
                    ("\n\t- ".join(available_tuners.keys())))
        raise NotImplementedError(err_msg)


def register_tuner(UserTuner):
    global available_tuners
    if Tuner not in UserTuner.__bases__:
        raise AttributeError("The provider tuner should inherit from the "
                             "Tuner base class.")
    available_tuners[UserTuner.name] = UserTuner


class Tuner:
    override_validation = {}

    def __init__(self):
        self._validator = self._build_validator()
        self._r_squared = None

    @property
    def r_squared(self):
        if self._r_squared:
            return self._r_squared
        else:
            raise AttributeError("Tuner hasn't ran yet, metric unavailable.")

    def __call__(self, **hyperparameters):
        return self.Model(**hyperparameters)

    def Model(self, **hyperparameters):
        kwargs = hyperparameters.get("kwargs", {})
        validated_kwargs = self._validator(dict(**hyperparameters,
                                                **kwargs))
        return self.Regressor(**validated_kwargs)

    def fit(self, X, y, **kwargs):
        """Initialize and train the model"""
        model = self.Model(**kwargs)
        model.fit(X, y)
        return model

    def auto_tune(self, X, y, num_evals=50, num_folds=2, nprocs=1):
        @optunity.cross_validated(x=X, y=y, num_folds=num_folds)
        def performance(x_train, y_train, x_test, y_test, **hyperparameters):
            # fit the model
            model = self.fit(x_train, y_train, **hyperparameters)
            # predict the test set
            predictions = model.predict(x_test)
            return optunity.metrics.r_squared(y_test, predictions)

        if nprocs == -1:
            nprocs = mp.cpu_count()
        if nprocs != 1:
            pmap = optunity.parallel.create_pmap(nprocs)
        else:
            pmap = inspect.signature(
                optunity.minimize_structured).parameters["pmap"].default
        optimal_configuration, info, _ = optunity.maximize_structured(
            performance,
            search_space=self.search,
            num_evals=num_evals,
            pmap=pmap)

        self._r_squared = performance(**optimal_configuration)

        return optimal_configuration

    def _build_validator(self):
        self._sig = inspect.signature(self.Regressor)
        default_hyperpars = {key: value.default
                             for key, value
                             in self._sig.parameters.items()
                             if value.default not in [None, inspect._empty]}
        schema = Schema.infer(default_hyperpars).schema
        coerce_schema = {key: Coerce(value)
                         for key, value
                         in schema.items()
                         if key != "kwargs"}
        coerce_schema.update(self.override_validation)
        self._coerce_schema = Schema(coerce_schema,
                                     extra=ALLOW_EXTRA)

        def validator(pars):
            dpars = default_hyperpars.copy()
            dpars.update({key: value
                          for key, value
                          in pars.items()
                          if value is not None})
            pars = self._coerce_schema(dpars)
            return pars

        return validator


class MultipleTuner(Tuner):

    def __init__(self, tunes):
        self._tunes = {tune.name: tune for tune in tunes}
        self.search = {"algorithm": {tune.name: tune.search
                                     for tune in tunes}}
        self._r_squared = None

    @staticmethod
    def intersection_hyperparameters(kwargs, sig):
        """Inspect the function signature to identify the relevant keys
        in a dictionary of named parameters.
        """
        func_parameters = sig.parameters
        kwargs = {key: value
                  for key, value
                  in kwargs.items()
                  if (key in func_parameters and key != "algorithm")}
        return kwargs

    def Model(self, **hyperparameters):
        algorithm = hyperparameters["algorithm"]
        self.name = algorithm
        tune = self._tunes[algorithm]

        filtered_hypars = self.intersection_hyperparameters(hyperparameters,
                                                            tune._sig)

        return tune.Model(**filtered_hypars)


class RandomForestTuner(Tuner):
    name = "random-forest"
    search = {'n_estimators': [10, 50]}
    Regressor = RandomForestRegressor


class KnnTuner(Tuner):
    name = "k-nn"
    search = {'n_neighbors': [1, 5]}
    Regressor = KNeighborsRegressor


class SVMTuner(Tuner):
    name = "svm"
    search = {'kernel': {'linear': {'C': [0, 2]},
                         'rbf': {'gamma': [0, 1],
                                 'C': [0, 10]},
                         'poly': {'degree': [2, 5],
                                  'C': [0, 50],
                                  'coef0': [0, 1]}
                         }
              }
    Regressor = SVR
    override_validation = dict(degree=Coerce(float),
                               gamma=Any('auto', Coerce(float)))


available_tuners = {
    cls[1].name: cls[1]
    for cls
    in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if Tuner in cls[1].__bases__ and getattr(cls[1], "name", False)}
