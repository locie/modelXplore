#!/usr/bin/env python
# coding=utf8

import uuid

import numpy as np
import pandas as pd
import pendulum
from SALib.analyze import rbd_fast

from .model import MetaModel, Model
from .sampler import Sampler, available_samplers, get_sampler
from .utils import sort_by_values


class Explorer:
    def __init__(self, bounds, model=None, sampler="lhs"):
        """[summary]

        Arguments:
            bounds {[type]} -- [description]

        Keyword Arguments:
            model {[type]} -- [description] (default: {None})
            sampler {str} -- [description] (default: {"lhs"})

        Raises:
            ValueError -- [description]
        """
        self._vars, self._bounds = zip(*bounds)
        self._problem = dict(num_vars=len(self._vars),
                             names=self._vars,
                             bounds=self._bounds)
        if model:
            self._model = Model(bounds, model)

        self._df = pd.DataFrame(columns=["batch", *self._vars, "y", "time"])

        if isinstance(sampler, Sampler):
            self.sample = sampler(self._problem)
        elif isinstance(sampler, str):
            self.sample = get_sampler(sampler)(self._problem)
        else:
            raise ValueError("sampler should be a Sampler instance, "
                             "or one of the following: "
                             "%s" % [sampler[0]
                                     for sampler
                                     in available_samplers])

    @property
    def bounds(self):
        return dict(zip(self._vars, self._bounds))

    def __len__(self):
        return len(self._vars)

    def clean(self):
        self._df = pd.DataFrame(columns=["batch", *self._vars, "y", "time"])

    def explore(self, n_samples=None, *, X=None, y=None,
                batch_name=None, nprocs=1):
        """[summary]

        Keyword Arguments:
            n_samples {[type]} -- [description] (default: {None})
            X {[type]} -- [description] (default: {None})
            y {[type]} -- [description] (default: {None})
            batch_name {[type]} -- [description] (default: {None})

        Raises:
            ValueError -- [description]
            NotImplementedError -- [description]

        Returns:
            [type] -- [description]
        """
        if not batch_name:
            batch_name = str(uuid.uuid1())[:8]
        time = pendulum.now().to_cookie_string()
        try:
            if n_samples and (X is None and y is None):
                new_inputs = self.sample(n_samples)
                new_outputs = self.model(new_inputs)
            elif n_samples is None and X is not None and y is None:
                new_inputs = X
                new_outputs = self.model(new_inputs)
            elif n_samples is None and X is not None and y is not None:
                new_inputs = X
                new_outputs = y
            else:
                raise ValueError("You should provide either: "
                                 "only nsamples, only X, or X and y.")
        except NotImplementedError:
            raise NotImplementedError("Model not available: "
                                      "you have to provide both X and y.")
        new_df = pd.DataFrame(dict(**{var: inputs
                                      for var, inputs
                                      in zip(self._vars, new_inputs.T)},
                                   y=new_outputs,
                                   batch=batch_name,
                                   time=time))
        self.data = self.data.append(new_df, sort=True)
        return new_df

    def sensitivity_analysis(self, inputs, output):
        """[summary]

        Arguments:
            inputs {[type]} -- [description]
            output {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        S1 = rbd_fast.analyze(self._problem, output, inputs)["S1"]
        return dict(sorted([(var, idx) for var, idx in zip(self._vars, S1)],
                           key=sort_by_values, reverse=True))

    @property
    def S1(self):
        return self.sensitivity_analysis(self.X, self.y)

    @property
    def X(self):
        return self.data[list(self._vars)].values

    def relevant_vars(self, n):
        """[summary]

        Arguments:
            n {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        relevant_vars, _ = zip(*sorted([(var, idx)
                                        for var, idx
                                        in self.S1.items()],
                                       key=sort_by_values, reverse=True)[:n])
        return relevant_vars

    def relevant_X(self, n):
        """[summary]

        Arguments:
            n {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return self._df[list(self.relevant_vars(n))].values

    @property
    def y(self):
        return self.data["y"].values

    @property
    def data(self):
        return self._df

    @data.setter
    def data(self, new_df):
        try:
            self.metamodel.fit(new_df)
        except AttributeError:
            pass
        self._df = new_df

    @property
    def model(self):
        if self._model is not None:
            return self._model
        raise NotImplementedError("Model hasn't been given by the user.")

    def select_metamodel(self, algorithms=["k-nn", "svm", "random-forest"],
                         hypopt=True, features="auto", threshold=.9,
                         num_evals=50, num_folds=2, nprocs=1,
                         **hyperparameters):
        """[summary]

        Keyword Arguments:
            algorithms {list} -- [description] (default: {["k-nn", "svm", "random-forest"]})
            hypopt {bool} -- [description] (default: {True})
            features {str} -- [description] (default: {"auto"})
            threshold {float} -- [description] (default: {.9})
            num_evals {int} -- [description] (default: {50})
            num_folds {int} -- [description] (default: {2})
            nprocs {int} -- [description] (default: {1})
        """
        y = self.y
        if features == "auto":
            sens_sorted_vars, sens_sorted_idx = zip(
                *sorted(self.S1.items(), key=sort_by_values, reverse=True))
            cum_idx = np.cumsum(sens_sorted_idx)
            i = np.where(cum_idx > threshold)

            if len(i[0]) == 0:
                meta_vars = list(sens_sorted_vars)
            else:
                meta_vars = np.array(sens_sorted_vars)[:i[0][0]].tolist()
        elif isinstance(features, int):
            meta_vars = list(self.relevant_vars(features))
        else:
            meta_vars = features

        X = self.data[meta_vars].values
        y = self.y
        tuned_model, hyperparameters = MetaModel.tune_metamodel(
            X, y,
            algorithms=algorithms, hypopt=hypopt,
            num_evals=num_evals, num_folds=num_folds, nprocs=nprocs)

        meta_bounds = [(var, self.bounds[var])
                       for var in meta_vars]
        metamodel = MetaModel(meta_bounds, tuned_model, hyperparameters)
        metamodel.fit(self.data)
        self._metamodel = metamodel
        return metamodel

    @property
    def metamodel(self):
        if self._metamodel is not None:
            return self._metamodel
        raise NotImplementedError("Metamodel hasn't been defined yet")
