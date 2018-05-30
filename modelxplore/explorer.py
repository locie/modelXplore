#!/usr/bin/env python
# coding=utf8

import uuid
import warnings

import numpy as np
import pandas as pd
import pendulum
from SALib.analyze import rbd_fast

from .model import MetaModel, Model
from .sampler import Sampler, available_samplers, get_sampler
from .tuner import available_tuners
from .utils import sort_by_values


class Explorer:
    def __init__(self, bounds, model=None, sampler="lhs"):
        """Object used to help the user to explore a numerical or experimental
        phenomena.

        Arguments:
            bounds {list} -- list of variable name - bound, like ("x1", (0, 1))

        Keyword Arguments:
            model {callable} -- function with signature (var1, var2...) -> y
            sampler {str or Sampler} -- sampler used to generate inputs (default: {"lhs"})

        Examples:
            Use the explorer without function

            >>> explorer = Explorer([("x1", (0, 1)), ("x2", (-5, 5))])

            Use the explorer with function

            >>> def my_model(x1, x2):
            ...    return np.cos(x1) * np.cos(x2)
            >>> explorer = Explorer([("x1", (0, 1)), ("x2", (-5, 5))], my_model)

            Use the incremental sampler

            >>> explorer = Explorer([("x1", (0, 1)), ("x2", (-5, 5))],
            ...                     my_model, sampler="incremental")

            Do some exploration of the model

            >>> new_data = explorer.explore(150)


        """  # noqa
        self._vars, self._bounds = zip(*bounds)
        self._problem = dict(num_vars=len(self._vars),
                             names=self._vars,
                             bounds=self._bounds)
        if model:
            self._model = Model(bounds, model)

        self._df = pd.DataFrame(columns=["batch", *self._vars, "y", "time"])

        if isinstance(sampler, Sampler):
            self.sample = sampler(bounds)
        elif isinstance(sampler, str):
            self.sample = get_sampler(sampler)(bounds)
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
        """Is a number of samples is provided, generate inputs via the explorer
        sampler. If X is provided, compute y via the attached model. If both
        X and y are provided, attached them to the explorer (this is the only
        option if the user didn't provide a function at the explorer
        initialization.

        Keyword Arguments:
            n_samples {int} -- number of samples to generate (default: {None})
            X {np.array (size, nvar)} -- input samples (default: {None})
            y {np.array or list, (size,)} -- output of the model (default: {None})
            batch_name {str} -- name of the batch. Use uuid if not provided by the user (default: {None})

        Returns:
            DataFrame -- the new data generated
        """  # noqa
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.data = self.data.append(new_df)
        return new_df

    def sensitivity_analysis(self, force=False):
        """Run a RBD-fast sensitivity analysis and return the first order
        indices. Theses indices are not trustworthy if the number of samples
        are not large enough (which depend of the number of dimension and
        the non-linearity of the model), but can be used for parameters
        sorting.

        Arguments:
            inputs {np.array, (size, nvar)}
            output {np.array or list, (size,)}

        Returns:
            dict -- first order sensitivity indices
        """
        S1 = rbd_fast.analyze(self._problem, self.y, self.X)["S1"]
        if self.y.size < 60 and not force:
            raise ValueError("Too few samples for sensitivity analysis."
                             " You will need extra samples for proper"
                             " sensitivity analysis. Use force=True to"
                             " override that behaviour.")
        return dict(sorted([(var, idx) for var, idx in zip(self._vars, S1)],
                           key=sort_by_values, reverse=True))

    @property
    def S1(self):
        return self.sensitivity_analysis()

    @property
    def X(self):
        return self.data[list(self._vars)].values

    def relevant_vars(self, n):
        """return the sorted n most relevant variable names according to the
        sensitivity analysis

        Arguments:
            n {int} -- number of vars needed

        Returns:
            list(str) -- sorted list of n most relevant variable names.
        """
        relevant_vars, _ = zip(*sorted([(var, idx)
                                        for var, idx
                                        in self.S1.items()],
                                       key=sort_by_values, reverse=True)[:n])
        return relevant_vars

    def relevant_X(self, n):
        """return the sorted n most relevant variables inputs according to the
        sensitivity analysis

        Arguments:
            n {[type]} -- [description]

        Returns:
            np.array(size, n) -- array with the n most revelant variables.
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

    def select_metamodel(self, algorithms=available_tuners.keys(),
                         hypopt=True, features="auto", threshold=.9,
                         num_evals=50, num_folds=2, opt_metric="r_squared",
                         nprocs=1, **hyperparameters):
        """

        Keyword Arguments:
            algorithms {list or str} -- [description] (default: all available tuners)
            hypopt {bool} -- [description] (default: {True})
            features {"auto", int or list(str)} -- choice of the feature. Can be auto, an integer of a list of variables (default: {"auto"})
            threshold {float} -- fraction of the variable explained by the selected variables (default: {.9})
            num_evals {int} -- number of tuner optimization evaluations (default: {50})
            num_folds {int} -- number of tuner folds (default: {2})
            opt_metric {str} -- metric used to choose the metamodel ("r_squared" or "mse") (default: {"r_squared"})
            nprocs {int} -- number of processes used by the optimization (-1 for all available cpu) (default: {1})

        Returns:
            MetaModel -- the chosen and trained metamodel.

        Examples:
            Let optunity choose between the all the available tuners (can be very long)
            >>> metamodel = explorer.select_metamodel()

            Let optunity choose between a list of available tuners
            >>> metamodel = explorer.select_metamodel(["k-nn", "random-forest"])

            Use mean-squared error to choose the metamodel
            >>> metamodel = explorer.select_metamodel(opt_metric="mse")

            Fix the tuner, let optunity choose the optimal hyperparameters
            >>> metamodel = explorer.select_metamodel("random-forest")

            Fix the tuner, disable hyperparameter optimization
            >>> metamodel = explorer.select_metamodel("svm", kernel="rbf", hypopt=False)

            Run the tuners with specified features
            >>> metamodel = explorer.select_metamodel("svm", features=["x1"])

            Run the tuners with the 2 more sensitive features
            >>> metamodel = explorer.select_metamodel("svm", features=2)

            Run the tuners with the enough features to have 50% of the variance explain
            >>> metamodel = explorer.select_metamodel("svm", features="auto", threshold=.5)

        """  # noqa
        y = self.y
        if features == "auto":
            try:
                sens_sorted_vars, sens_sorted_idx = zip(
                    *sorted(self.S1.items(), key=sort_by_values, reverse=True))
                cum_idx = np.cumsum(sens_sorted_idx)
                i = np.where(cum_idx > threshold)
            except ValueError:
                raise ValueError("Not enough sample to automatic feature "
                                 "selection. Please specify features as "
                                 "list of variables")

            if len(i[0]) == 0:
                meta_vars = list(sens_sorted_vars)
            else:
                meta_vars = np.array(sens_sorted_vars)[:i[0][0]].tolist()
        elif isinstance(features, int):
            try:
                meta_vars = list(self.relevant_vars(features))
            except ValueError:
                raise ValueError("Not enough sample to automatic feature "
                                 "selection. Please specify features as "
                                 "list of variables")
        else:
            meta_vars = features

        meta_bounds = [(var, self.bounds[var])
                       for var in meta_vars]
        X = self.data[meta_vars].values
        y = self.y
        metamodel = MetaModel.tune_metamodel(
            X, y, meta_bounds,
            algorithms=algorithms, hypopt=hypopt,
            num_evals=num_evals, num_folds=num_folds,
            opt_metric=opt_metric, nprocs=nprocs)

        metamodel.fit(self.data)
        self._metamodel = metamodel
        return metamodel

    @property
    def metamodel(self):
        if self._metamodel is not None:
            return self._metamodel
        raise NotImplementedError("Metamodel hasn't been defined yet")
