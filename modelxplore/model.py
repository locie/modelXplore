#!/usr/bin/env python
# coding=utf8


import functools as ft
import itertools as it

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from SALib.analyze import rbd_fast, sobol
from SALib.sample import latin, saltelli

from .sampler import LhsSampler
from .tuner import MultipleTuner, get_tuner
from .utils import sort_by_values


class Model:
    def __init__(self, bounds, function):
        """[summary]

        Arguments:
            bounds {[type]} -- [description]
            function {[type]} -- [description]
        """
        self._vars, self._bounds = zip(*bounds)
        self._problem = dict(num_vars=len(self),
                             names=self._vars,
                             bounds=self._bounds)
        self._function = np.vectorize(function)

    def __call__(self, *args, **kwargs):
        """[summary]

        Raises:
            ValueError -- [description]

        Returns:
            [type] -- [description]
        """
        if args != [] and kwargs == {}:
            if len(args) == 1:
                X = args[0]
                return self._function(*X.T)
            if len(args) == len(self):
                return self._function(*args)

        if kwargs != {}:
            args = [kwargs[var] for var in self._vars]
            return self._function(*args)
        raise ValueError("You should either provide X as an array"
                         " (n samples x dim),"
                         " positional arguments ordonned as %s"
                         " or give them as named argument." %
                         ", ".join(self._vars))

    def __len__(self):
        return len(self._vars)

    def response(self, n, mode="accurate", grid="uniform"):
        """[summary]

        Arguments:
            n {[type]} -- [description]

        Keyword Arguments:
            mode {str} -- [description] (default: {"accurate"})
            grid {str} -- [description] (default: {"uniform"})

        Raises:
            ValueError -- [description]

        Returns:
            [type] -- [description]
        """
        def get_shape(i):
            shape = [1] * (len(self) - 1)
            shape.insert(i, -1)
            return shape
        if isinstance(n, int):
            if grid == "uniform":
                n = [n] * len(self)
            elif grid == "sensitivity":
                sensitivity = self.full_sensitivity_analysis()
                S1 = sensitivity["S1"]
                S2 = {var: sum([value
                                for key, value in sensitivity["S2"].items()
                                if var in key])
                      for var in self._vars}
                n = n * len(self)
                n = [n * max(S1[var] + S2[var], .05)
                     for var in self._vars]
        coords = [np.linspace(*dict(self.bounds)[var], n)
                  for n, var in zip(n, self._vars)]
        if mode == "fast":
            lhs_sampler = LhsSampler(self._problem)
            random_coords = lhs_sampler((n // 10) ** len(self))
            corners = np.array(list(it.product(*self._problem["bounds"])))
            centers = np.vstack(set([tuple(np.vstack(corner)
                                           .mean(axis=0)
                                           .tolist())
                                     for corner
                                     in it.combinations(corners, 2)]))
            random_coords = np.vstack([random_coords, *corners, *centers])
            _y = self(random_coords)
            _coords = [coord.flatten()
                       for coord
                       in np.meshgrid(*coords, indexing="ij")]
            y = griddata(
                random_coords, _y, tuple(_coords),
            ).reshape([n] * len(self))
        elif mode == "accurate":
            y = self(**{key: value.reshape(get_shape(i))
                        for i, (key, value)
                        in enumerate(zip(self._vars, coords))})
        else:
            raise ValueError("mode should be either 'fast' or 'accurate'")

        da = xr.DataArray(y, name="y", coords=coords, dims=self._vars)

        return da

    def sensitivity_analysis(self, N=1000):
        """[summary]

        Keyword Arguments:
            N {int} -- [description] (default: {1000})

        Returns:
            [type] -- [description]
        """
        X = latin.sample(self._problem, N)
        y = self(X)
        S1 = rbd_fast.analyze(self._problem, y, X)["S1"]
        return dict(sorted([(var, idx)
                            for var, idx
                            in zip(self._vars, S1)],
                           key=sort_by_values, reverse=True))

    def full_sensitivity_analysis(self, N=1000):
        """[summary]

        Keyword Arguments:
            N {int} -- [description] (default: {1000})

        Returns:
            [type] -- [description]
        """
        X = saltelli.sample(self._problem, N)
        y = self(X)
        S = sobol.analyze(self._problem, y)
        S1 = {var: s1 for var, s1 in zip(self._vars, S["S1"])}
        S2 = {}
        for j in range(len(self)):
            for k in range(j + 1, len(self)):
                S2[(self._vars[j], self._vars[k])] = S['S2'][j, k]
        return dict(S1=S1, S2=S2)

    @property
    def S1(self):
        return self.sensitivity_analysis()

    @property
    def bounds(self):
        return dict(zip(self._vars, self._bounds))


def meta_factory(model, n_features, *args):
    X = np.array(args).reshape((-1, n_features))
    return model.predict(X)


class MetaModel(Model):

    def __init__(self, bounds, model, hyperparameters):
        """[summary]

        Arguments:
            bounds {[type]} -- [description]
            model {[type]} -- [description]
            hyperparameters {[type]} -- [description]
        """
        self._metamodel = model
        _metamodel_func = ft.partial(meta_factory, model, len(bounds))
        self._hyperparameters = hyperparameters
        super().__init__(bounds, _metamodel_func)

    def fit(self, samples):
        self._metamodel.fit(samples[list(self._vars)]
                            .values.reshape((-1, len(self))),
                            samples["y"].values)

    @property
    def r_squared(self):
        return self._metamodel.r_squared

    @staticmethod
    def tune_metamodel(X, y,
                       algorithms=["k-nn", "SVM", "random-forest"],
                       hypopt=True, num_evals=50, num_folds=2, nprocs=1,
                       **hyperparameters):
        """[summary]

        Arguments:
            X {[type]} -- [description]
            y {[type]} -- [description]

        Keyword Arguments:
            algorithms {list} -- [description] (default: {["k-nn", "SVM", "random-forest"]})
            hypopt {bool} -- [description] (default: {True})
            num_evals {int} -- [description] (default: {50})
            num_folds {int} -- [description] (default: {2})
            nprocs {int} -- [description] (default: {1})

        Raises:
            ValueError -- [description]

        Returns:
            [type] -- [description]
        """
        if isinstance(algorithms, str):
            tune = get_tuner(algorithms)
        elif len(algorithms) == 1:
            tune = get_tuner(algorithms[0])
        else:
            if not hypopt:
                raise ValueError(
                    "if hypopt is False, algorithms cannot be a list.")
            tune = MultipleTuner([get_tuner(algo) for algo in algorithms])
        if hypopt:
            optimal_hyperparameters = tune.auto_tune(
                X,
                y, nprocs=nprocs)
            hyperparameters = dict(
                **optimal_hyperparameters, **hyperparameters)
            metamodel.r_squared = tune.r_squared
        metamodel = tune(**hyperparameters)
        return tune.name, metamodel, hyperparameters
