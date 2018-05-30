#!/usr/bin/env python
# coding=utf8


import itertools as it
from functools import partial, wraps

import numpy as np
import pandas as pd
import xarray as xr
from optunity.metrics import mse, r_squared
from SALib.analyze import rbd_fast, sobol
from SALib.sample import latin, saltelli
from scipy.interpolate import griddata

from .sampler import LhsSampler
from .tuner import MultipleTuner, get_tuner
from .utils import sort_by_values


class Model:
    def __init__(self, bounds, function):
        """Base classe for generate the outputs from some inputs and a function.
        It contains all the method for the surface response generation, sensitivity
        analysis and more.
        This class is a callable, so you can use its instances as a function.

        Arguments:
            bounds {list((str, (float, float))} -- [varname, (high, low)]
            function {callable} -- the function used to generate the output from the inputs

        Attributes:
            inputs {list(str)} -- the name of the model inputs
            bounds {dict} -- the bounds of the model
            S1 {dict} -- first order sensitivity indices computed by the rbd-fast method

        Methods:
            response -- compute the response surface of the model.
            sensitivity analysis -- use the rbd-fast method to perform a sensivity analysis of the model
            full sensitivity analysis -- use the sobol method to perform a sensivity analysis of the model

        Examples:

            >>> model = Model(bounds, func)

            The model call accept different inputs: as well a
            -   (size, ndim) array
            >>> y = model(X)

            -   inputs as positionnal arguments (float, list or array)
            >>> y = model(x1, x2)

            -   inputs as named arguments (float, list or array)
            >>> y = model(x2=0, x1=0)

        """  # noqa
        self._vars, self._bounds = zip(*bounds)
        self._problem = dict(num_vars=len(self),
                             names=self.inputs,
                             bounds=self._bounds)
        self._function = np.vectorize(function)
        self._expensive = True
        self.__call__ = wraps(self.__call__)

    def __call__(self, *args, **kwargs):
        if args != [] and kwargs == {}:
            if len(args) == 1:
                X = args[0]
                return self._function(*X.T)
            if len(args) == len(self):
                return self._function(*args)

        if kwargs != {}:
            args = [kwargs[var] for var in self.inputs]
            return self._function(*args)
        raise ValueError("You should either provide X as an array"
                         " (n samples x dim),"
                         " positional arguments ordonned as %s"
                         " or give them as named argument." %
                         ", ".join(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def response(self, n=50, mode="accurate", grid="uniform", force=False):
        """Compute the response surface of the model. This is an expensive
        method that will request a lot of model call.

        Arguments:
            n {int or list(int)} -- number of step in each dimension (default: {50})

        Keyword Arguments:
            mode {str} -- "fast" or "accurate". (default: {"accurate"})
            "accurate" will compute the model to the full grid. "fast" will
            compute n / 10 ** ndim samples then interpolate these samples on
            the grid via a nearest neighbour interpolator.

            grid {str} -- "uniform" or "sensitivity" (default: {"uniform"})
            if "uniform", a (n x n x ...) grid is used. if "sensitivity", the
            steps will depend of the sensitivity index of each inputs. It
            allows to be more accurate of relevant variables.
            Ignored if n is a list.

        Returns:
            xarray.DataArray -- The gridded surface response as a DataArray
        """  # noqa
        if self._expensive and not force:
            raise ValueError("response surface should not be computed on "
                             "expensive function, but only on metamodel or "
                             "test functions. Use force=True to override that "
                             "behavior")

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
                      for var in self.inputs}
                n = n * len(self)
                n = [n * max(S1[var] + S2[var], .05)
                     for var in self.inputs]
            else:
                raise ValueError(
                    "grid should be either 'uniform' or 'sensitivity'")
        coords = [np.linspace(*dict(self.bounds)[var], n)
                  for n, var in zip(n, self.inputs)]
        if mode == "fast":
            lhs_sampler = LhsSampler(self.bounds)
            random_coords = lhs_sampler(int(np.mean(n) / 10) ** len(self))
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
                        in enumerate(zip(self.inputs, coords))})
        else:
            raise ValueError("mode should be either 'fast' or 'accurate'")

        da = xr.DataArray(y, name="y", coords=coords, dims=self.inputs)

        return da

    def sensitivity_analysis(self, N=1000, force=False):
        """Run a RBD-fast sensitivity analysis and return the first order
        indices.

        Keyword Arguments:
            N {int} -- number of sampled used to run the analysis (default: {1000})

        Returns:
            dict -- first order sensitivity indices
        """  # noqa
        if self._expensive and not force:
            raise ValueError("sensitivity analysis should not be done on "
                             "expensive function, but only on metamodel or "
                             "test functions. Use force=True to override that "
                             "behavior")
        X = latin.sample(self._problem, N)
        y = self(X)
        S1 = rbd_fast.analyze(self._problem, y, X)["S1"]
        return dict(sorted([(var, idx)
                            for var, idx
                            in zip(self.inputs, S1)],
                           key=sort_by_values, reverse=True))

    def full_sensitivity_analysis(self, N=1000, force=False):
        """Run a Sobol sensitivity analysis and return the first and second
        order indices.

        Keyword Arguments:
            N {int} -- number of sampled used to run the analysis (default: {1000})

        Returns:
            dict -- first order sensitivity indices
        """  # noqa
        if self._expensive and not force:
            raise ValueError("full sensitivity analysis should not be done on "
                             "expensive function, but only on metamodel or "
                             "test functions. Use force=True to override that "
                             "behavior")
        X = saltelli.sample(self._problem, N)
        y = self(X)
        S = sobol.analyze(self._problem, y)
        S1 = {var: s1 for var, s1 in zip(self.inputs, S["S1"])}
        S2 = {}
        for j in range(len(self)):
            for k in range(j + 1, len(self)):
                S2[(self.inputs[j], self.inputs[k])] = S['S2'][j, k]
        return dict(S1=S1, S2=S2)

    @property
    def S1(self):
        return self.sensitivity_analysis()

    @property
    def bounds(self):
        return dict(zip(self.inputs, self._bounds))

    @property
    def inputs(self):
        return self._vars


def meta_factory(model, n_features, *args):
    X = np.array(args).reshape((-1, n_features))
    return model.predict(X)


class MetaModel(Model):

    def __init__(self, bounds, model, hyperparameters):
        """A metamodel that can be fit on a more complex model or on
        experimental data.

        Arguments:
            bounds {list((str, (int, int))} -- [(namevar, (low, high))]
            model {sklearn regressor like} -- the metamodel class (obtain via a Tuner or other)
            It needs to accept some hyperparameters as kwargs and to have a fit and a predict method.
            hyperparameters {dict} -- the metamodel hyperparameters
        """  # noqa
        self.samples = pd.DataFrame()
        self._metamodel = model
        _metamodel_func = partial(meta_factory, model, len(bounds))
        self._hyperparameters = hyperparameters
        super().__init__(bounds, _metamodel_func)
        self._expensive = False

    def fit(self, samples):
        self.samples = self.samples.append(samples)
        self._metamodel.fit(self.X, self.y)

    @property
    def X(self):
        return self.samples[list(self.inputs)] .values.reshape((-1, len(self)))

    @property
    def y(self):
        return self.samples["y"].values

    @property
    def r_squared(self):
        return r_squared(self.y, self(self.X))

    @property
    def mse(self):
        return mse(self.y, self(self.X))

    @property
    def metrics(self):
        return dict(r_squared=self.r_squared, mse=self.mse)

    @staticmethod
    def tune_metamodel(X, y, meta_bounds,
                       algorithms=["k-nn", "SVM", "random-forest"],
                       hypopt=True, num_evals=50, num_folds=2,
                       opt_metric="r_squared", nprocs=1, **hyperparameters):
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
                X, y, num_evals=num_evals, num_folds=num_folds,
                opt_metric=opt_metric, nprocs=nprocs)
            hyperparameters = dict(
                **optimal_hyperparameters, **hyperparameters)
        tuned_metamodel = tune(**hyperparameters)
        metamodel = MetaModel(meta_bounds, tuned_metamodel,
                              hyperparameters)
        metamodel.name = tune.name
        return metamodel
