#!/usr/bin/env python
# coding=utf8

import inspect
import sys

import numpy as np
from fuzzywuzzy import process
from scipy.interpolate import NearestNDInterpolator, interp1d
from scipy.stats import beta
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from SALib.sample import latin


def get_sampler(algorithm):
    """[summary]

    Arguments:
        algorithm {[type]} -- [description]

    Raises:
        NotImplementedError -- [description]

    Returns:
        [type] -- [description]
    """
    try:
        return available_samplers[algorithm]
    except KeyError:
        err_msg = "%s sampler is not registered." % algorithm
        (suggest, score), = process.extract(algorithm,
                                            available_samplers.keys(),
                                            limit=1)
        if score > 70:
            err_msg += ("\n%s is available and seems to be close. "
                        "It may be what you are looking for !" % suggest)
        err_msg += ("\nFull list of available samplers:\n\t- %s" %
                    ("\n\t- ".join(available_samplers.keys())))
        raise NotImplementedError(err_msg)


def register_sampler(UserSampler):
    """[summary]

    Arguments:
        UserSampler {[type]} -- [description]

    Raises:
        AttributeError -- [description]
    """
    global available_samplers
    if Sampler not in UserSampler.__bases__:
        raise AttributeError("The provider sampler should inherit from the "
                             "Sampler base class.")
    available_samplers[UserSampler.name] = UserSampler


class Sampler:
    def __init__(self, bounds):
        self._vars, self._bounds = zip(*bounds)
        self.ndim = len(bounds)
        self.bounds = bounds
        self._problem = dict(num_vars=self.ndim,
                             names=self._vars,
                             bounds=self._bounds)

    def __call__(self, size):
        return self.rvs(size)


class LhsSampler(Sampler):
    name = "lhs"

    def rvs(self, size=1):
        return latin.sample(self._problem, size)


class IncrementalSampler(Sampler):
    """[summary]
    """
    name = "incremental"

    def __init__(self, bounds, n=1000, a=.95, b=.01):
        """[summary]

        Arguments:
            problem {[type]} -- [description]

        Keyword Arguments:
            n {int} -- [description] (default: {1000})
            a {float} -- [description] (default: {.95})
            b {float} -- [description] (default: {.01})
        """
        super().__init__(bounds)
        self._X = None
        self._generation_dist = beta(a, b)
        self._lhs_sampler = LhsSampler(bounds)
        self._x_flatten = self._lhs_sampler.rvs(n)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new):
        self.update_sampler(new)
        self._X = new

    def update_sampler(self, X):

        _distance = MinMaxScaler((0, 1)).fit_transform(
            self._distance(self._x_flatten, X)[:, None])

        self._pdf = NearestNDInterpolator(self._x_flatten, _distance)

        self._reversed_funcs = [interp1d(_distance.squeeze(),
                                         x.squeeze(),
                                         fill_value="extrapolate")
                                for x in self._x_flatten.T]

    def _reverse(self, x):
        return np.vstack([reversed_func(x)
                          for reversed_func
                          in self._reversed_funcs]).T

    def rvs(self, size=1):
        if self._X is not None:
            new_samples = np.zeros((size, self.ndim))
            for i in range(size):
                new_sample = self._reverse(self._generation_dist.rvs())
                self.X = np.vstack([self.X, new_sample])
                new_samples[i, :] = new_sample
            return new_samples
        else:
            self.X = self._lhs_sampler(size)
            return self.X

    def _distance(self, inputs, X):
        return euclidean_distances(inputs, X).min(axis=1)

    def pdf(self, coords):
        return self._pdf(coords)


available_samplers = {
    cls[1].name: cls[1]
    for cls
    in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if Sampler in cls[1].__bases__ and getattr(cls[1], "name", False)}
