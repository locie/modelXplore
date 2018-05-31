#!/usr/bin/env python
# coding=utf8

import inspect
import sys

import numpy as np
from fuzzywuzzy import process
from SALib.sample import latin
from scipy.interpolate import NearestNDInterpolator, interp1d
from scipy.stats import beta, norm
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


def get_sampler(name):
    """get a sampler by its name

    Arguments:
        name {str} -- sampler name

    Raises:
        NotImplementedError -- raised if the sampler is not available.

    Returns:
        Sampler -- the requested sampler
    """
    try:
        return available_samplers[name]
    except KeyError:
        err_msg = "%s sampler is not registered." % name
        (suggest, score), = process.extract(name,
                                            available_samplers.keys(),
                                            limit=1)
        if score > 70:
            err_msg += ("\n%s is available and seems to be close. "
                        "It may be what you are looking for !" % suggest)
        err_msg += ("\nFull list of available samplers:\n\t- %s" %
                    ("\n\t- ".join(available_samplers.keys())))
        raise NotImplementedError(err_msg)


def register_sampler(UserSampler):
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

    def __len__(self):
        return self.ndim

    @property
    def inputs(self):
        return self._vars

    def __call__(self, size):
        return self.rvs(size)


class LhsSampler(Sampler):
    name = "lhs"

    def rvs(self, size=1):
        return latin.sample(self._problem, size)


class IncrementalSampler(Sampler):
    name = "incremental"

    def __init__(self, bounds, n=1000, a=20, b=.1):
        """An incremental sampler designed to be history aware : after a first
        lhs sampling, any extra samples will be distributed in order to fill
        the void.

        How the samples are distributed on the distance map is tuned by the
        a and b arguments. You can see that distribution with the
        sampler.distance_dist property (it will return x and y for easy
        plotting). If the distribution is to much concentrated to the right,
        you will loose the randomization. Too much on the left, and the void
        will not be filled in an efficient way.

        Arguments:
            bounds -- a bounds as [(varname, (low, high)), ...]

        Keyword Arguments:
            n {int} -- number of point that will map the euclidian distance (more: slower but more accurate) (default: {1000})
            a {float} -- first beta dist shape arg (default: {20})
            b {float} -- second beta dist shape arg (default: {.1})
        """  # noqa
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

    @property
    def distance_dist(self):
        x = np.linspace(0, 1, 500)
        y = self._generation_dist.pdf(x)
        return x, y


class ResponsiveSampler(Sampler):
    name = "responsive"

    def __init__(self, bounds, n=20,
                 a_dist=20, b_dist=.1,
                 a_magn=5, b_magn=1):
        """A responsive sampler designed to samples according to the distance
        between the previous samples and the output gradient.

        Be careful! That sampler can be slow, it needs to map the input space
        to compute the gradient every time the output is updated. This make
        that sampler not suited for high dimension problem (more than 3) :
        you may want to reduce your in

        How the samples are distributed on the distance map is tuned by the
        a_dist and b_dist arguments.
        How the samples are distributed on the magnitude map is tuned by the
        a_magn and b_magn arguments.

        Arguments:
            bounds -- a bounds as [(varname, (low, high)), ...]

        Keyword Arguments:
            n {int} -- number of point that will map the euclidian distance (more: slower but more accurate) (default: {20})
            a_dist {float} -- first beta dist shape arg for distance map (default: {20})
            b_dist {float} -- second beta dist shape arg for distance map (default: {.1})
            a_magn {float} -- first beta dist shape arg for magnitude map (default: {20})
            b_magn {float} -- second beta dist shape arg for magnitude map (default: {.1})
        """  # noqa
        super().__init__(bounds)
        self._X = None
        self._y = None
        self._reversed_magn = None
        self._reversed_distance = None
        self._distance_dist = beta(a_dist, b_dist)
        self._magn_dist = beta(a_magn, b_magn)
        self._lhs_sampler = LhsSampler(bounds)
        xs, dxs = zip(*[np.linspace(*bound, n, retstep=True)
                        for var, bound in bounds])
        xxs = np.meshgrid(*xs, indexing="ij")
        self._xs = xs
        self._dxs = dxs
        self._xxs = xxs
        self._x_flatten = np.vstack([xx.flatten() for xx in xxs]).T
        self._shape = [n] * self.ndim

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new):
        self._X = new
        self.update_X()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new):
        self._y = new
        self.update_y()

    def update_X(self):
        distance = self._compute_distance(self._x_flatten, self.X).flatten()
        self._distance = distance.flatten()

    def update_y(self):
        gridded = SVR().fit(
            self.X, self.y).predict(self._x_flatten).reshape(self._shape)
        grads = np.gradient(gridded, *self._xs)
        magn = np.sqrt(sum([grad**2 for grad in grads]))
        self._magn = magn.flatten()

    def _reverse(self, xdist, xmagn):
        return np.vstack([reversed_func(xdist, xmagn)
                          for reversed_func
                          in self._reversed_total]).T.squeeze()

    @property
    def _reversed_total(self):
        distance = MinMaxScaler().fit_transform(
            self._distance.reshape((-1, 1))).squeeze()

        try:
            magn = MinMaxScaler().fit_transform(
                self._magn.reshape((-1, 1))).squeeze()
        except AttributeError:
            magn = np.linspace(0, 1, distance.size)

        self._metric = metric = np.vstack([distance, magn]).T

        reversed_funcs = [NearestNDInterpolator(metric, x.squeeze() +
                                                norm.rvs(size=x.size,
                                                         scale=dx / 2))
                          for x, dx in zip(self._x_flatten.T, self._dxs)]
        return reversed_funcs

    def rvs(self, size=1):
        if self._X is not None:
            new_samples = np.zeros((size, self.ndim))
            for i in range(size):
                new_sample = self._reverse(self._distance_dist.rvs(),
                                           self._magn_dist.rvs())
                self.X = np.vstack([self.X, new_sample])
                new_samples[i, :] = new_sample
            return new_samples
        else:
            self.X = self._lhs_sampler(size)
            return self.X

    def _compute_distance(self, inputs, X):
        return euclidean_distances(inputs, X).min(axis=1)


available_samplers = {
    cls[1].name: cls[1]
    for cls
    in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if Sampler in cls[1].__bases__ and getattr(cls[1], "name", False)}
