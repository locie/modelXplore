#!/usr/bin/env python
# coding=utf8


import inspect
import sys

import numpy as np
from fuzzywuzzy import process

from SALib.test_functions import Ishigami as Ishi
from SALib.test_functions import Sobol_G

from .model import Model


def get_test_function(algorithm):
    try:
        return available_test_functions[algorithm]
    except KeyError:
        err_msg = "%s test_function is not registered." % algorithm
        (suggest, score), = process.extract(algorithm,
                                            available_test_functions.keys(),
                                            limit=1)
        if score > 70:
            err_msg += ("\n%s is available and seems to be close. "
                        "It may be what you are looking for !" % suggest)
        err_msg += ("\nFull list of available test_functions:\n\t- %s" %
                    ("\n\t- ".join(available_test_functions.keys())))
        raise NotImplementedError(err_msg)


class TestFunction(Model):
    def __init__(self):
        super().__init__(self.bounds, self.model)
        self._expensive = False


class StyblinskiTang(TestFunction):
    name = "styblinski_tang"

    def __init__(self, N):
        self.bounds = [("x%i" % i, (-5, 5)) for i in range(1, N + 1)]
        super().__init__()

    def model(self, *x):
        return sum([x_i**4 - 16 * x_i**2 + 5 * x_i / 2 for x_i in x])


class McCormick(TestFunction):
    name = "mc_cormick"
    bounds = [("x1", (-1.5, 4)),
              ("x2", (-3, 4))]

    def model(self, x1, x2):
        return (np.sin(x1 + x2) + (x1 - x2)**2 -
                1.5 * x1 + 2.5 * x2 + 1)


class SobolG(TestFunction):
    name = "sobol_g"
    bounds = [("x%i" % i, (0, 1))
              for i in range(1, 9)]

    def model(self, *x):
        values = np.hstack(x).reshape((-1, 8))
        return Sobol_G.evaluate(values)


class Ishigami(TestFunction):
    name = "ishigami"
    bounds = [("x%i" % i, (-np.pi, np.pi))
              for i in range(1, 4)]

    def model(self, x1, x2, x3):
        values = np.hstack([x1, x2, x3]).reshape((-1, 3))
        return Ishi.evaluate(values)


available_test_functions = {
    cls[1].name: cls[1]
    for cls
    in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if TestFunction in cls[1].__bases__ and getattr(cls[1], "name", False)}
