import pytest
import numpy
import modelxplore as mx


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy


@pytest.fixture(autouse=True)
def add_base_modelxplore(doctest_namespace):
    doctest_namespace['Explorer'] = mx.Explorer
    doctest_namespace['Model'] = mx.Model
    mc = mx.get_test_function("mc_cormick")()
    doctest_namespace['mc_cormick_model'] = mc
    doctest_namespace['mc_cormick_bounds'] = mc.bounds
    doctest_namespace['func'] = mc
    doctest_namespace['bounds'] = mc.bounds
    explorer = mx.Explorer(mc.bounds, mc)
    explorer.explore(120)
    doctest_namespace['explorer'] = explorer
    doctest_namespace['X'] = explorer.X
    doctest_namespace['y'] = explorer.y
    doctest_namespace['x1'] = explorer.X[:, 0]
    doctest_namespace['x2'] = explorer.X[:, 1]
