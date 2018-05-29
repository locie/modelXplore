import pytest
import numpy
import modelxplore as mx


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy


@pytest.fixture(autouse=True)
def add_base_modelxplore(doctest_namespace):
    doctest_namespace['Explorer'] = mx.Explorer
    mc = mx.get_test_function("mc_cormick")()
    doctest_namespace['mc_cormick_model'] = mc
    doctest_namespace['mc_cormick_bounds'] = mc.bounds
    explorer = mx.Explorer(mc.bounds, mc)
    explorer.explore(150)
    doctest_namespace['explorer'] = explorer
