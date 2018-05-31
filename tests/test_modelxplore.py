import numpy as np
import pytest
from modelxplore import (Explorer, Sampler, __version__, get_sampler,
                         get_test_function, register_sampler,
                         available_samplers)
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler


def test_version():
    assert __version__ == '0.1.0'


@pytest.fixture
def model():
    return get_test_function("ishigami")()


def test_explorer(model):
    pass


@pytest.mark.parametrize("sampler_name", ("lhs", "incremental", "responsive"))
def test_samplers(model, sampler_name):
    sampler = get_sampler(sampler_name)(model.bounds)
    assert len(sampler) == len(model.bounds)
    assert sorted(sampler.inputs) == sorted(model.inputs)
    X = sampler.rvs(150)
    y = model(X)

    assert X.shape == (150, len(model.bounds))
    for var, bound in model.bounds:
        assert X.min() > bound[0]
        assert X.max() < bound[1]

    X = sampler.rvs(20)
    y = model(X)
    sampler.X = X
    sampler.y = y
    X = sampler.rvs(20)
    y = model(X)


def test_register_sampler(model):
    class MonteCarloSampler(Sampler):
        name = "monte-carlo"

        def rvs(self, size=1):
            scalers = [MinMaxScaler(bound) for bound in self._bounds]
            samples = uniform.rvs(size=(size, self.ndim))
            samples = [scaler.fit_transform(sample[:, None]).T
                       for scaler, sample
                       in zip(scalers, samples.T)]
            samples = np.vstack(samples).T
            return samples

    register_sampler(MonteCarloSampler)
    explorer = Explorer(model.bounds, model, sampler="monte-carlo")
    explorer.explore(50)

    err_msg = "%s sampler is not registered." % "monte-calo"
    err_msg += ("\n%s is available and seems to be close. "
                "It may be what you are looking for !" % "monte-carlo")
    err_msg += ("\nFull list of available samplers:\n\t- %s" %
                ("\n\t- ".join(available_samplers.keys())))

    with pytest.raises(NotImplementedError, message=err_msg):
        explorer = Explorer(model.bounds, model, sampler="monte-calo")

    with pytest.raises(NotImplementedError):
        sampler = get_sampler("fake-sampler")

    with pytest.raises(NotImplementedError):
        sampler = get_sampler("fake-sampler")

    class BadSampler:
        pass

    with pytest.raises(AttributeError):
        register_sampler(BadSampler)
