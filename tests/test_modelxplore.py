import numpy as np
import pytest
from modelxplore import (Explorer, Sampler, __version__, get_sampler,
                         get_test_function, register_sampler, Tuner,
                         available_samplers, get_tuner, register_tuner)
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor


def test_version():
    assert __version__ == '0.1.0'


@pytest.fixture
def model():
    return get_test_function("ishigami")()


def test_explorer(model):
    explorer = Explorer([("x1", (0, 1)), ("x2", (-5, 5))])

    explorer = Explorer(model.bounds, model, get_sampler("lhs"))
    with pytest.raises(ValueError):
        explorer = Explorer(model.bounds, model, sampler=None)
    explorer = Explorer(model.bounds, model)

    assert len(explorer) == 3
    assert explorer.bounds == dict(x1=(-np.pi, np.pi),
                                   x2=(-np.pi, np.pi),
                                   x3=(-np.pi, np.pi))

    explorer.explore(150)
    explorer.clean()
    X = explorer.sample(150)
    explorer.explore(X=X)
    explorer.clean()
    X = explorer.sample(150)
    y = model(X)
    explorer.clean()
    explorer.explore(X=X, y=y)

    with pytest.raises(ValueError):
        explorer.explore(50, X=X)

    explorer = Explorer(model.bounds)
    with pytest.raises(NotImplementedError):
        explorer.explore(50)


def test_explorer_sensitivity(model):
    explorer = Explorer(model.bounds, model)
    explorer.explore(5000)
    S1 = explorer.sensitivity_analysis()
    assert np.isclose(S1["x1"], .3, atol=.05)
    assert np.isclose(S1["x2"], .45, atol=.05)
    assert np.isclose(S1["x3"], 0., atol=.05)
    explorer = Explorer(model.bounds, model)
    explorer.explore(20)
    with pytest.raises(ValueError):
        explorer.sensitivity_analysis()
    explorer.explore(200)
    explorer.S1
    assert explorer.relevant_vars(2) == ("x2", "x1")
    assert explorer.relevant_X(2).shape == (220, 2)


def test_explorer_metaselection(model):
    explorer = Explorer(model.bounds, model)
    explorer.explore(80)
    with pytest.raises(NotImplementedError):
        explorer.metamodel

    explorer.select_metamodel()
    explorer.select_metamodel("svm")
    explorer.select_metamodel(["k-nn"])
    explorer.select_metamodel(["svm", "k-nn"])
    with pytest.raises(ValueError):
        explorer.select_metamodel(["svm", "k-nn"], hypopt=False)
    assert explorer.select_metamodel(
        "k-nn", features=["x1", "x2"]).inputs == ("x1", "x2")
    assert explorer.select_metamodel(
        "k-nn", features=2).inputs == explorer.relevant_vars(2)
    assert explorer.select_metamodel(
        "k-nn", features=3).inputs == explorer.relevant_vars(3)
    assert explorer.metamodel.inputs == explorer.relevant_vars(3)
    explorer = Explorer(model.bounds, model)
    explorer.explore(800)
    explorer.select_metamodel("k-nn", threshold=.7)
    explorer.select_metamodel("k-nn", threshold=.6)
    explorer.select_metamodel("k-nn", threshold=.5)

    explorer = Explorer(model.bounds, model)
    explorer.explore(40)

    with pytest.raises(ValueError):
        explorer.select_metamodel("k-nn")
    with pytest.raises(ValueError):
        explorer.select_metamodel("k-nn", features=2)


def test_meta(model):
    explorer = Explorer(model.bounds, model)
    explorer.explore(80)
    meta = explorer.select_metamodel("svm")
    assert isinstance(meta.r_squared, float)
    assert isinstance(meta.mse, float)
    assert meta.metrics["r_squared"] == meta.r_squared
    assert meta.metrics["mse"] == meta.mse
    explorer.select_metamodel("svm", nprocs=-1, opt_metric="mse")

    explorer.explore(800)
    meta = explorer.select_metamodel("k-nn", nprocs=-1, features=2)
    meta.response(20)
    meta.response([20, 10])
    meta.response(20, grid="sensitivity")
    with pytest.raises(ValueError):
        meta.response(20, grid="test")
    with pytest.raises(ValueError):
        explorer.model.response(20)

    meta.S1
    meta.full_sensitivity_analysis()

    with pytest.raises(ValueError):
        explorer.model(explorer.X, explorer.data["x1"].values)

    with pytest.raises(ValueError):
        explorer.model.sensitivity_analysis()

    with pytest.raises(ValueError):
        explorer.model.full_sensitivity_analysis()


def test_register_tuner(model):

    class KnnTuner(Tuner):
        name = "k-nn2"
        search = {'n_neighbors': [1, 5]}
        Regressor = KNeighborsRegressor

    register_tuner(KnnTuner)
    with pytest.raises(NotImplementedError):
        get_tuner("knn")
    with pytest.raises(NotImplementedError):
        get_tuner("fake-tuner")
    get_tuner("k-nn2")

    class BadTuner:
        name = "bad"
    with pytest.raises(AttributeError):
        register_tuner(BadTuner)


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
    sampler.clean()


@pytest.mark.parametrize("funcname",
                         ("ishigami",
                          "sobol_g",
                          "mc_cormick",
                          "styblinski_tang"))
def test_tests_function(funcname):
    model = get_test_function(funcname)()
    sampler = get_sampler("lhs")(model.bounds)
    X = sampler.rvs(50)
    y = model(X)
    assert y.size == 50


def test_get_tests_function():
    with pytest.raises(NotImplementedError):
        get_test_function("isigami")
    with pytest.raises(NotImplementedError):
        get_test_function("fake-function")


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
