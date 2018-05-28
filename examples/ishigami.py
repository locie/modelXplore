import pylab as pl
import numpy as np

from modelxplore import Explorer, get_test_function

# The tests function are pre-defined models used for testing purpose.
ishigami = get_test_function("ishigami")()

# Because they are models, you can compute the surface response easily.
# By default, the grid is uniform with the same number of points per dimension,
# but you can specify different number of point by dimension, or even let
# the model choose the number of point according to the sensitivity indices.

# The response is a xarray DataArray, and profits all their features
# (see http://xarray.pydata.org/ for more details).

# You shouldn't do that for cpu-expansive models : we allow ourself to
# compute that surface because this is a test model with neglectible cost.

response = ishigami.response(50)

pl.figure(figsize=(8, 3))

pl.subplot(1, 2, 1)
response.sel(x3=-np.pi, method="nearest").plot()

pl.subplot(1, 2, 2)
response.sel(x3=0, method="nearest").plot()

pl.savefig("docs/reference_ishigami.png")


# We can now initialize the explorer with that toy model.

explorer = Explorer(bounds=ishigami.bounds, model=ishigami)

# The sample method is here to sample N inputs, according to the chosen
# sampler. By default, this is a lhs sampler, but others are available.

X = explorer.sample(400)

# We can thus run our model with these values

y = explorer.model(X)

# and feed the explorer with these outputs

explorer.explore(X=X, y=y)

# We can provide only the inputs and let the explorer call the model function.

explorer.clean()
explorer.explore(X=X)

# Or just give a number of samples and let the explorer generate the inputs.

explorer.clean()
explorer.explore(400)

# The explorer can be used to compute the sensivity indices. The already
# available data is used, combined with the rbd-fast method : the cost is
# neglectible, but the results could not be as accurate as possible if the
# number of run is to low.
# For that function, at least 400 run is necessery. Yet, even with less than
# that, it is possible to have an idea of their relative importance.

print("First order sensitivity indices via rbd-fast (model):\n\t-",
      "\n\t- ".join(["%s: %g" % (key, value)
                     for key, value
                     in explorer.S1.items()]))

# Once the explorer is fed with inputs and outputs (which should be the most
# expensive part of the work), we can let the auto-tuner chose a metamodel.
# These meta-models are essentially sklearn regressor, but other can be
# implemented. They will be trained with the current inputs and outputs,
# and the hyperparameters of the model will be choosen in order to minimize
# a metric (here, the R²). You can create the metamodel on all or part of
# the features. By default, the features are sorted by their sensitivity index,
# and taken in order to have a cumulative index bigger than a threshold.
# In other words, we take as few inputs as possible, but we ensure to have
# 90% of the total variance explained.
# For that example, we will fix the features.

explorer.select_metamodel(features=["x1", "x2", "x3"])

# Now, the meta model is attached to the explorer, and will be trained every
# time the explorer is fed with new data.
# Because the metamodel inherit from the Model class, it has all its features.
# It will be way less cpu-expensive than the explored function, and the
# response surface as well as a full sensitivity analysis is easy to obtain.

meta_response = explorer.metamodel.response(50)

pl.figure(figsize=(8, 3))

pl.subplot(1, 2, 1)
meta_response.sel(x3=-np.pi, method="nearest").plot()

pl.subplot(1, 2, 2)
meta_response.sel(x3=0, method="nearest").plot()

pl.savefig("docs/metamodel_ishigami.png")

# The full sensitivity analysis is computed with the Sobol method. This method
# give access to the interaction between the inputs, but is expensive to
# compute : the metamodel having a neglectible cost, we can afford to run that
# king of analysis.

s_idx = explorer.metamodel.full_sensitivity_analysis()
S1 = s_idx["S1"]
S2 = s_idx["S2"]

print("First order sensitivity indices via sobol (metamodel):\n\t-",
      "\n\t- ".join(["%s: %g" % (key, value)
                     for key, value
                     in S1.items()]))

print("Second order sensitivity indices via sobol (metamodel):\n\t-",
      "\n\t- ".join(["%s: %g" % (key, value)
                     for key, value
                     in S2.items()]))

# You now have way more information that you would have only with the computed
# outputs. But keep in mind that these informations have been extracted from a
# metamodel, and their accuracy is directly linked to its fit quality :
# they are not magic method.
