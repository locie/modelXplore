ModelXplore, a python based model exploration
=============================================

ModelXplore is an helper library that give some tool to facilitate the
exploration of time-expensive models (or experimentation).

It give access to a variety of samplers, of regression function
(called meta-model), easy access to sensitivity analysis, and make easy the
computation of response surface.

Installation
------------

During the alpha phase (v0.1.0), the software is obviously not production
ready. Thus, it is installable via the github repository :

.. code-block:: bash

  pip install git+git://github.com/locie/modelxplore

Dependencies should be up-to-date. If not, or if something went wrong, feel
free to open an issue : the bug will be fixed asap.

In order to reproduce the notebook examples, you will need holoviews as well:

.. code-block:: bash

  pip install holoviews


Overview
--------

.. code-block:: python3

  from modelxplore import Explorer, get_test_function
  ishigami = get_test_function("ishigami")()

  # We create the explorer with the bounds of the problem and the function.
  expl = Explorer(bounds=ishigami.bounds, function=ishigami)

  # We generate 150 samples, and generate the outputs.
  expl.explore(150)

  # We let the autotuner chose a well-suited metamodel on the 2 most sensitive
  # inputs

  expl.select_metamodel(features=2)

  # We compute, then plot the surface response of the obtained metamodel
  response = expl.metamodel.response(50)
  response.plot()

  # We run a sobol sensitivity analysis on the metamodel (first and second order)
  print(expl.metamodel.full_sensitivity_analysis())

In detail
---------

Model
*****

Meta Model
**********

Explorer
********

Sampler
*******

available samplers are available with

.. code-block:: python3

  from modelxplore import available_samplers
  print(available_samplers)


Tuner
*****

.. code-block:: python3

  from modelxplore import available_tuners
  print(available_tuners)
