.. _code:

Overview
==================

The Quantum Machine Learning Toolbox consists of the two :class:`CircuitLearner` classes for automatic and numerical differentation. These classes can be used for optimization, supervised and unsupervised learning with variational circuits.

Software components
-------------------

**Frontend:**

* :mod:`qmlt.helpers` - collection of learner-independent helpers; these can be used with either backend.

**Numerical backend:**

* :mod:`qmlt.numerical` - learner class for the training of user-provided quantum circuits.
* :mod:`qmlt.numerical.losses` - collection of loss functions.
* :mod:`qmlt.numerical.regularizers` - collection of regularizers.
* :mod:`qmlt.numerical.helpers` - collection of helpers to set up an experiment with the learners. 
* :mod:`qmlt.numerical.plot` - collection of functions for plotting the outputs of log files. 


**TF backend:**

* :mod:`qmlt.tf` - learner class for the training of user-provided quantum circuits.
* :mod:`qmlt.tf.helpers` - collection of helpers to set up an experiment with the learners. 


Code details
------------
.. automodule:: qmlt
   :members:
   :inherited-members:

