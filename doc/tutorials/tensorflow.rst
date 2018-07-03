.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html

.. _tut_tensorflow:

Automatic learning with TensorFlow
==================================

To use the Machine Learning Toolbox for automatic differentiation we have to use the :class:`CircuitLearner` class of the `qmlt.tf` module. The user-provided functions (such as :func:`circuit`, :func:`myloss`, :func:`myregularizer`, and :func:`outputs_to_predictions`) have to be fully written in TensorFlow, so that gradients can be passed through them. At the moment, this excludes any measurements, since these sample from a distribution.

The tf learner extends TensorFlow's :class:`tf.Estimator` `class <https://www.tensorflow.org/programmers_guide/estimators>`_.


To showcase the ``qmlt.tf.CircuitLearner``, we use exactly the same examples as for the numerical learner and highlight the differences. It helps if you familiarize yourself with the :ref:`previous tutorial <tut_numerical>` first. In a nutshell, the differences between the numerical and TensorFlow learner are the following:

* Initial variables are created in the circuit function, rather than passed to it via a hyperparameter.
* For logging, one can simply refer to the name of tensors outside the circuit function.
* Monitoring is executed by TensorBoard instead of matplotlib.


Optimization
------------

Import is similar to the :ref:`numerical learner tutorial <tut_numerical>`, but instead of numpy, we have to import TensorFlow, and the 'numerical' module is replaced by the 'tf' module.

.. code-block:: python

    import strawberryfields as sf
    from strawberryfields.ops import Dgate
    import tensorflow as tf
    from qmlt.tf import CircuitLearner
    from qmlt.tf.helpers import make_param


There are two major differences when constructing :func:`circuit`. First, the circuit parameters are defined in the function (and not given as hyperparameters as before), so they do not have to be passed to the circuit. Second, all operations in :func:`circuit` have to be coded in TensorFlow and be able to pass gradients through.

.. note::

   One has to create at least one TensorFlow variable within :func:`circuit`, otherwise TensorFlow complains that it has no object to train. The learner takes care of initialising the variable.

.. code-block:: python

    def circuit():

        alpha = make_param(name='alpha', constant=0.1)

        eng, q = sf.Engine(1)

        with eng:
            Dgate(alpha) | q[0]

        state = eng.run('tf', cutoff_dim=7, eval=False)

        prob = state.fock_prob([1])
        circuit_output = tf.identity(prob, name="prob")

        return circuit_output

Here we created the variable ``alpha`` using the :func:`~.tf.helpers.make_param` helper function, but one can of course also use TensorFlow's native :func:`tf.get_variable` method.

Next, we define a loss function that takes the ``outputs`` tensor returned by :func:`circuit` and returns a real-valued scalar tensor whose value we intend to minimize during optimisation. Here the output is the negative probability of measuring the Fock state :math:`| 1 \rangle`. Formally this looks exactly like in the numerical tutorial.

.. code-block:: python

    def myloss(circuit_output):
        return -circuit_output

Since we create the circuit parameter in the function, we do not have to pass it to the hyperparameters any more. The rest is the same as before.

.. code-block:: python

    hyperparams = {'circuit': circuit,
                   'task': 'optimization',
                   'optimizer': 'SGD',
                   'init_learning_rate': 0.1,
                   'loss': myloss}

    learner = CircuitLearner(hyperparams=hyperparams)

    learner.train_circuit(steps=50)

.. note::

    Tensorflow prints logs as an error output. In some programming environments this may appear as red writing. 

Again, we arrive at the same result of a probability of :math:`0.3678794` after 50 steps of optimization.


Including custom logging, regularization and monitoring
*******************************************************

Regularization
++++++++++++++

Regularization works the same as in the numerical tutorial. When making the circuit parameter, set

.. code-block:: python

    alpha = make_param(name='alpha', constant=0.2, regularize=True)

As a regularizer we can use a standard TensorFlow method.

.. code-block:: python

    def myregularizer(regularized_params):
        return tf.nn.l2_loss(regularized_params)

Add the regularizer and a regularization strength to the hyperparameters.

.. code-block:: python

    hyperparams = {...
                   'regularizer': myregularizer,
                   'regularization_strength': 0.5,
                   ...
                  }


Monitoring
++++++++++

If we also mark the circuit parameter for monitoring via

.. code-block:: python

    alpha = make_param(name='alpha', constant=0.2, regularize=True, monitor=True)

we can look at the monitored variable and other values during training with **TensorBoard**. For this you have to install TensorBoard, open a terminal and navigate it to the directory that contains the (newly created) folder "logsAUTO" or the name of your custom logging directory if you used one. In the terminal,  run the command ``tensorboard --logdir=logsAUTO``. This should return a link to a local server which can be opened in a browser. The browser window shows live updates during training.

.. note::

    Tensorboard shows also some default information, for example about the usage of the dataqueue, or the runtime per training step.

Play around with the 'regularization_strength' and see how a large value forces alpha to zero.


Custom logging
++++++++++++++

Since TensorFlow knows the name of tensors in the computational graph at all times, logging is even easier. Outside of :func:`circuit` (that is, anywhere in your code), create the log dictionary and pass it to the learner with the keyword ``tensors_to_log`` for training.

.. code-block:: python

    log = {'Prob': 'prob'}

    learner.train_circuit(steps=50, tensors_to_log=log)

The keys ``'Prob'`` and ``'Trace'`` are your choice, while ``'prob'`` and ``'trace'`` are names of tensors defined in :func:`circuit`.


Unsupervised learning
---------------------

Basic example
*************

The basic example for unsupervised learning looks the same as the numerical learner, except for

* We have to import TensorFlow instead of numpy
* Parameters are created in the :func:`circuit` function
* myloss and myregularizer are TensorFlow functions

Here is the entire code:

.. code-block:: python

    import strawberryfields as sf
    from strawberryfields.ops import *
    import numpy as np
    import tensorflow as tf
    from qmlt.tf import CircuitLearner
    from qmlt.tf.helpers import make_param
    from qmlt.helpers import sample_from_distr


    steps = 100


    def circuit():

        phi = make_param(name='phi', stdev=0.2, regularize=False)
        theta = make_param(name='theta', stdev=0.2, regularize=False)
        a = make_param(name='a', stdev=0.2,  regularize=True, monitor=True)
        rtheta = make_param(name='rtheta', stdev=0.2, regularize=False, monitor=True)
        r = make_param(name='r', stdev=0.2, regularize=True, monitor=True)
        kappa = make_param(name='kappa', stdev=0.2, regularize=True, monitor=True)

        eng, q = sf.Engine(2)

        with eng:
            BSgate(phi, theta) | (q[0], q[1])
            Dgate(a) | q[0]
            Rgate(rtheta) | q[0]
            Sgate(r) | q[0]
            Kgate(kappa) | q[0]

        state = eng.run('tf', cutoff_dim=7, eval=False)
        circuit_output = state.all_fock_probs()

        return circuit_output


    def myloss(circuit_output, X):
        probs = tf.gather_nd(params=circuit_output, indices=X)
        prob_total = tf.reduce_sum(probs, axis=0)
        return -prob_total


    def myregularizer(regularized_params):
        return tf.nn.l2_loss(regularized_params)


    X_train = np.array([[0, 1],
                        [0, 2],
                        [0, 3],
                        [0, 4]])

    hyperparams = {'circuit': circuit,
                   'task': 'unsupervised',
                   'optimizer': 'SGD',
                   'init_learning_rate': 0.1,
                   'loss': myloss,
                   'regularizer': myregularizer,
                   'regularization_strength': 0.1
                   }

    learner = CircuitLearner(hyperparams=hyperparams)

    learner.train_circuit(X=X_train, steps=steps)

    outcomes = learner.run_circuit()
    final_distribution = outcomes['outputs']

    for i in range(10):
        sample = sample_from_distr(distr=final_distribution)
        print("Fock state sample {}:{} \n".format(i, sample))


Layered circuit architectures
*****************************

Using layers is even easier than in the numerical case, because we can create tensors of multiple parameters directly with the :func:`~.tf.helpers.make_param` function, by defining ``shape=[depth]``. In the :func:`layer` function, we can call the gate parameter ``phi`` for the l'th layer by using ``phi[l]``.

.. code-block:: python

    depth = 5
    steps = 500

    def circuit():

        phi = make_param(name='phi', stdev=0.2, shape=[depth], regularize=False)
        theta = make_param(name='theta', stdev=0.2, shape=[depth], regularize=False)
        a = make_param(name='a', stdev=0.2, shape=[depth], regularize=True, monitor=True)
        rtheta = make_param(name='rtheta', stdev=0.2, shape=[depth], regularize=False, monitor=True)
        r = make_param(name='r', stdev=0.2, shape=[depth], regularize=True, monitor=True)
        kappa = make_param(name='kappa', stdev=0.2, shape=[depth], regularize=True, monitor=True)

        def layer(l):
            BSgate(phi[l], theta[l]) | (q[0], q[1])
            Dgate(a[l]) | q[0]
            Rgate(rtheta[l]) | q[0]
            Sgate(r[l]) | q[0]
            Kgate(kappa[l]) | q[0]

        eng, q = sf.Engine(2)

        with eng:
            for d in range(depth):
                layer(d)

        state = eng.run('tf', cutoff_dim=7, eval=False)
        circuit_output = state.all_fock_probs()

        return circuit_output


We use 500 steps for training, since optimization now happens in a space with a lot more dimension and is therefore harder.

Also try another optimizer, for example the Adam optimizer:

.. code-block:: python

        'optimizer': 'Adam'


If we print the parameters after training with the command

.. code-block:: python

    learner.get_circuit_parameters(only_print=True)

we see that the Adam optimizer creates additional parameters during training.



Supervised learning
-------------------


Basic example
*************

Do the usual imports:

.. code-block:: python

    import strawberryfields as sf
    from strawberryfields.ops import Dgate, BSgate
    import tensorflow as tf
    from qmlt.tf.helpers import make_param
    from qmlt.tf import CircuitLearner

We define a circuit that depends on a tensor of input features ``X``. The tensorflow backend can process data in batches (i.e., compute the output for multiple inputs in parallel). To use this, we have to get the number of inputs from ``X`` and pass it to the :meth:`eng.run` function's ``batch_size`` argument.

.. code-block:: python

    def circuit(X):
        phi = make_param('phi', constant=2.)

        eng, q = sf.Engine(2)

        with eng:
            Dgate(X[:, 0], 0.) | q[0]
            Dgate(X[:, 1], 0.) | q[1]
            BSgate(phi=phi) | (q[0], q[1])
            BSgate() | (q[0], q[1])

        num_inputs = X.get_shape().as_list()[0]
        state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)

        p0 = state.fock_prob([0, 2])
        p1 = state.fock_prob([2, 0])
        normalisation = p0 + p1 + 1e-10
        circuit_output = p1/normalisation

        return circuit_output


.. note::
   Instead of using the quantum circuit in batch mode, we could also use tensorflow's :func:`map_fn` function to compute the circuit for every element in ``X`` as in the numerical tutorial.

.. warning::

   Always add a tiny offset ``1e-10`` to a normalisation factor. Otherwise one might divide by a very small number or even zero, which leads to numerical instabilities.

.. note::

   We could use a softmax layer instead of normalising the two outputs in order to interpret it as a probabilistic outcome. However, since the output of the model is very small, the softmax function maps both outputs to a value that is close to ``0.5`` and the signal is very weak.

As a loss function, we use tensorflow's :func:`~.tf.losses.mean_squared_error` function.

.. code-block:: python

    def myloss(circuit_output, targets):
        return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)


Next, we make some data.

.. code-block:: python

   X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
   Y_train = [1, 1, 0, 0]
   X_test = [[0.25, 0.5], [0.5, 0.25]]
   Y_test = [1, 0]
   X_pred = [[0.4, 0.5], [0.5, 0.4]]


The function that defines how to get predictions from the outputs has to be coded in tensorflow.

.. code-block:: python

    def outputs_to_predictions(circuit_output):
        return tf.round(circuit_output)


The rest is equivalent to the corresponding numerical tutorial.

.. code-block:: python

    hyperparams = {'circuit': circuit,
                   'task': 'supervised',
                   'loss': myloss,
                   'optimizer': 'SGD',
                   'init_learning_rate': 0.5
                   }

    learner = CircuitLearner(hyperparams=hyperparams)

    learner.train_circuit(X=X_train, Y=Y_train, steps=100)

    test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                       outputs_to_predictions=outputs_to_predictions)
    print("\nPossible scores to print: {}".format(list(test_score.keys())))
    print("Accuracy on test set: ", test_score['accuracy'])
    print("Loss on test set: ", test_score['loss'])

    outcomes = learner.run_circuit(X=X_pred, outputs_to_predictions=outputs_to_predictions)

    print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
    print("Predictions for new inputs: {}".format(outcomes['predictions']))




Using an adaptive learning rate, printing, warm start and batch mode
*********************************************************************

These adaptations work exactly as in the numerical tutorial.
