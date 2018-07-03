#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
.. currentmodule:: qmlt.examples.tf

.. code-author:: Maria Schuld <maria@xanadu.ai>

We revisit the example of a simple supervised learning task with the tensorflow circuit learner
and introduce adaptive learning rate, printing, warm start and batch mode.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import tensorflow as tf

from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner


steps = 100
batch_size = 2


def circuit(X):

    phi = make_param('phi', constant=2., monitor=True)

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


def myloss(circuit_output, targets):
    return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)


def outputs_to_predictions(outpt):
    return tf.round(outpt)


X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
Y_train = [1, 1, 0, 0]
X_test = [[0.25, 0.5], [0.5, 0.25]]
Y_test = [1, 0]
X_pred = [[0.4, 0.5], [0.5, 0.4]]


# There are some changes here:
# We decay the learning rate by a factor 1/(1-decay*step) in each step.
# We train_circuit with batches of 2 training inputs (instead of the full batch).
# We also print out the results every 10th step.
# Finally, you can set 'warm start': True to continue previosu training.
# (MAKE SURE YOU RUN THE SAME SCRIPT ONCE WITH A COLD START,
# ELSE YOU GET ERRORS WHEN LOADING THE MODEL!).
# This loads the final parameters from the previous training. You can see
# that the global step starts where it ended the last time you ran the script.
hyperparams = {'circuit': circuit,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.5,
               'decay': 0.1,
               'print_log': True,
               'warm_start': False,
               'log_every': 10
               }

learner = CircuitLearner(hyperparams=hyperparams)

learner.train_circuit(X=X_train, Y=Y_train, steps=steps, batch_size=batch_size)

test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
# The score_circuit() function returns a dictionary of different metrics.
print("\nPossible scores to print: {}".format(list(test_score.keys())))
# We select the accuracy and loss.
print("Accuracy on test set: ", test_score['accuracy'])
print("Loss on test set: ", test_score['loss'])

outcomes = learner.run_circuit(X=X_pred,
                               outputs_to_predictions=outputs_to_predictions)
# The run_circuit() function returns a dictionary of different outcomes.
print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
# We select the predictions
print("Predictions for new inputs: {}".format(outcomes['predictions']))
