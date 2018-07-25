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
.. currentmodule:: qmlt.examples.numerical

.. code-author:: Maria Schuld <maria@xanadu.ai>

We revisit the example of a simple supervised learning task with the numerical circuit learner and
introduce adaptive learning rate, printing, warm start and batch mode.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss

steps = 200
batch_size = 2

my_init_params = [make_param(constant=2.)]


def circuit(X, params):

    eng, q = sf.Engine(2)

    def single_input_circuit(x):

        eng.reset()
        with eng:
            Dgate(x[0], 0.) | q[0]
            Dgate(x[1], 0.) | q[1]
            BSgate(phi=params[0]) | (q[0], q[1])
            BSgate() | (q[0], q[1])
        state = eng.run('fock', cutoff_dim=10, eval=True)

        p0 = state.fock_prob([0, 2])
        p1 = state.fock_prob([2, 0])
        normalization = p0 + p1 + 1e-10
        output = p1 / normalization
        return output

    circuit_output = [single_input_circuit(x) for x in X]

    return circuit_output


def myloss(circuit_output, targets):
    return square_loss(outputs=circuit_output, targets=targets)


def outputs_to_predictions(circuit_output):
    return round(circuit_output)


X_train = np.array([[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]])
Y_train = np.array([1., 1., 0., 0.])
X_test = np.array([[0.25, 0.5], [0.5, 0.25]])
Y_test = np.array([1., 0.])
X_pred = np.array([[0.4, 0.5], [0.5, 0.4]])

# There are some changes here:
# We decay the learning rate by a factor 1/(1-decay*step) in each step.
# When indicating a batch_size, we train_circuit in every step with only a (randomly selected) batch of the data.
# We also print out the results every 10th step.
# Finally, we choose a warm start. This loads the final parameters from the previous training. You can see
# that the global step starts where it ended the last time you ran the script.
hyperparams = {'circuit': circuit,
               'init_circuit_params': my_init_params,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.5,
               'decay': 0.01,
               'log_every': 10,
               'warm_start': False #Set this to True after first run
               }

# Create the learner
learner = CircuitLearner(hyperparams=hyperparams)

# Train the learner
learner.train_circuit(X=X_train, Y=Y_train, steps=steps, batch_size=batch_size)

# Evaluate the score of a test set
test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
# The score_circuit() function returns a dictionary of different metrics. We select the accuracy and loss.
print("\nAccuracy on test set: {}".format(test_score['accuracy']))
print("Loss on test set: {}".format(test_score['loss']))

# Predict the labels of the new inputs
predictions = learner.run_circuit(X=X_pred,
                                  outputs_to_predictions=outputs_to_predictions)
print("\nPredictions for new inputs: ", predictions['outputs'])




