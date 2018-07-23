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

Example of a simple supervised learning task with the numerical circuit learner.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss

steps = 100

# Create a parameter with an initial value of 2.
my_init_params = [make_param(name='phi', constant=2.)]


# Define the variational circuit and its output
def circuit(X, params):

    eng, q = sf.Engine(2)

    # Since X is a batch of data, define a circuit for a single input
    # If you use the tf backend, you can pass batches into gates
    # like in the supervised tf learner example.
    def single_input_circuit(x):

        eng.reset()
        with eng:
            Dgate(x[0], 0.) | q[0]
            Dgate(x[1], 0.) | q[1]
            BSgate(phi=params[0]) | (q[0], q[1])
            BSgate() | (q[0], q[1])
        state = eng.run('fock', cutoff_dim=10, eval=True)

        # Define the output as the probability of measuring |0,2> as opposed to |2,0>
        p0 = state.fock_prob([0, 2])
        p1 = state.fock_prob([2, 0])
        normalization = p0 + p1 + 1e-10
        output = p1 / normalization
        return output

    # Apply the single circuit to every input in the batch
    circuit_output = [single_input_circuit(x) for x in X]

    return circuit_output


# Define a loss function that takes the outputs of the variational circuit
# and compares them to the targets
def myloss(circuit_output, targets):
    # We use the square loss function provided by MLT
    return square_loss(outputs=circuit_output, targets=targets)


# Define how to translate the outputs of the circuit into model predictions
def outputs_to_predictions(circuit_output):
    return round(circuit_output)


# Generate some data
X_train = np.array([[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]])
Y_train = np.array([1., 1., 0., 0.])
X_test = np.array([[0.25, 0.5], [0.5, 0.25]])
Y_test = np.array([1., 0.])
X_pred = np.array([[0.4, 0.5], [0.5, 0.4]])

# Set the hyperparameters of the model and the training algorithm
hyperparams = {'circuit': circuit,
               'init_circuit_params': my_init_params,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.5
               }

# Create the learner
learner = CircuitLearner(hyperparams=hyperparams)

# Train the learner
learner.train_circuit(X=X_train, Y=Y_train, steps=steps)

# Evaluate the score of a test set
test_score = learner.score_circuit(X=X_test, Y=Y_test, outputs_to_predictions=outputs_to_predictions)
# The score_circuit() function returns a dictionary of different metrics.
print("\nPossible scores to print: {}".format(list(test_score.keys())))
# We select the accuracy and loss.
print("Accuracy on test set: {}".format(test_score['accuracy']))
print("Loss on test set: {}".format(test_score['loss']))

outcomes = learner.run_circuit(X=X_pred, outputs_to_predictions=outputs_to_predictions)
# The run_circuit() function returns a dictionary of different outcomes.
print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
# We select the predictions
print("Predictions for new inputs: {}".format(outcomes['predictions']))




