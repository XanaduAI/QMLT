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

Example of a simple supervised learning task with the tensorflow circuit learner.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner


steps = 100


# Define the variational circuit and its output.
def circuit(X):
    # Create a parameter with an initial value of 2.
    phi = make_param('phi', constant=2.)

    eng, q = sf.Engine(2)

    with eng:
        # Note that we are feeding 1-d tensors into gates, not scalars!
        Dgate(X[:, 0], 0.) | q[0]
        Dgate(X[:, 1], 0.) | q[1]
        BSgate(phi=phi) | (q[0], q[1])
        BSgate() | (q[0], q[1])

    # We have to tell the engine how big the batches (first dim of X) are
    # which we feed into gates
    num_inputs = X.get_shape().as_list()[0]
    state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)

    # Define the output as the probability of measuring |0,2> as opposed to |2,0>
    p0 = state.fock_prob([0, 2])
    p1 = state.fock_prob([2, 0])
    normalisation = p0 + p1 + 1e-10
    circuit_output = p1/normalisation

    return circuit_output


# Define a loss function that takes the outputs of the variational circuit
# and compares them to the targets
def myloss(circuit_output, targets):
    # Use tensorflow's predefined loss
    return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)


# Define how to translate the outputs of the circuit into model predictions
def outputs_to_predictions(circuit_output):
    return tf.round(circuit_output)


# Generate some data
X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
Y_train = [1, 1, 0, 0]
X_test = [[0.25, 0.5], [0.5, 0.25]]
Y_test = [1, 0]
X_pred = [[0.4, 0.5], [0.5, 0.4]]


# Set the hyperparameters of the model and the training algorithm
# Due to the workings of tensorflow, we have to define the batch size for every
# mode of operation. The 'train_circuit' batch_size indicates the number of
# samples used in every training step.
# The 'eval' and 'infer' "batch_sizes" are the number of data points we plan
# to feed into the score() and run_circuit() methods.
hyperparams = {'circuit': circuit,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.5,
               'print_log': True}

# Create a learner
learner = CircuitLearner(hyperparams=hyperparams)

# Train the learner
learner.train_circuit(X=X_train, Y=Y_train, steps=steps)

# Get the accuracy and loss for the test data
test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
# The score_circuit() function returns a dictionary of different metrics.
print("\nPossible scores to print: {}".format(list(test_score.keys())))
# We select the accuracy and loss.
print("Accuracy on test set: ", test_score['accuracy'])
print("Loss on test set: ", test_score['loss'])

outcomes = learner.run_circuit(X=X_pred, outputs_to_predictions=outputs_to_predictions)

# The run_circuit() function returns a dictionary of different outcomes.
print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
# We select the predictions
print("Predictions for new inputs: {}".format(outcomes['predictions']))
