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

Example of a simple unsupervised learning task with the numerical circuit learner.

This example fails to learn the structure of the data, namely to have zero photons in the first mode,
irrespective of the second mode.

"""

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
from qmlt.helpers import sample_from_distribution


# Create some parameters. Mark some of them to be regularized.
my_params = [
    make_param(name='phi', stdev=0.2, regularize=False),
    make_param(name='theta', stdev=0.2, regularize=False),
    make_param(name='a', stdev=0.2, regularize=True),
    make_param(name='rtheta', stdev=0.2, regularize=False),
    make_param(name='r', stdev=0.2, regularize=True),
    make_param(name='kappa', stdev=0.2, regularize=True)
]


# Define the variational circuit and its output
def circuit(params):

    eng, q = sf.Engine(2)

    with eng:
        BSgate(params[0], params[1]) | (q[0], q[1])
        Dgate(params[2]) | q[0]
        Rgate(params[3]) | q[0]
        Sgate(params[4]) | q[0]
        Kgate(params[5]) | q[0]

    state = eng.run('fock', cutoff_dim=7)
    circuit_output = state.all_fock_probs()

    return circuit_output


# Define a loss function that maximises the probabilities of the states we want to learn
def myloss(circuit_output, X):
    probs = [circuit_output[x[0], x[1]] for x in X]
    prob_total = sum(np.reshape(probs, -1))
    return -prob_total


def myregularizer(regularized_params):
    return l2(regularized_params)


# Generate some training data.
# The goal is to learn that the first mode contains no photons.
X_train = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4]])

# Set the hyperparameters of the model and the training algorithm
hyperparams = {'circuit': circuit,
               'init_circuit_params': my_params,
               'task': 'unsupervised',
               'optimizer': 'Nelder-Mead',
               'loss': myloss,
               'regularizer': myregularizer,
               'regularization_strength': 0.1,
               'print_log': True,
               'log_every': 100
               }

# Create the learner
learner = CircuitLearner(hyperparams=hyperparams)

# Train the learner
learner.train_circuit(X=X_train, steps=500)

# Get the final distribution, which is the circuit output
outcomes = learner.run_circuit()
final_distribution = outcomes['outputs']


# Use a helper function to sample fock states from this state.
# They should show a similar distribution to the training data
for i in range(10):
    sample = sample_from_distribution(distribution=final_distribution)
    print("Fock state sample {}:{}".format(i, sample))



