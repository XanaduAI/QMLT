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

Example of a simple optimization task with the tensorflow circuit learner.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate
import tensorflow as tf
from qmlt.tf import CircuitLearner
from qmlt.tf.helpers import make_param


# Define the variational circuit and its output
def circuit():

    # Create a parameter with an initial value of 0.1
    alpha = make_param(name='alpha', constant=0.1)

    eng, q = sf.Engine(1)

    with eng:
        Dgate(alpha) | q[0]

    state = eng.run('tf', cutoff_dim=7, eval=False)

    # As the output we take the probability of measuring one photon in the mode
    prob = state.fock_prob([1])
    circuit_output = tf.identity(prob, name="prob")
    return circuit_output


# Define a loss function on the outputs of circuit().
# We use the negative probability of measuring |1>
# so that minimization increases the probability.
def myloss(circuit_output):
    return -circuit_output


# Set the hyperparameters of the model and the training algorithm
hyperparams = {'circuit': circuit,
               'task': 'optimization',
               'optimizer': 'SGD',
               'init_learning_rate': 0.1,
               'loss': myloss
               }

# Create the learner
learner = CircuitLearner(hyperparams=hyperparams)

# Train the learner
learner.train_circuit(steps=50)

