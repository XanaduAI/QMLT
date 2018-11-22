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

We revisit the example of a simple unsupervised learning task with the tf circuit learner and
introduce layered circuit architectures.

This time the circuit learns a quantum state where Fock
states with zero photons in the first mode have a high measurement probablity.

"""

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import tensorflow as tf
from qmlt.tf import CircuitLearner
from qmlt.tf.helpers import make_param
from qmlt.numerical.helpers import sample_from_distribution


# Number of layers
depth = 5
steps = 500


def circuit():

    # Create parameters for 'depth' number of layers.
    # Mark some of them to be regularized and some to be monitored.
    phi = make_param(name='phi', stdev=0.2, shape=[depth], regularize=False)
    theta = make_param(name='theta', stdev=0.2, shape=[depth], regularize=False)
    a = make_param(name='a', stdev=0.2, shape=[depth], regularize=True, monitor=True)
    rtheta = make_param(name='rtheta', stdev=0.2, shape=[depth], regularize=False, monitor=True)
    r = make_param(name='r', stdev=0.2, shape=[depth], regularize=True, monitor=True)
    kappa = make_param(name='kappa', stdev=0.2, shape=[depth], regularize=True, monitor=True)

    # Define a single layer of gates
    def layer(l):
        BSgate(phi[l], theta[l]) | (q[0], q[1])
        Dgate(a[l]) | q[0]
        Rgate(rtheta[l]) | q[0]
        Sgate(r[l]) | q[0]
        Kgate(kappa[l]) | q[0]

    eng, q = sf.Engine(2)

    with eng:
        # Make depth layers
        for d in range(depth):
            layer(d)

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

# Use a helper function to sample fock states from this state.
# They should show a similar distribution to the training data.
# In case of tensor data use sample_from_distribution from qmlt.tf.helpers
for i in range(10):
    sample = sample_from_distribution(distribution=final_distribution)
    print("Fock state sample {}:{} \n".format(i, sample))

# Note: the learner really generalises. Sometimes (albeit rarely) it will
# sample Fock state |0, 5> which it has never seen during training
