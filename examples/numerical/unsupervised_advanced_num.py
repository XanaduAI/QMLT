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

We revisit the example of a simple unsupervised learning task with the numerical circuit learner and
introduce layered circuit architectures.

This time the circuit learns a quantum state where Fock
states with zero photons in the first mode have a high measurement probablity.

"""

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
from qmlt.numerical.helpers import sample_from_distribution


# Number of layers
depth = 5
steps = 500

# This time we use a dynamic way to create parameters for each layer
my_params = []

for i in range(depth):
    my_params.append(make_param(name='phi_' + str(i), stdev=0.2, regularize=False))
    my_params.append(make_param(name='theta_' + str(i), stdev=0.2, regularize=False))
    my_params.append(make_param(name='a_'+str(i), stdev=0.2, regularize=True, monitor=True))
    my_params.append(make_param(name='rtheta_'+str(i), stdev=0.2, regularize=False, monitor=True))
    my_params.append(make_param(name='r_'+str(i), stdev=0.2, regularize=True, monitor=True))
    my_params.append(make_param(name='kappa_'+str(i), stdev=0.2, regularize=True, monitor=True))


def circuit(params):

    # Reshape to access parameter of a layer easier
    params = np.reshape(params, (depth, 6))

    # We define the architecture of a single layer.
    def layer(i):
        BSgate(params[i, 0], params[i, 1]) | (q[0], q[1])
        Dgate(params[i, 2]) | q[0]
        Rgate(params[i, 3]) | q[0]
        Sgate(params[i, 4]) | q[0]
        Kgate(params[i, 5]) | q[0]

    eng, q = sf.Engine(2)

    with eng:
        # Build the circuit of 'depth' layers
        for d in range(depth):
            layer(d)

    state = eng.run('fock', cutoff_dim=7)
    circuit_output = state.all_fock_probs()

    return circuit_output


def myloss(circuit_output, X):
    circuit_output = np.array(circuit_output)
    probs = [circuit_output[x[0], x[1]] for x in X]
    prob_total = sum(np.reshape(probs, -1))
    return -prob_total


def myregularizer(regularized_params):
    return l2(regularized_params)


X_train = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4]])

hyperparams = {'circuit': circuit,
               'init_circuit_params': my_params,
               'task': 'unsupervised',
               'optimizer': 'Nelder-Mead',
               'init_learning_rate': 0.1,
               'loss': myloss,
               'regularizer': myregularizer,
               'regularization_strength': 0.1,
               'log_every': 100
               }

learner = CircuitLearner(hyperparams=hyperparams)

learner.train_circuit(X=X_train, steps=steps)

outcomes = learner.run_circuit()
final_distribution = outcomes['outputs']

for i in range(10):
    sample = sample_from_distribution(distribution=final_distribution)
    print("Fock state sample {}:{} \n".format(i, sample))

# Note: the learner really generalises. Sometimes (albeit rarely) it will
# sample Fock state |0, 5> which it has never seen during training
