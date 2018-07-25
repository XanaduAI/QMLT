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

We revisit the example of a simple optimization task with the numerical circuit learner and
introduce regularisation, custom logging and monitoring/plotting,
and look at the final parameters.

"""

import strawberryfields as sf
from strawberryfields.ops import Dgate
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2

# This time we want to keep the parameter small via regularization and monitor its evolution
# By logging it into a file and plotting it
my_init_params = [make_param(name='alpha', constant=0.1, regularize=True, monitor=True)]


def circuit(params):

    eng, q = sf.Engine(1)

    with eng:
        Dgate(params[0]) | q[0]

    state = eng.run('fock', cutoff_dim=7)

    circuit_output = state.fock_prob([1])
    trace = state.trace()

    # Log the trace of the state to check if it is 1
    log = {'Prob': circuit_output,
           'Trace': trace}

    # The second return value can be an optional log dictionary
    # of one or more values
    return circuit_output, log


def myloss(circuit_output):
    return -circuit_output


# We have to define a regularizer function that penalises large parameters that we marked to be regularized
def myregularizer(regularized_params):
    # The function is imported from the regularizers module and simply computes the squared Euclidean length of the
    # vector of all parameters
    return l2(regularized_params)


# We add the regularizer function to the model
# The strength of regularizer is regulated by the
# hyperparameter 'regularization_strength'.
# Setting 'plot' to an integer automatically plots some default values
# as well as the monitored circuit parameters. (Requires matplotlib).
hyperparams = {'circuit': circuit,
               'init_circuit_params': my_init_params,
               'task': 'optimization',
               'loss': myloss,
               'regularizer': myregularizer,
               'regularization_strength': 0.5,
               'optimizer': 'SGD',
               'init_learning_rate': 0.1,
               'log_every': 1,
               'plot': True
               }


learner = CircuitLearner(hyperparams=hyperparams)

learner.train_circuit(steps=50)

# Print out the final parameters
final_params = learner.get_circuit_parameters()
# final_params is a dictionary
for name, value in final_params.items():
    print("Parameter {} has the final value {}.".format(name, value))

# Look in the 'logsNUM' directory, there should be a file called 'log.csv' that records what happened to alpha
# during training. Play around with the 'regularization_strength' and see how a large strength forces alpha to zero.

