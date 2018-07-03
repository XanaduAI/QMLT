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

We revisit the example of a simple optimization task with the tensorflow circuit learner and
introduce regularisation, custom logging and monitoring,
and look at the final parameters.


"""

import strawberryfields as sf
from strawberryfields.ops import Dgate
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner


def circuit():

    # This time we want to keep the parameter small via regularization and visualize its evolution in tensorboard
    alpha = make_param(name='alpha', constant=0.1, regularize=True, monitor=True)
    eng, q = sf.Engine(1)

    with eng:
        Dgate(alpha) | q[0]

    state = eng.run('tf', cutoff_dim=7, eval=False)

    circuit_output = state.fock_prob([1])

    # The identity() function allows us to give this tensor a name
    # which we can refer to below
    circuit_output = tf.identity(circuit_output, name="prob")
    trace = tf.identity(state.trace(), name='trace')

    return circuit_output


def myloss(circuit_output):
    return -circuit_output


# We have to define a regularizer function that penalises large parameters that we marked to be regularized
def myregularizer(regularized_params):
    # The function is imported from tensorflow and simply computes the squared Euclidean length of the
    # vector of all parameters
    return tf.nn.l2_loss(regularized_params)


# We add the regularizer function to the model
# The strength of regularizer is regulated by the
# hyperparameter 'regularization_strength'.
hyperparams = {'circuit': circuit,
               'task': 'optimization',
               'loss': myloss,
               'regularizer': myregularizer,
               'regularization_strength': 0.5,
               'optimizer': 'SGD',
               'init_learning_rate': 0.1
               }

learner = CircuitLearner(hyperparams=hyperparams)

# Define the tensors we want displayed in the training log that gets printed,
# and a name to display it.
log = {'Prob': 'prob',
       'Trace': 'trace'}

learner.train_circuit(steps=50, tensors_to_log=log)

# Print out the final parameters
final_params = learner.get_circuit_parameters()
# final_params is a dictionary
for name, value in final_params.items():
    print("\nFinal parameter {} has the value {}.".format(name, value))

# To monitor the training, install tensorboard, navigate with a terminal to the directory that contains
# the newly created folder "logAUTO" and run "tensorboard --logdir=logAUTO". This should return a link
# which can be opened in a browser.

# You can track the changes in the variable alpha. Tensorboard gets live updated if you rerun this script.
# Play around with the 'regularization_strength' and see how a large value forces alpha to zero.
