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
Numerical Quantum Circuit Learner
========================================================

**Module name:** :mod:`qmlt.numerical`

.. currentmodule:: qmlt.numerical

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

This module contains a class to train models for machine learning and optimization based on variational quantum circuits.
The optimization is executed by scipy's numerical optimisation library. The user defines a function that computes
the outputs of the variational circuit, as well as the training objective, and specifies the model and training
hyperparameters.

There are three basic functionalities. The circuit can be trained, run with the current parameters, and scored.

The numerical learner module has been designed for the training of continuous-variable circuits written in StrawberryFields or
BlackBird (using any backend), but is in principle able to train any user-provided model coded in python.

.. note::
    Numerical differentiation is not robust, which means that some models fail to be trained. For example, the approximations
    of gradients for gradient-based methods are not precise enough to find the steepest descent in plateaus of the
    optimization landscape. This can sometimes be rectified by choosing good hyperparameters, but ultimately poses a limit
    to training quantum circuits with numerical methods.


CircuitLearner class
---------------------

.. currentmodule:: qmlt.numerical.CircuitLearner

.. autosummary::
    train_circuit
    run_circuit
    score_circuit
    get_circuit_parameters


Helper methods
--------------

.. currentmodule:: qmlt.numerical

.. autosummary::
    check
    check_X
    check_Y
    check_steps
    check_batch_size
    check_logs

Code details
------------
"""

from .learner import (CircuitLearner,
                      _check as check,
                      _check_X as check_X,
                      _check_Y as check_Y,
                      _check_steps as check_steps,
                      _check_batch_size as check_batch_size,
                      _check_logs as check_logs)

__all__ = ['CircuitLearner', 'check', 'check_X', 'check_Y', 'check_steps', 'check_batch_size', 'check_logs']
