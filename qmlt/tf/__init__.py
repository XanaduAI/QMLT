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
TF Quantum Circuit Learner
========================================================

**Module name:** :mod:`qmlt.tf`

.. currentmodule:: qmlt.tf

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

This module contains a class to train models for machine learning and optimization based on variational quantum circuits.
The class extends TensorFlow's ``tf.estimator`` class, and adapts it to unsupervised learning and optimization tasks. It
hides the complexity of defining input functions, hooks and configurations.

The user defines a function that computes the outputs of the variational circuit, as well as the training objective,
and specifies the model and training hyperparameters.

There are three basic functionalities. The circuit can be trained, run with the current parameters, and scored.

The TensorFlow learner module has been designed for the training of continuous-variable circuits written in StrawberryFields or
BlackBird using the 'tf' backend only, but is in principle able to train any user-provided model coded in tensorflow.


CircuitLearner class
---------------------

.. currentmodule:: qmlt.tf.CircuitLearner

.. autosummary::
    get_circuit_parameters


Helper methods
--------------

.. currentmodule:: qmlt.tf

.. autosummary::
    qcv_model_fn
    make_input_fn
    check
    check_X
    check_Y
    check_steps
    check_batch_size
    check_shuffle

Code details
------------
"""

from .learner import (CircuitLearner,
                      _qcv_model_fn as qcv_model_fn,
                      _check as check,
                      _check_X as check_X,
                      _check_Y as check_Y,
                      _check_steps as check_steps,
                      _check_batch_size as check_batch_size,
                      _check_shuffle as check_shuffle,
                      _make_input_fn as make_input_fn)

__all__ = ['CircuitLearner', 'qcv_model_fn', 'make_input_fn', 'check', 'check_X', 'check_Y', 'check_steps', 'check_batch_size', 'check_shuffle']
