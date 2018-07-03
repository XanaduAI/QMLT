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
Regularizers
========================================================

**Module name:** :mod:`qmlt.numerical.regularizers`

.. currentmodule:: qmlt.numerical.regularizers

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

A collection of regularizers to facilitate experiments with the numerical circuit learner.

Summary
-------

.. autosummary::
	l2
	l1


Code details
------------

"""

import numpy as np


def l2(circuit_params):
    r"""L2 regulariser :math:`0.5 \sum_{i=1}^N w_i^2` for a vector :math:`w = (w_1,...,w_N)` of circuit parameters.

    Args:
        circuit_params (ndarray): 1-d array containing the values of the circuit parameters to regularize.

    Returns:
        float: Scalar l2 loss.

    """
    circuit_params = np.array(circuit_params)

    if circuit_params.ndim > 1:
        raise ValueError("Regulariser expects a 1-dimensional array, got {} dimensions".format(circuit_params.ndim))

    return 0.5*np.dot(circuit_params, circuit_params)


def l1(circuit_params):
    r"""L1 regulariser :math:`\sum_{i=1}^N |w_i|` for a vector :math:`w = (w_1,...,w_N)` of circuit parameters.

    Args:
        circuit_params (ndarray): 1-d array containing the values of the circuit parameters to regularize.

    Returns:
        float: Scalar l1 loss.

    """

    circuit_params = np.array(circuit_params)

    if circuit_params.ndim > 1:
        raise ValueError("Regulariser expects a 1-dimensional array, got {} dimensions".format(circuit_params.ndim))

    return np.sum(np.absolute(circuit_params))
