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
Losses
========================================================

**Module name:** :mod:`qmlt.tf.losses`

.. currentmodule:: qmlt.tf.losses

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

A collection of loss functions for tensorflow that are specific to quantum machine learning and optimization.


Summary
-------

.. autosummary::
    trace_distance
    expectation

Code details
------------
"""


import tensorflow as tf


def trace_distance(rho, sigma):
    r""" Trace distance :math:`\frac{1}{2}\tr \{ \sqrt{ (\rho - \sigma})^2  \}` between quantum states :math:`\rho` and :math:`\sigma`.

    The inputs and outputs are tensors of dtype float, and all computations support automatic differentiation.

    Args:
        rho (tf.Tensor): 2-dimensional Hermitian matrix representing state :math:`\rho`.
        sigma (tf.Tensor): 2-dimensional Hermitian matrix of the same dimensions and dtype as rho,
            representing state :math:`\sigma`.

    Returns:
        tf.Tensor: Returns the scalar trace distance.
    """

    if rho.shape != sigma.shape:
        raise ValueError("Cannot compute the trace distance if inputs have"
                         " different shapes {} and {}".format(rho.shape, sigma.shape))

    diff = rho - sigma
    eig = tf.self_adjoint_eigvals(diff)
    abs_eig = tf.abs(eig)
    return 0.5*tf.real(tf.reduce_sum(abs_eig))


def expectation(rho, operator):
    r""" Expectation value :math:`\tr\{ \rho O\}` of operator :math:`O` with respect to the quantum state :math:`\rho`.

    The inputs and outputs are tensors of dtype float, and all computations support automatic differentiation.


    Args:
        rho (tf.Tensor) : 2-dimensional Hermitian tensor representing state :math:`\rho`.
        operator (tf.Tensor):  2-dimensional Hermitian tensor of the same dimensions and dtype as rho.

    Returns:
        tf.Tensor: Returns the scalar expectation value.

    """
    if rho.shape != operator.shape:
        raise ValueError("Cannot compute expectation value if rho and operator have"
                         " different shapes {} and {}".format(rho.shape, operator.shape))
    if len(rho.shape) != 2:
        raise ValueError("Expectation loss expects a 2-d array representing a density matrix.")

    exp = tf.real(tf.trace(tf.matmul(rho, operator)))
    return exp
