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

**Module name:** :mod:`qmlt.numerical.losses`

.. currentmodule:: qmlt.numerical.losses

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

A collection of loss functions for numpy.

Summary
-------

.. autosummary::
    trace_distance
    expectation
    square_loss
    _softmax
    cross_entropy_with_softmax

Code details
------------
"""

import numpy as np


def trace_distance(rho, sigma):
    r"""Trace distance :math:`\frac{1}{2}\tr \{ \sqrt{ (\rho - \sigma)^2}  \}` between quantum states :math:`\rho` and :math:`\sigma`.

    Args:
        rho (ndarray or list): 2-dimensional square matrix representing the state :math:`\rho`.
        sigma (ndarray or list): 2-dimensional square matrix of the same dimensions and dtype as rho,
            representing the state :math:`\sigma`

    Returns:
        float: Scalar trace distance.

    """

    rho = np.array(rho)
    sigma = np.array(sigma)

    if rho.shape != sigma.shape:
        raise ValueError("Cannot compute the trace distance if inputs have"
                         " different shapes {} and {}".format(rho.shape, sigma.shape))
    if rho.ndim != 2:
        raise ValueError("Trace distance loss expects 2-d arrays representing density matrices.")

    diffs = rho - sigma
    eigvals = np.linalg.eigvals(diffs)
    return 0.5 * sum(np.absolute(eigvals))


def expectation(rho, operator):
    r""" Expectation value :math:`\tr\{ \rho O\}` of operator :math:`O` with respect to the quantum state :math:`\rho`.


    Args:
        rho (ndarray or list): 2-dimensional array representing the state :math:`\rho`.
        operator (ndarray or list): 2-dimensional array of the same dimensions and dtype as rho,
            representing the operator :math:`O`

    Returns:
        float: Scalar expectation value.

    """

    rho = np.array(rho)
    operator = np.array(operator)

    if rho.shape != operator.shape:
        raise ValueError("Cannot compute expectation value if rho and operator have"
                         " different shapes {} and {}".format(rho.shape, operator.shape))
    if rho.ndim != 2:
        raise ValueError("Expectation loss expects a 2-d array representing a density matrix.")

    exp = np.trace(rho@operator)

    if np.imag(exp) > 1e-5:
        raise ValueError("Expectation value has a non-negligible imaginary contribution."
                         "Something went wrong.")

    return exp


def square_loss(outputs, targets):
    r"""Mean squared loss :math:`0.5 \sum\limits_{m=1}^M  |y^m - t^m|^2` between outputs :math:`y^m` and
    targets :math:`t^m` for :math:`m = 1,...,M`.

    Args:
        outputs (ndarray or list): array of dimension M x 1 containing the 1-dimensional outputs.
        targets (ndarray or list): array of the same dimension and type as outputs, containing the targets.

    Returns:
        float: Scalar mean squared loss.

    """

    outputs = np.array(outputs)
    targets = np.array(targets)

    if outputs.shape != targets.shape:
        raise ValueError("Cannot compute squared loss if outputs and targets have"
                         " different shapes {} and {}".format(outputs.shape, targets.shape))

    if outputs.ndim > 2:
        raise ValueError("Mean squared loss expects 1-d outputs, dimension of current outputs"
                         " is {}.".format(outputs.ndim - 1))

    diff = outputs - targets
    res = 0.5*sum(np.dot(d, d) for d in diff)
    return res


def _softmax(logits):
    r"""Softmax function, turns a vector of real values into a vector of probabilities.

    Args:
        logits (ndarray 1-d): Real 1-d vector of model outputs

    Returns:
        ndarray: Vector of probabilities

    """
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def cross_entropy_with_softmax(outputs, targets):
    r"""
    Cross-entropy loss that measures the probability error in discrete classification tasks (with mutually exclusive classes).
    Useful for one-hot-encoded vectors.

    Args:
        outputs (ndarray): Real 2-dim array representing a batch of model outputs. Also called logits.
        targets (ndarray): Real 2-dim array representing a batch of target outputs.

    Return:
        float: Scalar loss.
    """
    probs = [_softmax(outp) for outp in outputs]
    terms = np.array([y*np.log(p) for p, y in zip(probs, targets)])
    loss = (-1)*np.sum(terms)
    return loss
