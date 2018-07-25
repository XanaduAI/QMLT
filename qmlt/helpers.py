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
Helpers
========================================================

**Module name:** :mod:`qmlt.helpers`

.. currentmodule:: qmlt.helpers

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

Collection of helpers to set up an experiment with either the numerical or tf
circuit learner.

Summary
-------

.. autosummary::
    sample_from_distribution

Code details
------------
"""


import numpy as np


def sample_from_distribution(distribution):
    r"""
    Sample a Fock state from a nested probability distribution of Fock states.

    Args:
        distribution (ndarray): Nested array containing probabilities of Fock state.
          Fock state :math:`|i,j,k \rangle` is retrieved by ``distribution([i,j,k])``.
          Can be the result of :func:`state.all_fock_probs`.

    Return: List of photon numbers representing a Fock state.
    """

    distribution = np.array(distribution)
    cutoff = distribution.shape[0]
    num_modes = len(distribution.shape)

    probs_flat = np.reshape(distribution, (-1))
    indices_flat = np.arange(len(probs_flat))
    indices = np.reshape(indices_flat, [cutoff] * num_modes)
    sample_index = np.random.choice(indices_flat, p=probs_flat / sum(probs_flat))
    fock_state = np.asarray(np.where(indices == sample_index)).flatten()

    return fock_state
