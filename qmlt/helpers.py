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
    sample_from_distr

Code details
------------
"""


import numpy as np


def sample_from_distr(distr):
    r"""
    Sample a Fock state from a nested probability distribution of Fock states.

    Args:
        distr (ndarray): Nested array containing probabilities of Fock state.
          Fock state :math:`|i,j,k \rangle` is retrieved by ``distr([i,j,k])``.
          Can be the result of :func:`state.all_fock_probs`.

    Return: List of photon numbers representing a Fock state.
    """

    distr = np.array(distr)
    cutoff = distr.shape[0]
    num_modes = len(distr.shape)

    probs_flat = np.reshape(distr, (-1))
    indices_flat = np.arange(len(probs_flat))
    indices = np.reshape(indices_flat, [cutoff] * num_modes)
    smpl_index = np.random.choice(indices_flat, p=probs_flat / sum(probs_flat))
    fock_state = np.asarray(np.where(indices == smpl_index)).flatten()

    return fock_state
