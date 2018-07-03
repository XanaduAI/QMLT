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

**Module name:** :mod:`qmlt.numerical.helpers`

.. currentmodule:: qmlt.numerical.helpers

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

Collection of helpers to set up an experiment with the numerical circuit learner.

Summary
-------

.. autosummary::
    make_param

Code details
------------
"""
# pylint: disable=too-many-arguments
import numpy as np


def make_param(name=None, stdev=None, mean=0., interval=None, constant=None,
               regularize=False, monitor=False, seed=None):
    r"""Return a circuit parameter.

    Args:
        name (str): name of the variable
        stdev (float): If not None, initialise from normal distribution.
        mean (float): If stdev is not None, use this mean for normal
          distribution.
        interval (list of length 2): If stdev is None and interval is not None,
          initialise from random value sampled
          uniformly from this interval.
        constant (float): If stdev and interval are both None and constant
          is not None, use this as an initial value.
          If constant is also None, use 0 as an initial value (not recommended!).
        regularize (boolean): If true, regularize this parameter.
        monitor (boolean): If true, monitor this variable for plotting.

    Return:
        Dictionary: Dictionary representing a circuit parameter.
    """

    if seed is not None:
        np.random.seed(seed)

    if stdev is not None:
        var = {'name': name,
               'val': np.random.normal(loc=mean, scale=stdev),
               'regul': regularize,
               'monitor': monitor}

    elif interval is not None:
        var = {'name': name,
               'val': interval[0]+np.random.random()*(interval[1]-interval[0]),
               'regul': regularize,
               'monitor': monitor}

    elif constant is not None:
        var = {'name': name,
               'val': constant,
               'regul': regularize,
               'monitor': monitor}

    else:
        var = {'name': name,
               'val': 0.1*np.random.random(),
               'regul': regularize,
               'monitor': monitor}
    return var
