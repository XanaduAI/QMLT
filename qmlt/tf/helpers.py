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
# pylint: disable=too-many-arguments,not-context-manager,too-many-branches
"""
Helpers
========================================================

**Module name:** :mod:`qmlt.tf.helpers`

.. currentmodule:: qmlt.tf.helpers

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

Collection of helpers to set up an experiment with the tensorflow circuit learner.

Summary
-------

.. autosummary::
    make_param

Code details
--------------
"""

from random import choice
from string import ascii_letters, digits
import tensorflow as tf


def make_param(name=None, stdev=None, mean=0., interval=None, constant=None, shape=(),
               regularize=False, monitor=False, seed=None):
    r"""Return a tensorflow variable.

    Args:
        name (str): name of the variable
        stdev (float): If not None, initialise from normal distribution.
        mean (float): If stdev is not None, use this mean for normal distribution. Defaults to 0.
        interval (list of length 2): If stdev is None and interval is not None, initialise from random value sampled
          uniformly from this interval.
        constant (float): If stdev and interval are both None and constant is not None, use this as an initial value.
          If constant is also None, use 0 as an initial value (not recommended!).
        shape (tuple or list): Shape of variable tensor. Useful for layered architectures.
        regularize (boolean): If true, mark this parameter for regularization.
        monitor (boolean): Whether to add this variable to tensorboard summary.
        seed (int): Use this seed to generate random numbers.
    """

    if name is None:
        name = ''.join(choice(ascii_letters + digits) for _ in range(5))

    if stdev is not None:
        if regularize:
            with tf.variable_scope("regularized"):
                var = tf.get_variable(name,
                                      initializer=tf.random_normal(shape=shape, stddev=stdev, mean=mean, seed=seed),
                                      dtype=tf.float32)
        else:
            var = tf.get_variable(name,
                                  initializer=tf.random_normal(shape=shape, stddev=stdev, mean=mean, seed=seed),
                                  dtype=tf.float32)
    elif interval is not None:
        if regularize:
            with tf.variable_scope("regularized"):
                var = tf.get_variable(name,
                                      initializer=tf.random_uniform(shape=(), minval=interval[0], maxval=interval[1],
                                                                    seed=seed),
                                      dtype=tf.float32)
        else:
            var = tf.get_variable(name,
                                  initializer=tf.random_uniform(shape=(), minval=interval[0], maxval=interval[1],
                                                                seed=seed),
                                  dtype=tf.float32)

    elif constant is not None:
        if regularize:
            with tf.variable_scope("regularized"):
                var = tf.get_variable(name,
                                      shape=shape,
                                      initializer=tf.constant_initializer(value=constant),
                                      dtype=tf.float32)
        else:
            var = tf.get_variable(name,
                                  shape=shape,
                                  initializer=tf.constant_initializer(value=constant),
                                  dtype=tf.float32)
    else:
        if regularize:
            with tf.variable_scope("regularized"):
                var = tf.get_variable(name,
                                      shape=shape,
                                      initializer=tf.constant_initializer(value=0.),
                                      dtype=tf.float32)

        else:
            var = tf.get_variable(name,
                                  shape=shape,
                                  initializer=tf.constant_initializer(value=0.),
                                  dtype=tf.float32)

    if monitor:
        if shape == ():
            tf.summary.scalar(name=name, tensor=var)
        else:
            var_flat = tf.reshape(var, [-1])
            var_list = tf.unstack(var_flat, axis=0)
            for idx, v in enumerate(var_list):
                tf.summary.scalar(name=name+'_'+str(idx), tensor=v)

    return var
