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
Unit tests for qmlt losses and regularization
========================================================

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

"""

import unittest
import tensorflow as tf
import numpy as np


from qmlt.numerical.losses import (square_loss as _square_loss,
                                   expectation as _expectation,
                                   trace_distance as _trace_distance,
                                   cross_entropy_with_softmax as _cross_ent)

from qmlt.numerical.regularizers import l2 as _l2_regul
from qmlt.numerical.regularizers import l1 as _l1_regul

from qmlt.tf.losses import (expectation as _tf_expectation,
                            trace_distance as _tf_trace_distance)


class BaseCostHelpersTest(unittest.TestCase):
    """
    Baseclass for loss tests.
    Creates some random inputs.
    """

    def setUp(self, seed, shape):
        if not seed is None:
            np.random.seed(seed)
        self.input1_arr = np.random.random(shape)
        self.input2_arr = np.random.random(shape)
        self.input1_batch_arr = np.array([np.random.random(shape) for i in range(3)])
        self.input2_batch_arr = np.array([np.random.random(shape) for i in range(3)])
        self.input1_batch_list = [np.random.random(shape) for i in range(3)]
        self.input2_batch_list = [np.random.random(shape) for i in range(3)]

    def assertIsScalar(self, z, msg=None):
        """ Check if z is a scalar. """
        if np.isscalar(z):
            return
        standardMsg = '{} is not a scalar.'.format(z)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertIsTfScalar(self, z, msg=None):
        """ Check if z is a scalar. """
        if z.get_shape() == ():
            return
        standardMsg = '{} is not a scalar Tensor.'.format(z)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)


class TestRegularizer(BaseCostHelpersTest):
    """
    Tests for regularizers that take a 1-d list, or array of scalars.
    """

    def setUp(self):
        super().setUp(seed=1, shape=())

    def test_l2regul_returns_scalar(self):
        self.assertIsScalar(_l2_regul(self.input1_batch_list))
        self.assertIsScalar(_l2_regul(self.input1_batch_arr))

    def test_l2regul_return_value(self):
        inpt = [1., 2., -3.]
        self.assertAlmostEqual(_l2_regul(inpt), 7.)

    def test_l1regul_returns_scalar(self):
        self.assertIsScalar(_l1_regul(self.input1_batch_list))
        self.assertIsScalar(_l1_regul(self.input1_batch_arr))

    def test_l1regul_return_value(self):
        inpt = [1., 2., -3.]
        self.assertAlmostEqual(_l1_regul(inpt), 6.)

    def test_l2regul_throws_exception(self):
        twodarr = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _l2_regul(twodarr)

    def test_l1regul_throws_exception(self):
        twodarr = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _l1_regul(twodarr)


class TestLosses(BaseCostHelpersTest):
    """
    Tests for functions that compute the similarity of 1-d outputs and targets in a batch.
    """

    def setUp(self):
        super().setUp(seed=1, shape=(2,))

    def test_meansquare_returns_scalar(self):
        self.assertIsScalar(_square_loss(self.input1_batch_list, self.input2_batch_list))
        self.assertIsScalar(_square_loss(self.input1_batch_arr, self.input2_batch_arr))

    def test_meansquare_throws_exception(self):
        threedarr = np.array([[[0., 1.], [1., 2.]], [[0., 1.], [1., 2.]]])
        twodarr = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _square_loss(threedarr, twodarr)
        with self.assertRaises(ValueError):
            _square_loss(threedarr, threedarr)

    def test_crossent_returns_scalar(self):
        self.assertIsScalar(_cross_ent(self.input1_batch_arr, self.input2_batch_arr))


class TestQuantumStateMeasures(BaseCostHelpersTest):
    """
    Tests for functions that compute distance or expectations with respect to 2-d density matrices.
    """

    def setUp(self):
        super().setUp(seed=1, shape=(2, 2))

    def test_expectation_returns_scalar(self):
        self.assertIsScalar(_expectation(self.input1_arr, self.input2_arr))

    def test_tf_expectation_returns_scalar(self):
        input1_arr = tf.convert_to_tensor(self.input1_arr)
        input2_arr = tf.convert_to_tensor(self.input2_arr)
        self.assertIsTfScalar(_tf_expectation(input1_arr, input2_arr))

    def test_tracedist_returns_scalar(self):
        self.assertIsScalar(_trace_distance(self.input1_arr, self.input2_arr))

    def test_tf_tracedist_returns_scalar(self):
        input1_arr = tf.convert_to_tensor(self.input1_arr)
        input2_arr = tf.convert_to_tensor(self.input2_arr)
        self.assertIsTfScalar(_tf_trace_distance(input1_arr, input2_arr))

    def test_tracedist_for_same_inputs_is_zero(self):
        self.assertAlmostEqual(_trace_distance(self.input1_arr, self.input1_arr), 0.)

    def test_tf_tracedist_for_same_inputs_is_zero(self):
        input1_arr = tf.convert_to_tensor(self.input1_arr)
        with tf.Session() as sess:
            res = sess.run(_tf_trace_distance(input1_arr, input1_arr))
        self.assertAlmostEqual(res, 0.)

    def test_tracedist_is_nonnegative(self):
        self.assertGreaterEqual(_trace_distance(self.input1_arr, self.input2_arr), 0.)

    def test_tf_tracedist_is_nonnegative(self):
        input1_arr = tf.convert_to_tensor(self.input1_arr)
        input2_arr = tf.convert_to_tensor(self.input2_arr)
        with tf.Session() as sess:
            res = sess.run(_tf_trace_distance(input1_arr, input2_arr))
        self.assertGreaterEqual(res, 0.)

    def test_tf_tracedist_throws_exception(self):
        onedtens = tf.convert_to_tensor([0., 1.])
        twodtens = tf.convert_to_tensor([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _tf_trace_distance(onedtens, twodtens)

    def test_expectation_returns_eigenvalue(self):
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        diag_input1 = mat
        diag_input2 = np.diag([4., 3., 2., 1.])
        self.assertAlmostEqual(_expectation(diag_input1, diag_input2), 4.)

    def test_tf_expectation_returns_eigenvalue(self):
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        diag_input1 = tf.convert_to_tensor(np.array(mat))
        diag_input2 = tf.convert_to_tensor(np.array(np.diag([4., 3., 2., 1.])))
        with tf.Session() as sess:
            res = sess.run(_tf_expectation(diag_input1, diag_input2))
        self.assertAlmostEqual(res, 4.)

    def test_tf_expectation_throws_exception(self):
        onedtens = tf.convert_to_tensor([0., 1.])
        twodtens = tf.convert_to_tensor([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _tf_expectation(onedtens, twodtens)
        with self.assertRaises(ValueError):
            _tf_expectation(onedtens, onedtens)

    def test_expectation_throws_exception(self):
        onedarr = np.array([0., 1.])
        twodarr = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _expectation(onedarr, twodarr)
        with self.assertRaises(ValueError):
            _expectation(onedarr, onedarr)

    def test_tracedist_throws_exception(self):
        onedarr = np.array([0., 1.])
        twodarr = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            _trace_distance(onedarr, twodarr)
        with self.assertRaises(ValueError):
            _trace_distance(onedarr, onedarr)


if __name__ == "__main__":

    print('Testing losses of the QMLT app.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in [TestRegularizer, TestLosses, TestQuantumStateMeasures]:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

