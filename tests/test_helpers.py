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
Unit tests for qmlt helpers
============================

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

"""

import unittest
import tensorflow as tf
import numpy as np
from qmlt.helpers import sample_from_distribution
from qmlt.numerical.helpers import make_param as make_param_num
from qmlt.tf.helpers import make_param as make_param_tf


class BaseHelpersTest(unittest.TestCase):
    """
    Baseclass for helper tests.
    """

    def setUp(self):
        self.seed = 1

    def assertAlmostEqual(self, one, two, msg=None, delta=1e-6):
        """
        Assert that ``one`` almost equal to ``two``. Default threshold is 1e-6.
        """
        if one == two:
            return
        if np.abs(one - two) <= delta:
            return
        standardMsg = 'The values differ more than threshold delta = {}'.format(delta)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAlmostEqualArray(self, one_arr, two_arr, msg=None, delta=1e-6):
        """
        Assert that the elements in 1-d array ``one`` are almost equal to those in 1-d array ``two``.
        Default threshold is 1e-6.
        """
        if np.allclose(one_arr, two_arr, atol=delta):
            return
        standardMsg = 'The values differ more than threshold delta = {}'.format(delta)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertDictHasKey(self, dict, key, msg=None):
        """
        Assert that dict has the key.
        """
        if key in dict:
            return
        standardMsg = 'The key {} is not in the dictionary'.format(key)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertEquals(self, one, two, msg=None):
        """
        Assert that one == two. Compares lists, tuples and strings.
        """
        if one == two:
            return
        standardMsg = 'The strings {} and {} are not equal'.format(one, two)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)


class TestMakeParamsNum(BaseHelpersTest):
    """
    Tests that the numeric make_params helper makes valid parameters.
    """

    def setUp(self):
        super().setUp()
        self.inp = 0.5
        self.interval = [-1, 1]

    def test_num_has_all_fields(self):
        p = make_param_num(constant=self.inp)
        self.assertDictHasKey(p, 'name')
        self.assertDictHasKey(p, 'val')
        self.assertDictHasKey(p, 'regul')
        self.assertDictHasKey(p, 'monitor')

    def test_num_make_constant_param(self):
        p = make_param_num(constant=self.inp)
        self.assertAlmostEqual(p['val'], self.inp)

    def test_num_make_normal_param(self):
        p = make_param_num(stdev=self.inp, seed=self.seed)
        self.assertAlmostEqual(p['val'], 0.812172681831, delta=0.0001)

    def test_num_make_interval_param(self):
        p = make_param_num(interval=self.interval, seed=self.seed)
        self.assertAlmostEqual(p['val'], -0.16595599059, delta=0.0001)

    def test_num_make_blank_param(self):
        p = make_param_num(seed=self.seed)
        self.assertAlmostEqual(p['val'], 0.0417022004702, delta=0.0001)


class TestMakeParamsTf(BaseHelpersTest):
    """
    Tests that the tensorflow make_params helper makes valid parameters.
    """

    def setUp(self):
        super().setUp()
        self.inp = 0.5
        self.interval = [-1, 1]
        self.shape = (1, 2)
        self.sess = tf.Session()

    def test_tf_make_constant_param(self):
        p = make_param_tf(constant=self.inp)
        self.sess.run(tf.global_variables_initializer())
        p_res = self.sess.run(p)
        self.assertAlmostEqual(p_res, self.inp)

    def test_tf_make_normal_param(self):
        p = make_param_tf(stdev=self.inp, seed=self.seed)
        self.sess.run(tf.global_variables_initializer())
        p_res = self.sess.run(p)
        self.assertAlmostEqual(p_res, -0.4056591, delta=0.0001)

    def test_tf_make_interval_param(self):
        p = make_param_tf(interval=self.interval, seed=self.seed)
        self.sess.run(tf.global_variables_initializer())
        p_res = self.sess.run(p)
        self.assertAlmostEqual(p_res, -0.5219252, delta=0.0001)

    def test_tf_regul_param_name(self):
        p = make_param_tf(name='param', regularize=True)
        self.assertEquals(p.name, 'regularized/param:0')

    def test_tf_regul_constant_param_name(self):
        p = make_param_tf(name='param1', constant=self.inp, regularize=True)
        self.assertEquals(p.name, 'regularized/param1:0')

    def test_tf_regul_normal_param_name(self):
        p = make_param_tf(name='param2', stdev=self.inp, regularize=True)
        self.assertEquals(p.name, 'regularized/param2:0')

    def test_tf_regul_interval_param_name(self):
        p = make_param_tf(name='param3', interval=self.interval, regularize=True)
        self.assertEquals(p.name, 'regularized/param3:0')

    def test_tf_constant_monitored_param(self):
        p = make_param_tf(constant=self.inp, monitor=True)
        self.sess.run(tf.global_variables_initializer())
        p_res = self.sess.run(p)
        self.assertAlmostEqual(p_res, self.inp)

    def test_tf_check_shape_deep_monitored_params(self):
        p = make_param_tf(constant=self.inp, shape=self.shape, monitor=True)
        self.sess.run(tf.global_variables_initializer())
        self.assertEqual(p.shape, self.shape)

    def test_tf_check_shape_deep_params(self):
        p = make_param_tf(shape=self.shape)
        self.sess.run(tf.global_variables_initializer())
        self.assertEqual(p.shape, self.shape)

class TestSampling(BaseHelpersTest):
    """
    Tests that the sampling helper returns a sample from a distribution.
    """

    def setUp(self):
        super().setUp()
        self.distribution = np.array([[0., 1.], [0., 0.]])

    def test_samplefromdistr_returns_sample(self):
        desired_result = np.array([0, 1])
        self.assertAlmostEqualArray(sample_from_distribution(self.distribution), desired_result)


if __name__ == "__main__":

    print('Testing helpers of the QMLT app.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in [TestMakeParamsNum, TestMakeParamsTf, TestSampling]:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

