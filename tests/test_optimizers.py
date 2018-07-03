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
Unit tests for qmlt
========================================================

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

Test that the optimizers of the QMLT app:
    - decrease loss
    - decrease regularized parameter
    - find an easy minimum 
    - don't update when starting in the minimum
    
"""

import unittest
import numpy as np
import tensorflow as tf
from qmlt.numerical import CircuitLearner as CircuitLearnerNUM
from qmlt.tf import CircuitLearner as CircuitLearnerTF

ALLOWED_OPTIMIZERS_TF = ["Adagrad", "Adam",  "RMSProp", "SGD"] #"Ftrl",
ALLOWED_OPTIMIZERS_NUM = ["SGD", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B", 
                          "TNC", "COBYLA", "SLSQP"]


class BaseOptimizerTest(unittest.TestCase):
    """
    Base class to test the optimizers of the QMLT CircuitLearners.
    
    """

    def setUp(self):
        """
        Set up attributes for test.
        """

        self.hyperp = {'regularization_strength': 1.,
                       'init_learning_rate': 0.1,
                       'decay': 0.,
                       'print_log': False,
                       'warm_start': False}
        self.init_param = 1.
        self.X = np.array([[1.]]).astype(np.float32)
        self.Y = np.array([0.]).astype(np.float32)

    def assertDecreases(self, before, after, msg=None):
        """
        Assert that before > after.
        """
        if before > after:
            return
        standardMsg = 'The value did not decrease.'
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertIncreases(self, before, after, msg=None):
        """
        Assert that before < after.
        """
        if before < after:
            return
        standardMsg = 'The value did not increase.'
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

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


class OptimizerTfOptimizationTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit():
            with tf.variable_scope("regularized"):
                param = tf.get_variable(shape=(), name='dummy', dtype=tf.float32,
                                        initializer=tf.constant_initializer(self.init_param))
            return 1. * param  # 1. necessary to turn result into tensor

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'optimization'

    def get_cost(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(steps=steps)
        evalu = learner.score_circuit()
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output):
            return (circuit_output - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """
        def myloss(circuit_output):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(before, after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """
        def myloss(circuit_output):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """
        def myloss(circuit_output):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)


class OptimizerTfUnsupervisedTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit():
            with tf.variable_scope("regularized"):
                param = tf.get_variable(shape=(), name='dummy', dtype=tf.float32,
                                        initializer=tf.constant_initializer(self.init_param))
            return 1. * param  # 1. necessary to turn result into tensor

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'unsupervised'


    def get_cost(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, steps=steps)
        evalu = learner.score_circuit(X=self.X)
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output, X):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output, X):
            return (circuit_output - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """

        def myloss(circuit_output, X):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(before, after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """

        def myloss(circuit_output, X):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """

        def myloss(circuit_output, X):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)


class OptimizerTfSupervisedTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit(X):
            with tf.variable_scope("regularized"):
                param = tf.get_variable(shape=(), name='dummy', dtype=tf.float32,
                                        initializer=tf.constant_initializer(self.init_param))
            return 1. * param * X[0, 0]  # 1. necessary to turn result into tensor

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'supervised'

    def get_cost(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, Y=self.Y, steps=steps)
        evalu = learner.score_circuit(X=self.X, Y=self.Y)
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerTF(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, Y=self.Y, steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output, targets):
            return (circuit_output + targets[0]) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output, targets):
            return (circuit_output + targets[0] - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """

        def myloss(circuit_output, targets):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(before, after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """

        def myloss(circuit_output, targets):
            return (circuit_output - targets[0]) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """

        def myloss(circuit_output, targets):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_TF:
            if opt != "Adagrad": # Adagrad saturates because input nonsparse
                with self.subTest(i=opt):
                    self.hyperp['optimizer'] = opt
                    after = self.get_circuit_params(steps=100)
                    self.assertAlmostEqual(0., after, delta=0.01)


class OptimizerNumericOptimizationTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit(params):
            return params[0]

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'optimization'
        self.hyperp['init_circuit_params'] = [{'val':self.init_param,
                                               'name':'dummy',
                                               'regul': True,
                                               'monitor':False}]
            
    def get_cost(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(steps=steps)
        evalu = learner.score_circuit()
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output):
            return (circuit_output - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """
        def myloss(circuit_output):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(before, after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """
        def myloss(circuit_output):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):       
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """
        def myloss(circuit_output):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)


class OptimizerNumericUnsupervisedTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit(params):
            return params[0]

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'unsupervised'
        self.hyperp['init_circuit_params'] = [{'val':self.init_param,
                                       'name':'dummy',
                                       'regul': True,
                                       'monitor':False}]

    def get_cost(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, steps=steps)
        evalu = learner.score_circuit(X=self.X)
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output, X):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output, X):
            return (circuit_output - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """

        def myloss(circuit_output, X):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(before, after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """

        def myloss(circuit_output, X):
            return circuit_output ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """

        def myloss(circuit_output, X):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)


class OptimizeNumericSupervisedTest(BaseOptimizerTest):

    def setUp(self):
        super().setUp()

        def mycircuit(X, params):
            return X[0, 0] * params[0]

        self.hyperp['circuit'] = mycircuit
        self.hyperp['task'] = 'supervised'
        self.hyperp['init_circuit_params'] = [{'val':self.init_param,
                                               'name':'dummy',
                                               'regul': True,
                                               'monitor': False}]

    def get_cost(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, Y=self.Y, steps=steps)
        evalu = learner.score_circuit(X=self.X, Y=self.Y)
        cost = evalu['loss']
        return cost

    def get_circuit_params(self, steps):
        learner = CircuitLearnerNUM(hyperparams=self.hyperp)
        learner.train_circuit(X=self.X, Y=self.Y, steps=steps)
        params = learner.get_circuit_parameters()
        param_value = params['regularized/dummy']
        return param_value

    def test_optimizer_step_decreases_loss(self):
        """
        Test if the loss decreases after three steps using a square loss centered at zero.

        """

        def myloss(circuit_output, targets):
            return (circuit_output + targets[0]) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                before = 1.
                after = self.get_cost(steps=3)
                self.assertDecreases(before, after)

    def test_no_update_in_minimum(self):
        """
        Test if the parameter does not change if we start in the minimum of a
        quadratic loss centered at 1.

        """

        def myloss(circuit_output, targets):
            return (circuit_output + targets[0] - 1.) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertAlmostEqual(1., after)

    def test_regularizer_decreases_param(self):
        """
        Test that with zero loss, a l2 regularizer has the effect of
        decreasing the parameter after three steps.
        (Note: some optimization methods cannot deal with l1 loss.)

        """

        def myloss(circuit_output, targets):
            return 0.

        def myregularizer(regularized_params):
            return (regularized_params[0]) ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=3)
                self.assertDecreases(1., after)

    def test_find_squareloss_minimum_at_zero(self):
        """
        Test if optimizer finds the minimum of a simple square loss centered
        around 0 after 100 steps.

        """

        def myloss(circuit_output, targets):
            return (circuit_output - targets[0]) ** 2

        def myregularizer(regularized_params):
            return 0.

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)

    def test_find_l2regulariser_minimum_at_zero(self):
        """
        Test if the regularizer dampens the parameter to zero after 100 steps.

        """

        def myloss(circuit_output, targets):
            return 0.

        def myregularizer(regularized_params):
            return regularized_params[0] ** 2

        self.hyperp['loss'] = myloss
        self.hyperp['regularizer'] = myregularizer

        for opt in ALLOWED_OPTIMIZERS_NUM:
            with self.subTest(i=opt):
                self.hyperp['optimizer'] = opt
                after = self.get_circuit_params(steps=100)
                self.assertAlmostEqual(0., after, delta=0.01)


if __name__ == "__main__":

    print('Testing optimizers of the QMLT app.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in [OptimizerTfOptimizationTest, OptimizerTfUnsupervisedTest, OptimizerTfSupervisedTest,
              OptimizerNumericOptimizationTest, OptimizerNumericUnsupervisedTest, OptimizeNumericSupervisedTest]:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)