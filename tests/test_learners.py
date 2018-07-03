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
Unit tests for qmlt learners
============================

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

"""
import os
import unittest
from io import StringIO
import sys
from copy import deepcopy
import shutil
import tensorflow as tf
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from qmlt.numerical import CircuitLearner as Learner_num
from qmlt.tf import CircuitLearner as Learner_tf
from qmlt.numerical.helpers import make_param as make_param_num
from qmlt.tf.helpers import make_param as make_param_tf
import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
from qmlt.tf.learner import (_check_Y as tf_check_Y,
                             _check as tf_check,
                             _check_shuffle as tf_check_shuffle,
                             _check_X as tf_check_X,
                             _check_batch_size as tf_check_batch_size,
                             _check_steps as tf_check_steps)
from qmlt.numerical.learner import (_check_Y as num_check_Y,
                                    _check as num_check,
                                    _check_X as num_check_X,
                                    _check_batch_size as num_check_batch_size,
                                    _check_steps as num_check_steps,
                                    _check_logs as num_check_logs)

class BaseLearnerTest(unittest.TestCase):
    """
    Baseclass for learner tests.
    """

    def setUp(self):
        return

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

    def assertStringNotEmpty(self, mystring, msg=None):
        """
        Assert that str is a nonempty string.
        """
        if not mystring:
            standardMsg = 'String is empty'
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)
        else:
            return

    def assertPathExists(self, path, msg=None):
        if os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/"+path):
            return
        standardMsg = 'The path {} does not exist'.format(path)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)


class TestNumericalLearnerOptimization(BaseLearnerTest):
    """
    Test the numeric learner for optimization tasks
    """

    def setUp(self):
        super().setUp()

        my_init_params = [make_param_num(constant=0.1, name='alpha')]

        def circuit(params):
            eng, q = sf.Engine(1)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('gaussian')
            circuit_output = state.fock_prob([1])
            return circuit_output

        def myloss(circuit_output):
            return -circuit_output

        self.hyperparams = {'circuit': circuit,
                            'init_circuit_params': my_init_params,
                            'task': 'optimization',
                            'loss': myloss,
                            'print_log': False
                            }

        self.dummy_input = np.array([[0., 1.]])

    def test_basic_optimization_num(self):
        """
        Test to optimize a StrawberryFields Dgate with minimal parameters.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.1213333101, delta=0.0001)

    def test_optimization_custom_modeldir(self):
        """
        Test to optimize a StrawberryFields Dgate with custom model directory.
        """
        _ = Learner_num(hyperparams=self.hyperparams, model_dir="newpath/modeldir")
        self.assertPathExists('../newpath/')

    def test_optimization_with_regul_num(self):
        """
        Test to optimize a StrawberryFields Dgate with regularization.
        """
        def myregularizer(regularized_params):
            return sum(regularized_params)

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['regularizer'] = myregularizer
        new_hyperparams['regularization_strength'] = 1.
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.1213333101, delta=0.0001)

    def test_optimization_with_learningrate_num(self):
        """
        Test to optimize a StrawberryFields Dgate with a custom learning rate.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['init_learning_rate'] = 2.
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 1.449175616872, delta=0.0001)

    def test_optimization_with_decay_num(self):
        """
        Test to optimize a StrawberryFields Dgate with learning rate decay.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['decay'] = 0.5
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.108182953535, delta=0.0001)

    def test_optimization_with_warmstart_num(self):
        """
        Test to optimize a StrawberryFields Dgate with warm start.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['warm_start'] = True
        learner2 = Learner_num(hyperparams=new_hyperparams)
        learner2.train_circuit(steps=1)
        final_par = int(np.loadtxt("logsNUM/global_step.txt"))
        self.assertAlmostEqual(final_par, 11)

    def test_optimization_with_warmstart_raises_error_num(self):
        """
        Test if optimizing a StrawberryFields Dgate with warm start but no parameter file raises error.
        """
        with self.assertRaises(IOError):
            shutil.rmtree('logsNUM', ignore_errors=True)
            new_hyperparams = deepcopy(self.hyperparams)
            new_hyperparams['warm_start'] = True
            learner = Learner_num(hyperparams=new_hyperparams)
            learner.train_circuit(steps=1)
        with self.assertRaises(IOError):
            new_hyperparams['warm_start'] = False
            learner = Learner_num(hyperparams=new_hyperparams)
            learner.train_circuit(steps=1)
            shutil.rmtree('logsNUM/global_step.txt', ignore_errors=False)
            learner.train_circuit(steps=1)
        with self.assertRaises(IOError):
            new_hyperparams['warm_start'] = True
            new_hyperparams['init_circuit_params'] = [make_param_num(constant=0.1), make_param_num(constant=0.1)]
            learner = Learner_num(hyperparams=new_hyperparams)
            learner.train_circuit(steps=1)

    def test_optimization_with_adaptivelr_num(self):
        """
        Test to optimize a StrawberryFields Dgate with adaptive learning
        rate.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['adaptive_learning_rate_threshold'] = 0.5
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.346251338090, delta=0.0001)

    def test_optimization_with_logging_num(self):
        """
        Test to optimize a StrawberryFields Dgate while saving a log.
        """

        def circuit2(params):
            eng, q = sf.Engine(1)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('gaussian')
            circuit_output = state.fock_prob([1])
            log = {'log_value':1.}
            return circuit_output, log

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['log_every'] = 2
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = np.genfromtxt("logsNUM/log.csv", delimiter=',', skip_header=1)
        self.assertAlmostEqual(final_par[0, 5], 1)
        learner.run_circuit()
        learner.score_circuit()

    def test_optimization_with_printlog_SGD_regul_num(self):
        """
        Test to optimize a StrawberryFields Dgate with printing the log.
        """
        def circuit2(params):
            eng, q = sf.Engine(1)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('gaussian')
            circuit_output = state.fock_prob([1])
            log = {'log_value':1.}
            return circuit_output, log

        def myregularizer(regularized_params):
            return 0

        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['print_log'] = True
        new_hyperparams['regularizer'] = myregularizer
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=1)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(capturedOutput.getvalue())

    def test_optimization_with_printlog_CG_num(self):
        """
        Test to optimize a StrawberryFields Dgate with printing the log.
        """
        def circuit2(params):
            eng, q = sf.Engine(1)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('gaussian')
            circuit_output = state.fock_prob([1])
            log = {'log_value':1.}
            return circuit_output, log

        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['optimizer'] = 'CG'
        new_hyperparams['print_log'] = True
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=1)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(capturedOutput.getvalue())

    def test_optimization_SGD_with_monitoring_num(self):
        """
        Test to optimize a StrawberryFields Dgate while saving a log of the variable using the SGD optimizer.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['init_circuit_params'] = [make_param_num(constant=0.1,
                                                                 name='alpha',
                                                                 monitor=True)]

        new_hyperparams['log_every'] = 2
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = np.genfromtxt("logsNUM/log.csv", delimiter=',', skip_header=1)
        self.assertAlmostEqual(final_par[0, 5], 0.1)

    def test_optimization_CG_with_monitoring_num(self):
        """
        Test to optimize a StrawberryFields Dgate while saving a log of the variable using the CG optimizer.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['init_circuit_params'] = [make_param_num(constant=0.1,
                                                                 name='alpha',
                                                                 monitor=True)]

        new_hyperparams['log_every'] = 2
        new_hyperparams['optimizer'] = 'CG'
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = np.genfromtxt("logsNUM/log.csv", delimiter=',', skip_header=1)
        self.assertAlmostEqual(final_par[0, 5], 0.1)

    def test_optimization_score_num(self):
        """
        Test to optimize a StrawberryFields Dgate and score the circuit.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        test_score = learner.score_circuit()
        res = test_score['loss']
        self.assertAlmostEqual(res, -0.01450662909, delta=0.0001)
        with self.assertRaises(ValueError):
            learner.score_circuit(X=self.dummy_input)

    def test_optimization_run_num(self):
        """
        Test to optimize a StrawberryFields Dgate and run the circuit.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        test_score = learner.run_circuit()
        res = test_score['outputs']
        self.assertAlmostEqual(res, 0.014506629095, delta=0.0001)
        with self.assertRaises(ValueError):
            learner.run_circuit(X=self.dummy_input)

    def test_optimization_with_printing_params_num(self):
        """
        Test the get_circuit_parameters function printing.
        """

        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(steps=1)
        learner.get_circuit_parameters(only_print=True)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(capturedOutput.getvalue())


class TestNumericalLearnerUnsupervised(BaseLearnerTest):
    """
    Test the numeric learner for unsupervised learning tasks
    """

    def setUp(self):
        super().setUp()

        my_init_params = [make_param_num(name='alpha', constant=1.)]

        def circuit(params):
            eng, q = sf.Engine(2)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('fock', cutoff_dim=5)
            circuit_output = state.all_fock_probs()
            return circuit_output

        def myloss(circuit_output, X):
            probs = [circuit_output[x[0], x[1]] for x in X]
            prob_total = sum(np.reshape(probs, -1))
            return -prob_total

        self.X_train = np.array([[0, 0]])

        self.hyperparams = {'circuit': circuit,
                            'init_circuit_params': my_init_params,
                            'task': 'unsupervised',
                            'optimizer': 'CG',
                            'loss': myloss,
                            'print_log': False
                            }

    def test_basic_unsupervised_num(self):
        """
        Basic test to train a variational circuit in unsupervised fashion.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=2)
        final_par = learner.get_circuit_parameters()
        final_par = final_par['alpha']
        self.assertAlmostEqual(final_par, 0., delta=0.0001)

    def test_basic_unsupervised_SGD_num(self):
        """
        Basic test to train a variational circuit in unsupervised fashion with SGD optimizer.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['optimizer'] = 'SGD'
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(X=self.X_train, steps=2)
        final_par = learner.get_circuit_parameters()
        final_par = final_par['alpha']
        self.assertAlmostEqual(final_par, 0.98523, delta=0.01)

    def test_unsupervised_score_num(self):
        """
        Test to to train a variational circuit in unsupervised fashion
        and score the circuit.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=2)
        test_score = learner.score_circuit(X=self.X_train)
        res = test_score['loss']
        self.assertAlmostEqual(res, -1., delta=0.0001)
        with self.assertRaises(ValueError):
            learner.score_circuit()

    def test_unsupervised_run_num(self):
        """
        Test to to train a variational circuit in unsupervised fashion
        and run the circuit.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=30)
        outp = learner.run_circuit()['outputs']
        self.assertAlmostEqualArray(outp[0, 0], 1.)
        with self.assertRaises(ValueError):
            learner.run_circuit(X=self.X_train)

    def test_unsupervised_with_logging_num(self):
        """
        Test to to train a variational circuit in unsupervised fashion
         while saving a log.
        """

        def circuit2(params):
            eng, q = sf.Engine(2)
            with eng:
                Dgate(params[0]) | q[0]
            state = eng.run('fock', cutoff_dim=5)
            circuit_output = state.all_fock_probs()
            log = {'log_value': 1.}
            return circuit_output, log

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['log_every'] = 2
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(X=self.X_train, steps=10)
        final_par = np.genfromtxt("logsNUM/log.csv", delimiter=',', skip_header=1)
        self.assertAlmostEqual(final_par[0, 5], 1)
        learner.run_circuit()
        learner.score_circuit(X=self.X_train)

    def test_unsupervised_batch_num(self):
        """
        Basic test to train a variational circuit in supervised fashion
        with batched SGD.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=2, batch_size=1, seed=1)
        final_par = learner.get_circuit_parameters()
        final_par = final_par['alpha']
        self.assertAlmostEqual(final_par, 0., delta=0.0001)


class TestNumericalLearnerSupervised(BaseLearnerTest):
    """
    Test the numeric learner for supervised learning tasks.
    """

    def setUp(self):
        super().setUp()

        my_init_params = [make_param_num(name='alpha', constant=2.)]

        def circuit(X, params):
            eng, q = sf.Engine(2)

            def single_input_circuit(x):
                eng.reset()
                with eng:
                    Dgate(x[0], 0.) | q[0]
                    Dgate(x[1], 0.) | q[1]
                    BSgate(phi=params[0]) | (q[0], q[1])
                    BSgate() | (q[0], q[1])
                state = eng.run('fock', cutoff_dim=10, eval=True)
                p0 = state.fock_prob([0, 2])
                p1 = state.fock_prob([2, 0])
                normalisation = p0 + p1 + 1e-10
                outp = p1 / normalisation
                return outp

            circuit_output = [single_input_circuit(x) for x in X]
            return circuit_output

        def myloss(circuit_output, targets):
            diff = circuit_output - targets
            return 0.5 * sum(np.dot(d, d) for d in diff)

        self.X_train = np.array([[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]])
        self.Y_train = np.array([1., 1., 0., 0.])
        self.X_test = np.array([[0.25, 0.5], [0.5, 0.25]])
        self.Y_test = np.array([1., 0.])
        self.X_pred = np.array([[0.4, 0.5], [0.5, 0.4]])

        def outputs_to_predictions(circuit_output):
            return round(circuit_output)

        self.otp = outputs_to_predictions

        self.hyperparams = {'circuit': circuit,
                            'init_circuit_params': my_init_params,
                            'task': 'supervised',
                            'loss': myloss,
                            'init_learning_rate': 0.5,
                            'print_log': False
                            }

    def test_basic_supervised_num(self):
        """
        Basic test to to train a variational circuit in supervised fashion
        and score the result.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=30)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.1709563877, delta=0.0001)

    def test_supervised_score_num(self):
        """
        Test to to train a variational circuit in supervised fashion
        and score the result.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=40)
        test_score = learner.score_circuit(X=self.X_test, Y=self.Y_test,
                                           outputs_to_predictions=self.otp)
        self.assertAlmostEqual(test_score['accuracy'], 1., delta=0.0001)
        self.assertAlmostEqual(test_score['loss'], 0., delta=0.1)
        with self.assertRaises(ValueError):
            learner.score_circuit(X=None, Y=None, outputs_to_predictions=self.otp)

    def test_supervised_run_num(self):
        """
        Test to to train a variational circuit in supervised fashion
        and predict the result.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=30)
        outcomes = learner.run_circuit(X=self.X_test, outputs_to_predictions=self.otp)
        self.assertAlmostEqualArray(outcomes['predictions'], [1., 0.])
        with self.assertRaises(ValueError):
            learner.run_circuit(X=None, outputs_to_predictions=self.otp)

    def test_supervised_with_logging_num(self):
        """
        Test to to train a variational circuit in supervised fashion
         while saving a log.
        """

        def circuit2(X, params):
            eng, q = sf.Engine(2)

            def single_input_circuit(x):
                eng.reset()
                with eng:
                    Dgate(x[0], 0.) | q[0]
                    Dgate(x[1], 0.) | q[1]
                    BSgate(phi=params[0]) | (q[0], q[1])
                    BSgate() | (q[0], q[1])
                state = eng.run('fock', cutoff_dim=10, eval=True)
                p0 = state.fock_prob([0, 2])
                p1 = state.fock_prob([2, 0])
                normalisation = p0 + p1 + 1e-10
                outp = p1 / normalisation
                return outp

            circuit_output = [single_input_circuit(x) for x in X]
            log = {'log_value': 1.}
            return circuit_output, log

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['log_every'] = 2
        learner = Learner_num(hyperparams=new_hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=10)
        final_par = np.genfromtxt("logsNUM/log.csv", delimiter=',', skip_header=1)
        self.assertAlmostEqual(final_par[0, 5], 1)
        learner.run_circuit(X=self.X_test)
        learner.score_circuit(X=self.X_test, Y=self.Y_test)

    def test_supervised_batch_num(self):
        """
        Basic test to train a variational circuit in supervised fashion
        with batched SGD.
        """
        learner = Learner_num(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=30, batch_size=2)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 1.74412942216, delta=0.2)


class TestTfLearnerOptimization(BaseLearnerTest):
    """
    Test the TensorFlow learner for optimization tasks
    """

    def setUp(self):
        super().setUp()

        def circuit():
            alpha = make_param_tf(name='alpha', constant=0.1)
            eng, q = sf.Engine(1)
            with eng:
                Dgate(alpha) | q[0]
            state = eng.run('tf', cutoff_dim=7, eval=False)
            prob = state.fock_prob([1])
            circuit_output = tf.identity(prob, name="prob")
            return circuit_output

        def myloss(circuit_output):
            return -circuit_output

        self.hyperparams = {'circuit': circuit,
                            'task': 'optimization',
                            'loss': myloss,
                            'print_log': False
                            }
        self.dummy_input = np.array([[0., 1.]])

    def test_basic_optimization_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with minimal parameters.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.1213333101, delta=0.0001)

    def test_optimization_with_regul_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with regularization.
        """
        def myregularizer(regularized_params):
            return sum(regularized_params)

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['regularizer'] = myregularizer
        new_hyperparams['regularization_strength'] = 1.
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.1213333101, delta=0.0001)

    def test_optimization_with_learningrate_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with a custom learning rate.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['init_learning_rate'] = 2.
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 1.449175616872, delta=0.0001)

    def test_optimization_with_decay_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with learning rate decay.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['decay'] = 0.5
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.117403194, delta=0.0001)

    def test_optimization_with_warmstart_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with warm start.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['warm_start'] = True
        learner2 = Learner_tf(hyperparams=new_hyperparams)
        learner2.train_circuit(steps=1)
        final_par = learner.get_circuit_parameters()['global_step']
        self.assertAlmostEqual(final_par, 11)

    def test_optimization_with_logging_not_throwing_errors_tf(self):
        """
        Test to optimize a StrawberryFields Dgate while saving a log.
        """
        def circuit2():
            alpha = make_param_tf(name='alpha', constant=0.1, monitor=True)
            eng, q = sf.Engine(1)
            with eng:
                Dgate(alpha) | q[0]
            state = eng.run('tf', cutoff_dim=7, eval=False)
            prob = state.fock_prob([1])
            circuit_output = tf.identity(prob, name="prob")
            return circuit_output

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['print_log'] = True
        new_hyperparams['log_every'] = 2
        new_hyperparams['circuit'] = circuit2
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=4, tensors_to_log={'Prob': 'prob'})

    def test_optimization_with_monitoring_tf(self):
        """
        Test to optimize a StrawberryFields Dgate while saving a log of the variable.
        """

        def circuit2():
            alpha = make_param_tf(name='alpha', constant=0.1, monitor=True)
            eng, q = sf.Engine(1)
            with eng:
                Dgate(alpha) | q[0]
            state = eng.run('tf', cutoff_dim=7, eval=False)
            return state.fock_prob([1])

        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['circuit'] = circuit2
        new_hyperparams['log_every'] = 2
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=10)

    def test_optimization_score_tf(self):
        """
        Test to optimize a StrawberryFields Dgate and score the circuit.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        test_score = learner.score_circuit()
        res = test_score['loss']
        self.assertAlmostEqual(res, -0.01450662909, delta=0.0001)
        with self.assertRaises(ValueError):
            learner.score_circuit(X=self.dummy_input)

    def test_optimization_run_tf(self):
        """
        Test to optimize a StrawberryFields Dgate and run the circuit.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(steps=10)
        test_score = learner.run_circuit()
        res = test_score['outputs']
        self.assertAlmostEqual(res, 0.014506629095, delta=0.0001)
        with self.assertRaises(ValueError):
            learner.run_circuit(X=self.dummy_input)

    def test_optimization_with_printlog_tf(self):
        """
        Test to optimize a StrawberryFields Dgate with printing the log.
        """

        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['print_log'] = True
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(steps=1)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(capturedOutput.getvalue())

    def test_optimization_with_printing_params_tf(self):
        """
        Test the get_circuit_parameters function prints.
        """

        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(steps=1)
        learner.get_circuit_parameters(only_print=True)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(capturedOutput.getvalue())


class TestTfLearnerUnsupervised(BaseLearnerTest):
    """
    Test the TensorFLow learner for unsupervised learning tasks
    """

    def setUp(self):
        super().setUp()

        def circuit():
            alpha = make_param_tf(name='alpha', constant=1.)
            eng, q = sf.Engine(2)
            with eng:
                Dgate(alpha) | q[0]
            state = eng.run('tf', cutoff_dim=5, eval=False)
            circuit_output = state.all_fock_probs()
            return circuit_output

        def myloss(circuit_output, X):
            probs = tf.gather_nd(params=circuit_output, indices=X)
            prob_total = tf.reduce_sum(probs, axis=0)
            return -prob_total

        self.X_train = np.array([[0, 0]])

        self.hyperparams = {'circuit': circuit,
                            'task': 'unsupervised',
                            'optimizer': 'SGD',
                            'loss': myloss,
                            'print_log': False
                            }

    def test_ftrl_prints_warning_tf(self):
        """
        Basic test to check that using the FTRL optimizer prints a warning.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=30)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 0.75931054, delta=0.01)

    def test_unsupervised_score_tf(self):
        """
        Test to to train a variational circuit in unsupervised fashion
        and score the circuit.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=30)
        test_score = learner.score_circuit(X=self.X_train)
        res = test_score['loss']
        self.assertAlmostEqual(res, -0.561831, delta=0.01)
        with self.assertRaises(ValueError):
            learner.score_circuit(X=None)

    def test_unsupervised_run_tf(self):
        """
        Test to to train a variational circuit in unsupervised fashion
        and run the circuit.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=30)
        outp = learner.run_circuit()['outputs']
        self.assertAlmostEqual(outp[0, 0], 0.56183195, delta=0.01)
        with self.assertRaises(ValueError):
            learner.run_circuit(X=self.X_train)

    def test_unsupervised_with_logging_tf(self):
        """
        Test to to train a variational circuit in unsupervised fashion
         while saving a log.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['log_every'] = 2
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(X=self.X_train, steps=30)

    def test_unsupervised_batch_tf(self):
        """
        Basic test to train a variational circuit in supervised fashion
        with batched SGD.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, steps=30, batch_size=1, shuffle_data=False)
        final_par = learner.get_circuit_parameters()
        final_par = final_par['alpha']
        self.assertAlmostEqual(final_par, 0.75931054, delta=0.01)


class TestTfLearnerSupervised(BaseLearnerTest):
    """
    Test the TensorFLow learner for supervised learning tasks
    """

    def setUp(self):
        super().setUp()

        def circuit(X):
            alpha = make_param_tf('alpha', constant=2.)
            eng, q = sf.Engine(2)
            with eng:
                Dgate(X[:, 0], 0.) | q[0]
                Dgate(X[:, 1], 0.) | q[1]
                BSgate(phi=alpha) | (q[0], q[1])
                BSgate() | (q[0], q[1])
            num_inputs = X.get_shape().as_list()[0]
            state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)
            p0 = state.fock_prob([0, 2])
            p1 = state.fock_prob([2, 0])
            normalisation = p0 + p1 + 1e-10
            circuit_output = p1 / normalisation
            return circuit_output

        def myloss(circuit_output, targets):
            return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)

        self.X_train = np.array([[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]])
        self.Y_train = np.array([1., 1., 0., 0.])
        self.X_test = np.array([[0.25, 0.5], [0.5, 0.25]])
        self.Y_test = np.array([1., 0.])
        self.X_pred = np.array([[0.4, 0.5], [0.5, 0.4]])

        def outputs_to_predictions(circuit_output):
            return tf.round(circuit_output)

        self.otp = outputs_to_predictions

        self.hyperparams = {'circuit': circuit,
                            'task': 'supervised',
                            'loss': myloss,
                            'init_learning_rate': 0.5,
                            'print_log': False
                            }

    def test_basic_supervised_tf(self):
        """
        Basic test to to train a variational circuit in supervised fashion
        and score the result.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=30)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 1.7205561, delta=0.01)

    def test_supervised_score_tf(self):
        """
        Test to to train a variational circuit in supervised fashion
        and score the result.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=60)
        test_score = learner.score_circuit(X=self.X_test, Y=self.Y_test,
                                           outputs_to_predictions=self.otp)
        self.assertAlmostEqual(test_score['accuracy'], 1., delta=0.0001)
        self.assertAlmostEqual(test_score['loss'], 0., delta=0.1)
        with self.assertRaises(ValueError):
            learner.score_circuit(X=None, Y=None, outputs_to_predictions=self.otp)

    def test_supervised_run_tf(self):
        """
        Test to to train a variational circuit in supervised fashion
        and predict the result.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=60)
        outcomes = learner.run_circuit(X=self.X_test, outputs_to_predictions=self.otp)
        self.assertAlmostEqualArray(outcomes['predictions'], [1.,0.])
        with self.assertRaises(ValueError):
            learner.run_circuit(X=None, outputs_to_predictions=self.otp)

    def test_supervised_with_logging_tf(self):
        """
        Test to to train a variational circuit in supervised fashion
         while saving a log.
        """
        new_hyperparams = deepcopy(self.hyperparams)
        new_hyperparams['log_every'] = 2
        learner = Learner_tf(hyperparams=new_hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=10)

    def test_supervised_batch_tf(self):
        """
        Basic test to train a variational circuit in supervised fashion
        with batched SGD.
        """
        learner = Learner_tf(hyperparams=self.hyperparams)
        learner.train_circuit(X=self.X_train, Y=self.Y_train, steps=30, batch_size=2)
        final_par = learner.get_circuit_parameters()['alpha']
        self.assertAlmostEqual(final_par, 1.74412942216, delta=0.2)


class TestInputChecks(BaseLearnerTest):
    """
    Test input checks.
    """

    def setUp(self):
        super().setUp()
        self.onedarr = np.array([1., 0., 2.])
        self.onedlis = [1., 0.]
        self.twodarr = np.array([[1., 0.], [0., 1.]])
        self.twodlis = [[1., 0.], [0., 1.]]
        self.threedarr = np.array([[[0., 1.], [1., 2.]], [[0., 1.], [1., 2.]]])
        self.integer = 1
        self.neg = -1
        self.flt = 0.1
        self.string = "bla"
        self.dict = {'a a': 1.}

        def regul(regularized_params):
            return 0
        self.regul = regul


    def test_check_hp_throws_exception(self):
        """
        Test the exceptions of the hyperparameter check for both the TensorFlow and numerical learner.
        """
        num_check({'print_log': False, 'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': 1})
        # No task key
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'init_circuit_params': [{}]})
        with self.assertRaises(ValueError):
            tf_check({'circuit': 1, 'loss': 1})
        # No loss key
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'task': 'optimization', 'init_circuit_params': [{}]})
        with self.assertRaises(ValueError):
            tf_check({'circuit': 1,  'task': 'optimization'})
        # No circuit key
        with self.assertRaises(ValueError):
            num_check({'loss': 1, 'task': 'optimization', 'init_circuit_params': [{}]})
        with self.assertRaises(ValueError):
            tf_check({'loss': 1, 'task': 'optimization'})
        # No init param key
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'task': 'optimization'})
        # Wrong task
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'task': self.string, 'init_circuit_params': 1.1})
        with self.assertRaises(ValueError):
            tf_check({'circuit': 1, 'loss': 1, 'task': self.string, 'optimizer': 'sthfunny'})
        # Wrong init params
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': 1.1})
        # Unrecognized key
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': [{}], 'funny_key': 1.})
        with self.assertRaises(ValueError):
            tf_check({'circuit': 1, 'loss': 1, 'task': 'optimization', 'funny_key': 1.})
        # Unrecognized optimizer
        with self.assertRaises(ValueError):
            num_check({'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': [{}], 'optimizer': 'sthfunny'})
        with self.assertRaises(ValueError):
            tf_check({'circuit': 1, 'loss': 1, 'task': 'optimization', 'optimizer': 'sthfunny'})

    def test_check_hp_sets_default_regstrength(self):
        """
        Test that when regularizer is given but no regularization strength, the strength is set to 0.1, else it is
        set to user provided value.
        """
        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': [{}],
              'regularizer': self.regul}
        num_check(hp)
        self.assertAlmostEqual(hp['regularization_strength'], 0.1)

        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization', 'regularizer': self.regul}
        tf_check(hp)
        self.assertAlmostEqual(hp['regularization_strength'], 0.1)

        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization', 'init_circuit_params': [{}],
              'regularizer': self.regul, 'regularization_strength': 0.5}
        num_check(hp)
        self.assertAlmostEqual(hp['regularization_strength'], 0.5)

        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization', 'regularizer': self.regul, 'regularization_strength': 0.5}
        tf_check(hp)
        self.assertAlmostEqual(hp['regularization_strength'], 0.5)

    def test_check_hp_renames_regularized_params(self):
        """
        Test check that the initial parameter name contains string <regularized> when regularized.
        """
        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization',
              'init_circuit_params': [{'name': 'dummy', 'regul': True}]}
        num_check(hp)
        self.assertEquals(hp['init_circuit_params'][0]['name'], 'regularized/dummy')

    def test_check_hp_prints_warning_for_ftrl_optimizer(self):
        """
        Test Ftrl warning gets printed in hyperparameter check of tf learner.
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        hp = {'circuit': 1, 'loss': 1, 'task': 'optimization', 'optimizer': 'Ftrl', 'print_log':False}
        tf_check(hp)
        sys.stdout = sys.__stdout__
        self.assertStringNotEmpty(captured_output.getvalue())

    def test_checkX_throws_exception(self):
        """
        Test the exceptions of check_X for both the TensorFlow and numerical learner.
        """
        num_check_X(self.twodlis)
        tf_check_X(self.twodlis)

        with self.assertRaises(ValueError):
            num_check_X(self.dict)
        with self.assertRaises(ValueError):
            tf_check_X(self.dict)
        with self.assertRaises(ValueError):
            num_check_X(self.onedarr)
        with self.assertRaises(ValueError):
            tf_check_X(self.onedarr)
        with self.assertRaises(ValueError):
            num_check_X(self.onedlis)
        with self.assertRaises(ValueError):
            tf_check_X(self.onedlis)

    def test_checkY_throws_exception(self):
        """
        Test the exceptions of check_Y for both the TensorFlow and numerical learner.
        """
        with self.assertRaises(ValueError):
            num_check_Y(self.onedarr, self.twodarr)
        with self.assertRaises(ValueError):
            tf_check_Y(self.onedarr, self.twodarr)
        with self.assertRaises(ValueError):
            num_check_Y(self.twodarr, None)
        with self.assertRaises(ValueError):
            tf_check_Y(self.twodarr, None)
        with self.assertRaises(ValueError):
            num_check_Y(self.dict, self.integer)
        with self.assertRaises(ValueError):
            tf_check_Y(self.dict, self.integer)

    def test_checkshuffle_throws_exception(self):
        """
        Test the exceptions of check_shuffle for the TensorFlow learner.
        """
        with self.assertRaises(ValueError):
            tf_check_shuffle(self.integer)

    def test_checkbatches_throws_exception(self):
        """
        Test the exceptions of check_batch_size for both the TensorFlow and numerical learner.
        """
        num_check_batch_size(None, self.twodarr)
        tf_check_batch_size(None, self.twodarr)
        with self.assertRaises(ValueError):
            num_check_batch_size(self.dict, self.twodarr)
        with self.assertRaises(ValueError):
            tf_check_batch_size(self.dict, self.twodarr)
        with self.assertRaises(ValueError):
            num_check_batch_size(5, self.twodarr)
        with self.assertRaises(ValueError):
            tf_check_batch_size(5, self.twodarr)
        with self.assertRaises(ValueError):
            num_check_batch_size(self.neg, self.twodarr)
        with self.assertRaises(ValueError):
            tf_check_batch_size(self.neg, self.twodarr)

    def test_checksteps_throws_exception(self):
        """
        Test the exceptions of check_steps for both the TensorFlow and numerical learner.
        """
        num_check_steps(None)
        tf_check_steps(None)
        with self.assertRaises(ValueError):
            num_check_steps(self.dict)
        with self.assertRaises(ValueError):
            tf_check_steps(self.dict)
        with self.assertRaises(ValueError):
            num_check_X(self.neg)
        with self.assertRaises(ValueError):
            tf_check_X(self.neg)

    def test_check_logs_throws_exception(self):
        """
        Test the exceptions of check_logs for the numerical learner.
        """
        with self.assertRaises(ValueError):
            num_check_logs(self.integer)
        with self.assertRaises(ValueError):
            num_check_logs(self.dict)


if __name__ == "__main__":
    print('Testing learners of the QMLT app.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in [TestInputChecks,
              TestNumericalLearnerOptimization,
              TestNumericalLearnerUnsupervised,
              TestNumericalLearnerSupervised,
              TestTfLearnerOptimization,
              TestTfLearnerUnsupervised,
              TestTfLearnerSupervised,
             ]:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

