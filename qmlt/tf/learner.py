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
Tensorflow Quantum Circuit Learner
===================================================

**Module name:** :mod:`qmlt.tf.learner`

.. currentmodule:: qmlt.tf.learner

.. codeauthor:: Maria Schuld <maria@xanadu.ai>

This module contains a class to train_circuit models for machine learning and optimization based on variational quantum circuits.
The class extends tensorflow's Estimator.

The tensorflow learner module has been designed for the training of continuous-variable circuits written in strawberryfields or
blackbird using the 'tf' backend, but is in principle able to train_circuit any user-provided model coded in tensorflow.

Available methods:

.. currentmodule:: qmlt.tf.learner

.. autosummary::
    train_circuit(input_fn[, steps, hooks, max_steps])
    run_circuit(input_fn[, predict_keys, hooks, checkpoint_path])
    score_circuit(input_fn, [steps, hooks, checkpoint_path, name])
    get_circuit_parameters


.. autosummary::
    _qcv_model_fn
    _check
    _check_X
    _check_Y
    _check_steps
    _check_batch_size
    _check_shuffle
    _make_input_fn

-----------------------

"""
# pylint: disable=too-many-branches, too-many-statements, too-many-arguments
import math
import os
import shutil

import numpy as np

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


OPTIMIZER_NAMES = ["Adagrad", "Adam", "Ftrl", "RMSProp", "SGD"]
TASKS = ["optimization", "unsupervised", "supervised"]


def _check_X(X):
    r"""
        Checks if inputs have the right format.
    """
    if X is None:
        return
    if isinstance(X, list):
        if len(np.array(X).shape) > 1:
            return
    if isinstance(X, np.ndarray):
        if len(X.shape) > 1:
            return
    raise ValueError("Input X has to be a list or numpy array of dimension of at least 2. The first dimension is"
                     "the number of inputs (can be 1 or larger) and the second dimension is the size of the inputs."
                     "If you want to feed a single 1-dimensional input a, use [[a]].")


def _check_Y(Y, X):
    r"""
        Checks if targets have the right format.
    """

    if Y is None:
        return

    if X is None:
        raise ValueError("If X is None, Y has to be None too.")

    if isinstance(Y, (list, np.ndarray)):
        if len(X) != len(Y):
            raise ValueError("First dimension {} of X and {} Y have to be of equal size.".format(len(X), len(Y)))
        else:
            return

    raise ValueError("Target Y has to be a list or numpy array of dimension of at least 1. The first dimension is"
                     "the number of targets (can be 1 or larger). Should correspond to the first dimension of X.")


def _check_steps(steps):
    r"""
        Checks if step argument has the right format.
    """
    if steps is None:
        return
    if isinstance(steps, int):
        if steps >= 0:
            return
    else:
        raise ValueError("Steps has to be None or a positive integer (incl 0).")


def _check_batch_size(batch_size, X):
    r"""
        Checks if batch_size argument has the right format.
    """
    if batch_size is None:
        return
    if isinstance(batch_size, int):
        if batch_size > len(X):
            raise ValueError("Batch size cannot be larger than total number of inputs.")
        if batch_size >= 0:
            return
    raise ValueError("Steps has to be None or a positive integer (incl 0).")


def _check_shuffle(shuffle):
    r"""
        Checks if shuffle argument has the right format.
    """
    if isinstance(shuffle, bool):
        return
    else:
        raise ValueError("Shuffle has to be a boolean.")


def _check(hp):
    r"""
    Checks if the hyperparameter dictionary has all required keys, and adds default settings for missing entries.


    The final hyperparameters are printed.

    Args:
        hp (dict): Dictionary of hyperparameters
    """

    user_keys = list(hp.keys())

    def default_regularizer(regularized_params): #pylint: disable=unused-argument
        """Default regularizer is constant"""
        return tf.constant(0., dtype=tf.float32)

    defaults = {'optimizer': 'SGD',
                'regularizer': default_regularizer,
                'init_learning_rate': 0.01,
                'decay': 0.,
                'regularization_strength': 0.1,
                'print_log': True,
                'warm_start': False,
                'batch_size': None,
                'outputs_to_predictions': None,
                'plot_every': 1,
                'log_every': 1,
                'model_dir': None
               }

    default_keys = list(defaults.keys())
    required_keys = ['circuit', 'loss', 'task']
    recognized_keys = default_keys + required_keys

    for key in user_keys:
        if key not in recognized_keys:
            raise ValueError("Key {} is not a valid hyperparameter.".format(key))

    if 'circuit' not in user_keys:
        raise ValueError("No circuit passed to hyperparameters.")

    if 'loss' not in user_keys:
        raise ValueError("No loss passed to hyperparameters.")

    if 'task' not in user_keys:
        raise ValueError("No task passed to hyperparameters.")

    if hp['task'] not in ['supervised', 'unsupervised', 'optimization']:
        raise ValueError("Task not valid.")

    if ('regularizer' in user_keys) and ('regularization_strength' not in user_keys):
        print("Regularizer given, but no regularization strength. Strength is set to 0.1 by default.")

    for key in defaults:
        if key not in user_keys:
            hp[key] = defaults[key]

    if hp['optimizer'] not in OPTIMIZER_NAMES and isinstance(hp['optimizer'], str):
        raise ValueError("Optimizer is {}, but has to be a costom operation or in the list of "
                         "allowed optimizers {}".format(hp['optimizer'], OPTIMIZER_NAMES))
    if hp['optimizer'] == "Ftrl":
        print("Ftrl optimizer has not been tested with the QMLT.")

    if hp['print_log']:
        print("\n----------------------------- \n HYPERPARAMETERS: \n")
        for key in sorted(hp):
            if key == 'circuit':
                print("{} - User defined function.".format(key))
            elif key == 'loss':
                print("{} - User defined function.".format(key))
            elif key == 'regularizer':
                if key in user_keys:
                    print("{} - User defined function.".format(key))
                else:
                    print("{} - No regularizer provided.".format(key))
            else:
                if key in user_keys:
                    print("{} - {}".format(key, hp[key]))
                else:
                    print("{} - {} (default value)".format(key, hp[key]))

        print("\n -----------------------------")

    return hp


def _make_input_fn(X=None, Y=None, steps=None, batch_size=None, shuffle=False):
    r"""
    Creates input function that feeds dictionary of numpy arrays into the model.

    Args:
        X (ndarray or list) - Input data of shape (num_inputs, input_dim).
        steps (int) - Number of steps we train_circuit for with this data queue. Has to be the same than fed into the :func:`train_circuit`
          method. Used to calculate the number of epochs.
        Y (ndarray or list) - labels data of shape (num_inputs, label_dim).
        batch_size (int) - Number of inputs used per training step. If None, we use the full training set in each
          step ("full batch").
        shuffle (boolean) - Whether to shuffle the data queue. Recommended only for training.

    Returns:
        tensorflow input_fn data queue.
    """

    if X is None:
        # Hack: Make dummy input for optimization tasks
        return tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array([[1.]]).astype(np.float32)},
            y=None,
            batch_size=1,
            num_epochs=1,
            shuffle=False)
    else:
        if batch_size is None:
            batch_size = len(X)

        if steps is None:
            steps = 1

        num_epochs = math.ceil(steps*batch_size / len(X))

        X = np.array(X)
        if Y is not None:
            Y = np.array(Y)

        return tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=Y,
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=shuffle)


def _qcv_model_fn(features, labels, hyperparams, mode):
    """
    Custom model for tensorflow's Estimator class. Defines the cost, optimiser and core model of the circuit learner,
    depending on the mode (TRAIN, EVAL or PREDICT). Refer to tensorflow documentation on
    how to build custom models inheriting from tf.Estimator for details.

    Args:
        features (dict): Dictionary defined via a tensorflow input_fn function. The inputs of the current batch are
            accessible through the ``x`` key.
        labels (input queue): Labels defined via a tensorflow input_fn function. The labels of the current batch are
            directly accessible through ``labels``.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        tf.estimator.EstimatorSpec: Measures and outputs in the form of an EstimatorSpec.
    """

    circuit = hyperparams['circuit']
    task = hyperparams['task']
    myloss = hyperparams['loss']
    myregularizer = hyperparams['regularizer']
    reg_strength = hyperparams['regularization_strength']
    myoptimiser = hyperparams['optimizer']
    init_learning_rate = hyperparams['init_learning_rate']
    decay = hyperparams['decay']
    outputs_to_predictions = hyperparams['outputs_to_predictions']
    plot_every = hyperparams["plot_every"]
    model_dir = hyperparams["model_dir"]
    batch_size = hyperparams["batch_size"]

    X = features['x']

    if task == 'optimization':
        # Hack: multiply with dummy 1-batch [[1.]] to make dataqueue send stopping signal
        if mode == 'infer':
            circuit_out = circuit()
            outps = tf.expand_dims(circuit_out, axis=0)* tf.cast(X[0, 0], dtype=circuit_out.dtype)
        else:
            outps = circuit()

    elif task == 'unsupervised':
        if batch_size is not None:
            # Hack: Fix dynamic batch dimension because SF circuit cannot deal with dynmaic batch
            shpX = X.get_shape().as_list()
            shpX[0] = batch_size
            X.set_shape(shpX)

        if mode == 'infer':
            circuit_out = circuit()
            outps = tf.expand_dims(circuit_out, axis=0) * tf.cast(X[0, 0], dtype=circuit_out.dtype)
        else:
            outps = circuit()

    else:
        if batch_size is not None:
            # Hack: Fix dynamic batch dimension because SF circuit cannot deal with dynmaic batch
            shpX = X.get_shape().as_list()
            shpX[0] = batch_size
            X.set_shape(shpX)
            if labels is not None:
                shpY = labels.get_shape().as_list()
                shpY[0] = batch_size
                labels.set_shape(shpY)
        outps = circuit(X=X)

    circuit_outputs = {"outputs": outps}

    if outputs_to_predictions is not None:
        preds = outputs_to_predictions(outps)
        preds = tf.identity(preds, name='predictions')

    global_step = tf.train.get_global_step()
    global_step = tf.identity(global_step, name="global_step")

    cost = None
    train_op = None
    eval_metric = None

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:

        if task == 'optimization':
            loss = myloss(circuit_output=outps)

        elif task == 'unsupervised':
            loss = myloss(circuit_output=outps, X=X)

        else:
            loss = myloss(circuit_output=outps, targets=labels)

        loss = tf.identity(loss, name='loss')
        tf.summary.scalar(name='loss', tensor=loss)

        weights_to_regularize = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='regularized')
        regul = reg_strength*myregularizer(weights_to_regularize)
        regul = tf.identity(regul, name='regularization')
        tf.summary.scalar(name='regularization', tensor=regul)
        cost = tf.add(loss, regul, name='cost')

    if mode == tf.estimator.ModeKeys.EVAL:
        if outputs_to_predictions is not None:
            eval_metric = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=preds)}

    if mode == tf.estimator.ModeKeys.PREDICT:

        if outputs_to_predictions is not None:
            circuit_outputs['predictions'] = preds

    if mode == tf.estimator.ModeKeys.TRAIN:

        adapt_learning_rate = tf.train.inverse_time_decay(init_learning_rate,
                                                          global_step,
                                                          decay_steps=10,
                                                          decay_rate=decay,
                                                          name="learn_rate")
        train_op = tf.contrib.layers.optimize_loss(
            loss=cost,
            global_step=tf.train.get_global_step(),
            learning_rate=adapt_learning_rate,
            optimizer=myoptimiser)

    summary_hook = tf.train.SummarySaverHook(
        plot_every,
        output_dir=model_dir,
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=circuit_outputs,
        loss=cost,
        train_op=train_op,
        eval_metric_ops=eval_metric,
        training_hooks=[summary_hook]
        )


class CircuitLearner(Estimator):
    """
    Defines a learner based on tensorflow and automatic differentiation by extending
    a tensorflow custom estimator class. The extension is done by overwriting model_fn
    (the core model of the learner) to depend on a custom tensorflow model provided by the user.

    Args:
        hyperparams (dict): Dictionary of the following keys:

            * ``circuit`` (function): Function that computes the output of the variational circuit with the
              following keywords:

              * If *task='optimization'* use ``circuit()``
              * If *task='unsupervised'* use ``circuit()``
              * If *task='supervised'* use ``circuit(X)``

              Here, X is a batch of training inputs (2-d tensor).

            * ``task`` (str): One of 'optimization', 'unsupervised' or 'supervised'.

            * ``optimizer`` (string): One of "Adagrad", "Adam", "Ftrl", "RMSProp", "SGD", or a tensorflow
              optimizer class instance. Defaults to SGD.

            * ``loss`` (function): Loss function that outputs a scalar which measures the quality of a model.
              Default is a lambda function that returns zero. The function must have the following keywords:

              * If *task='optimization'*, use ``myloss(circuit_output)``
              * If *task='unsupervised'*, use ``myloss(circuit_output, X)``
              * If *task='supervised'*, use ``myloss(circuit_output, targets)``

              Here, ``outputs`` are the outputs of the circuit function (is a 2-d batch for unsupervised and supervised
              tasks), ``inputs`` is a 2-d ndarray representing a batch of inputs, and ``targets`` are the target outputs.

            * ``regularizer`` (function): Regularizer function of the form

              * ``myregularizer(regularized_params)``

              that maps a 1-d list of circuit parameters marked for regularization to a scalar.
              Default is a lambda function that returns zero.

            * ``regularization_strength`` (float): Strength of regularization. Defaults to 0.

            * ``init_learning_rate`` (float): Initial learning rate used in some optimizers. Defaults to 0.

            * ``decay`` (float): Reduce the learning rate to 1/(1+decay*step). Defaults to 0.

            * ``print_log`` (boolean): If true, prints out information on settings and training progress. Defaults to
              True.

            * ``log_every`` (int): Print log only every log_every steps. Defaults to 1.

            * ``warm_start`` (boolean): If true, load the initial parameters from the last stored model. If False,
              delete the logging directory. Defaults to False.

            * ``plot`` (int): Plot in tensorboard every so many steps. Defaults to 1.

        model_dir (str): Relative path to directory in which model and tensorboard logs get saved. None saves model in temporary directory (default: None).

        config (tf.RunConfig): Configurations for training. Defaults to None.


    """

    def __init__(self, hyperparams, model_dir=None, config=None):

        _check(hyperparams)

        self.hp = hyperparams

        if model_dir is None:
            model_dir = "logsAUTO"

        hyperparams['model_dir'] = model_dir

        if hyperparams['print_log']:
            tf.logging.set_verbosity(tf.logging.INFO)
        else:
            tf.logging.set_verbosity(tf.logging.ERROR)

        if not hyperparams['warm_start']:
            shutil.rmtree(model_dir, ignore_errors=True)

        def _model_fn(features, labels, mode):
            """ Defines the custom model """
            return _qcv_model_fn(features=features,
                                 labels=labels,
                                 hyperparams=hyperparams,
                                 mode=mode)

        super(CircuitLearner, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config
            )

    def train_circuit(self, X=None, Y=None, steps=None, batch_size=None, shuffle_data=False,
                      tensors_to_log=None):
        """ Simple version of the :func:`tf.Estimator.train_circuit` function. Calls :func:`tf.estimator.train`
        internally.

        Args:
            X (ndarray): Array of feature vectors.
            Y (ndarray): Array of target outputs.
            steps (int): Maximum number of steps of the algorithm.
            batch_size (int): Size of training batches. Has to be smaller than size of training set.
            shuffle_data (boolean): If true, shuffle data in each iteration.
            tensors_to_log (dict): Dictionary of tensors to log. The key is a name displayed during logging,
              and the value is the name of the tensor (as a string).
        """

        input_fn = _make_input_fn(X=X, Y=Y, steps=steps,
                                  batch_size=batch_size,
                                  shuffle=shuffle_data)

        _check_X(X)
        _check_Y(Y, X)
        _check_steps(steps)
        _check_batch_size(batch_size, X)
        _check_shuffle(shuffle_data)

        if X is not None:
            if batch_size is not None:
                self.hp['batch_size'] = batch_size
            else:
                self.hp['batch_size'] = len(X)

        # Make loggging hooks
        log = {}

        if self.hp['print_log']:
            default_log = {'Loss': 'loss',
                           'Cost': 'cost',
                           'Regularization': 'regularization',
                           'Step': 'global_step'}
            log.update(default_log)

            if tensors_to_log is not None:
                log.update(tensors_to_log)

            myhooks = [tf.train.LoggingTensorHook(tensors=log, every_n_iter=self.hp['log_every'])]

        else:
            myhooks = None

        self.train(input_fn=input_fn, hooks=myhooks, steps=steps)

    def run_circuit(self, X=None, outputs_to_predictions=None):
        """Get the outcomes when running the circuit with the current circuit parameters.
        Calls :func:`tf.estimator.predict` internally.

        Args:
            X (ndarray): Array of inputs (only for supervised learning tasks).
            outputs_to_predictions (function): Function of the form ``outputs_to_predictions(outps)`` that takes a
              single output and maps it to a prediction that can be compared to the targets in order to compute
              the accuracy of a classification task. If None, run_circuit will return the outputs only.

        Returns:
            Dictionary: Dictionary of outcomes. Always contains the key 'outputs' which is the argument returned
            by the circuit function.
        """

        _check_X(X)

        if (X is not None) and (self.hp['task'] in ['optimization', 'unsupervised']):
            raise ValueError("Do not feed input X for {} tasks.".format(self.hp['task']))
        if (X is None) and (self.hp['task'] in ['supervised']):
            raise ValueError("Please provide input X for {} tasks.".format(self.hp['task']))

        if X is not None:
            self.hp['batch_size'] = len(X)
        if outputs_to_predictions is not None:
            self.hp['outputs_to_predictions'] = outputs_to_predictions

        input_fn = _make_input_fn(X=X)
        outcomes_split = list(self.predict(input_fn=input_fn))

        outcomes = {}
        if 'predictions' in outcomes_split[0]:
            predictions = [o['predictions'] for o in outcomes_split]
            outcomes.update({'predictions': predictions})
        outputs = [o['outputs'] for o in outcomes_split]
        if len(outputs) == 1:
            outputs = outputs[0]
        outcomes.update({'outputs': outputs})

        return outcomes

    def score_circuit(self, X=None, Y=None, outputs_to_predictions=None):
        """Get the score of the circuit. Calls :func:`tf.estimator.evaluate` internally.

        Args:
            X (ndarray): Array of inputs.
            Y (ndarray): Array of targets.
            outputs_to_predictions (function): Function of the form ``outputs_to_predictions(outps)`` that takes a
              single output and maps it to a prediction that can be compared to the targets in order to compute
              the accuracy of a classification task. If None, no 'accuracy' is added to score metrics.

        Returns:
            Dictionary: Dictionary of scores.
        """

        _check_X(X)
        _check_Y(Y, X)

        if (self.hp['task'] == 'optimization') and ((X is not None) or (Y is not None)):
            raise ValueError("Do not feed inputs X or targets Y for optimization tasks.")
        if (self.hp['task'] == 'unsupervised') and (X is None):
            raise ValueError("Please provide inputs X for unsupervised tasks.")
        if (self.hp['task'] == 'supervised') and ((X is None) or (Y is None)):
            raise ValueError("Please provide inputs X and targets Y for supervised tasks.")

        if X is not None:
            self.hp['batch_size'] = len(X)
        if outputs_to_predictions is not None:
            self.hp['outputs_to_predictions'] = outputs_to_predictions

        input_fn = _make_input_fn(X=X, Y=Y)

        # Hack: steps = 1 interrups evaluation without waiting for data queue
        score = self.evaluate(input_fn=input_fn, steps=1)

        return score

    def get_circuit_parameters(self, only_print=False):
        """Print or return all trainable variables.

        Args:
            only_print (boolean): If True, print the variables and return nothing. Defaults to False.

        Returns:
            Dictionary or None: If only_print is False, return return a dictionary of variable names and values.

        """
        # ## Only works for tf 1.8:
        #
        #names = self.get_variable_names()
        #values = [self.get_variable_value(n) for n in names]
        #weights = dict(zip(names, values))

        weight_names = tf.train.list_variables(self.model_dir)
        weights = {}
        for w in weight_names:
            weights[w[0]] = tf.train.load_variable(ckpt_dir_or_file=self.model_dir, name=w[0])

        if only_print:
            for w in weights:
                print(w, ": ", weights[w])
                return None
        else:
            return weights
