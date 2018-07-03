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
Numerical Circuit learner

"""
# pylint: disable=too-many-branches, too-many-statements, too-many-arguments
import os
from string import ascii_letters, digits
from random import choice
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime


OPTIMIZER_NAMES = ["SGD", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
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
    raise ValueError("Input X has to be a list or numpy array of dimension of at least 2. The first dimension is "
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
        if len(np.array(Y).shape) >= 1:
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


def _check_logs(logs):
    r"""
        Checks if logs argument has the right format.
    """
    if isinstance(logs, dict):
        for key in logs:
            if ' ' in key:
                raise ValueError("The name (key) of a logged variable cannot contain white spaces.")
        return

    raise ValueError("The second output of the circuit function has to be a dictionary.")


def _check(hp):
    r"""
    Checks if the hyperparameter dictionary has all required keys, and adds default settings for missing entries.

    The final hyperparameters are printed.

    Args:
        hp (dict): Dictionary of hyperparameters
    """

    user_keys = list(hp.keys())

    defaults = {'optimizer': 'SGD',
                'regularizer': lambda regularized_params: 0,
                'init_learning_rate': 0.01,
                'decay': 0.,
                'adaptive_learning_rate_threshold': 0.,
                'regularization_strength': 0.1,
                'print_log': True,
                'plot': False,
                'log_every': 1,
                'warm_start': False,
                'epsilon': 1e-6}

    default_keys = list(defaults.keys())
    required_keys = ['circuit', 'loss', 'task', 'init_circuit_params']
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

    if 'init_circuit_params' not in user_keys:
        raise ValueError("No initial circuit parameters passed to hyperparameters.")

    init = hp['init_circuit_params']
    if isinstance(hp['init_circuit_params'], int):
        hp['init_circuit_params'] = [{} for _ in range(init)]

    init = hp['init_circuit_params']
    is_list_of_dics = isinstance(init, list) and all([isinstance(par, dict) for par in init])

    if not is_list_of_dics:
        raise ValueError("Initial circuit parameters have to be a list of dictionaries or an integer, got {}"
                         ".".format(type(init)))
    for par in init:
        if 'val' not in par:
            par['val'] = np.random.normal(loc=0., scale=0.1)
        if 'name' not in par:
            par['name'] = ''.join(choice(ascii_letters + digits) for _ in range(5))
        if 'regul' not in par:
            par['regul'] = False
        if 'monitor' not in par:
            par['monitor'] = False
        if par['regul']:
            name = par['name']
            if "regularized" not in name:
                par['name'] = "regularized/"+name

    if ('regularizer' in user_keys) and ('regularization_strength' not in user_keys):
        print("Regularizer given, but no regularization strength. Strength is set to 0.1 by default.")

    # add defaults
    for key in defaults:
        if key not in user_keys:
            hp[key] = defaults[key]

    if hp['optimizer'] not in OPTIMIZER_NAMES and isinstance(hp['optimizer'], str):
        raise ValueError("Optimizer is {}, but has to be in list of "
                         "allowed optimizers {}".format(hp['optimizer'], OPTIMIZER_NAMES))

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
            elif key == 'init_circuit_params':
                vals = [param['val'] for param in hp['init_circuit_params']]
                print("{} - Values: {}.".format(key, vals))
            else:
                if key in user_keys:
                    print("{} - {}".format(key, hp[key]))
                else:
                    print("{} - {} (default value)".format(key, hp[key]))

        print("\n -----------------------------")

    return hp


class CircuitLearner:
    r"""Defines a circuit learner based on numerical differentiation.
    The core model is a variational quantum circuit provided by the user.

    Args:
        hyperparams (dict): Dictionary of the following keys:

            * ``circuit`` (python function): Function that computes the output of the variational circuit with the
              following keywords:

              * If *task='optimization'* use ``circuit(params)``
              * If *task='unsupervised'* use ``circuit(params)``
              * If *task='supervised'* use ``circuit(X, params)``

              Here, ``params`` is a list of scalar circuit parameters. X is an ndarray representing a
              batch of training inputs. The name of the function is not mandatory.
              The function can return a dictionary of variables to log as a second argument, where the key is the name to be
              displayed during logging, and the value is the variable itself.

            * ``init_circuit_params`` (int or list of dictionaries): If int is given, it is interpreted as the number
              of circuit parameters that are generated with the default values. If list of dictionaries is given, each
              dictionary represents a circuit parameter and specifies the following optional keys:

              * 'init' (float): Initial value of this parameter. Defaults to normally distributed parameter
                with variance 0.1 and mean 0.
              * 'name' (string): Name given to this parameter. Defaults to random string
              * 'regul' (boolean): Indicates whether this parameter is regularized. Defaults to False.
              * 'monitor' (boolean): Indicates whether this parameter is monitored for visualisation. Defaults to False.

            * ``task`` (str): One of 'optimization', 'unsupervised' or 'supervised'.

            * ``loss`` (python function): Loss function that outputs a scalar which measures the quality of a model.
              Default is a lambda function that returns zero. The name of the function is not mandatory.
              The function must have the following keywords:

              * If *task='optimization'*, use ``myloss(circuit_output)``
              * If *task='unsupervised'*, use ``myloss(circuit_output, X)``
              * If *task='supervised'*, use ``myloss(circuit_output, targets)``

              Here, ``circuit_output`` is the (first) output of :func:`circuit`, ``inputs`` is a 2-d ndarray representing a batch of inputs,
              and ``targets`` are the target outputs.

            * ``optimizer`` (string): 'SGD' or name of these optimizers accepted by scipy's
              :func:`scipy.minimize` method: 'Nelder-Mead', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'.
              Defaults to 'SGD'.

            * ``regularizer`` (function): Regularizer function of the form

              * ``myregularizer(regularized_params)``

              that maps a 1-d list of circuit parameters marked for regularization to a scalar. The name of the function
              is not mandatory. Default is a lambda function that returns zero.

            * ``regularization_strength`` (float): Strength of regularization. Defaults to 0.

            * ``init_learning_rate`` (float): Initial learning rate used if optimizer='SGD'. Defaults to 0.1.

            * ``decay`` (float): Reduce the learning rate to 1/(1+decay*step) in each SGD step. Defaults to 0.

            * ``adaptive_learning_rate_threshold=0`` (float): If optimizer='SGD', and if all gradients are smaller than
              this value, multiply learning rate by factor 10. Defaults to 0.

            * ``print_log`` (boolean): If true, prints out information on settings and training progress. Defaults to
              True.

            * ``log_every`` (int): Log results to file every log_every training step. Defaults to 1.

            * ``warm_start`` (boolean): If true, load the initial parameters from the last stored model. Defaults to
              False.

            * ``epsilon`` (float): Step size for finite difference method for SGD optimizer. Defaults to 1e-06.

            * ``plot`` (boolean): If True, plot default values and monitored circuit parameters
              every log_every steps. If False, do not plot. Defaults to False.

        model_dir (str): Relative path to directory in which the circuit parameters are saved. If the directory
         does not exist, it is created. Defaults to 'logsNUM/' in current working directory.


    """

    def __init__(self, hyperparams, model_dir="logsNUM/"):

        _check(hyperparams)

        self._circuit_params = hyperparams['init_circuit_params']
        self._num_params = len(self._circuit_params)
        self._regularized_params_idx = [idx for idx in range(self._num_params) if self._circuit_params[idx]['regul']]
        self._monitored_params_idx = [idx for idx in range(self._num_params) if self._circuit_params[idx]['monitor']]
        self._model_dir = model_dir
        self._hp = hyperparams

        if not os.path.exists(os.path.dirname(self._model_dir)):
            try:
                os.makedirs(os.path.dirname(self._model_dir))
            except OSError: # pragma: no cover
                print("Model directory to store logs and params cannot be created.")
                raise

    def train_circuit(self, X=None, Y=None, steps=None, batch_size=None, seed=None):
        """Train the learner, optionally using input data X and target outputs Y.

        Args:
            X (ndarray): Array of inputs.
            Y (ndarray): Array of targets.
            steps (int): Maximum number of steps of the algorithm.
            batch_size (int): Number of training inputs that are subsampled and used in each training
              step. Must be smaller than the first dimension of X.
            seed (float): Seed for sampling batches of training data in each SGD step.
        """

        _check_X(X)
        _check_Y(Y, X)
        _check_steps(steps)
        _check_batch_size(batch_size, X)

        # Unpack the hyperparameters
        circuit = self._hp["circuit"]
        task = self._hp["task"]
        myloss = self._hp["loss"]
        myoptimizer = self._hp["optimizer"]
        myregularizer = self._hp["regularizer"]
        regul_strength = self._hp["regularization_strength"]
        init_lr = self._hp["init_learning_rate"]
        decay = self._hp['decay']
        adaptive_lr = self._hp['adaptive_learning_rate_threshold']
        print_log = self._hp["print_log"]
        log_every = self._hp["log_every"]
        warm_start = self._hp["warm_start"]
        eps = self._hp["epsilon"]
        plot = self._hp["plot"]

        if plot: # pragma: no cover
            try:
                from matplotlib import pyplot as plt
                from .plot import plot_all
            except ImportError:
                raise ImportError("For live plots of the training progress or monitored parameters, "
                                  "matplotlib must be installed")

        def loss_and_logs(params, inputs, targets):
            """Compute the loss and retrieve logs, details depending on the task"""
            logs = {}
            if task == 'optimization':
                outps = circuit(params=params)
                if isinstance(outps, tuple):
                    outps, logs = outps
                    _check_logs(logs)
                loss = myloss(circuit_output=outps)

            elif task == 'unsupervised':
                outps = circuit(params=params)
                if isinstance(outps, tuple):
                    outps, logs = outps
                    _check_logs(logs)
                loss = myloss(circuit_output=outps, X=inputs)

            else:
                outps = circuit(X=inputs, params=params)
                if isinstance(outps, tuple):
                    outps, logs = outps
                    _check_logs(logs)
                loss = myloss(circuit_output=outps, targets=targets)

            return loss, logs

        def regularizer(circuit_params):
            """Compute the regularization term"""
            regularized_params = circuit_params[self._regularized_params_idx]
            return regul_strength*myregularizer(regularized_params)

        # Load circuit parameters and global step if warm starting, or use provided values if not
        if warm_start:
            try:
                init_params = np.loadtxt(self._model_dir+"circuit_params.txt", ndmin=1)
            except IOError as e:
                print("I/O error({0}): {1}. No model weights for warm start found.".format(e.errno, e.strerror))
                raise
            if len(self._circuit_params) != len(init_params):
                raise IOError("Saved model has a different number of parameters than current model. "
                              "Cannot do a warm start")
            try:
                first_global_step = int(np.loadtxt(self._model_dir+"/global_step.txt"))
            except IOError as e: # pragma: no cover
                print("I/O error({0}): {1}. No global_step for warm start found.".format(e.errno, e.strerror))
                raise

        else:
            init_params = np.array([w['val'] for w in self._circuit_params])
            first_global_step = 0

        init_loss, init_log = loss_and_logs(init_params, X, Y)
        init_regul = regularizer(init_params)
        init_cost = init_loss + init_regul
        if print_log:
            print("Initial loss = {} | Initial regularizer = {} | "
                  "".format(init_loss, init_regul), end=" ")
            for key, value in init_log.items():
                print("{} = {} | ".format(key, value), end=" ")
            print("\n")

        if not warm_start:
            with open(self._model_dir + 'log.csv', 'w') as f_log:
                values_to_write = ["global_step", "cost", "loss", "regul", "learning_rate"]

                for key, value in init_log.items():
                    values_to_write.append(key)

                params_to_monitor = [p['name'] for p in self._circuit_params if p['monitor']]
                for p in params_to_monitor:
                    values_to_write.append(p)

                f_log.write(", ".join([str(i) for i in values_to_write]))
                f_log.write("\n")

        with open(self._model_dir + 'log.csv', 'a') as f_log:
            values_to_write = [0, init_cost, init_loss, init_regul, init_lr]
            for key, value in init_log.items():
                values_to_write.append(value)
            params_to_monitor = init_params[self._monitored_params_idx]
            for p in params_to_monitor:
                values_to_write.append(p)

            f_log.write(", ".join([str(i) for i in values_to_write]))
            f_log.write("\n")

        # Implementation of SGD optimizer
        if myoptimizer == "SGD":

            temp_params = init_params

            step = first_global_step
            for step in range(first_global_step, first_global_step + steps):

                # Make training data batches for optimization task
                if Y is None and X is None:
                    X_batch = None
                    Y_batch = None
                # Make training data batches for unsupervised task
                if Y is None and X is not None:
                    if batch_size is None:
                        batch_size = len(X)
                    np.random.seed(seed)
                    rnd = np.random.permutation(len(X))
                    X_batch = X[rnd[: batch_size]]
                    Y_batch = None
                # Make training data batches for supervised task
                if Y is not None and X is not None:
                    if batch_size is None:
                        batch_size = len(X)
                    np.random.seed(seed)
                    rnd = np.random.permutation(len(Y))
                    X_batch = X[rnd[: batch_size]]
                    Y_batch = Y[rnd[: batch_size]]

                def cost(circuit_params):
                    """Compute the cost"""
                    loss, _ = loss_and_logs(circuit_params, X_batch, Y_batch)
                    regul = regularizer(circuit_params)
                    return loss + regul

                # Compute gradients with scipy's approx_fprime method (finite difference method)
                grad = approx_fprime(temp_params, cost, eps)
                # Compute the current learning rate
                decayed_lr = init_lr/(1+decay*step)
                # boost learning rate if gradients are small
                if max(abs(np.array(grad))) < adaptive_lr:
                    decayed_lr = decayed_lr*10

                # define the update
                update = np.array([- decayed_lr * g for g in grad])
                # update the circuit parameters
                temp_params = temp_params + update

                if step % log_every == 0:
                    temp_loss, temp_log = loss_and_logs(temp_params, X, Y)
                    temp_regularizer = regularizer(temp_params)
                    temp_cost = temp_loss + temp_regularizer
                    if print_log:
                        print("Global step {} | Loss = {} | Regularizer = {} | "
                              " learning rate = {} | ".format(step+1, temp_loss, temp_regularizer, decayed_lr),
                              end=" ")
                        for key, value in temp_log.items():
                            print("{} = {} | ".format(key, value), end=" ")
                        print("")

                    with open(self._model_dir + 'log.csv', 'a') as f_log:
                        values_to_write = [step+1, temp_cost, temp_loss, temp_regularizer, decayed_lr]

                        for key, value in temp_log.items():
                            values_to_write.append(value)

                        params_to_monitor = temp_params[self._monitored_params_idx]
                        for p in params_to_monitor:
                            values_to_write.append(p)

                        f_log.write(", ".join([str(i) for i in values_to_write]))
                        f_log.write("\n")

                    if plot: # pragma: no cover
                        plt.ion()
                        if step == first_global_step:
                            fig, ax = plot_all(self._model_dir+'log.csv')
                            fig.canvas.set_window_title('Machine Learning Toolbox')
                        else:
                            for a in ax.ravel():
                                a.clear()
                            fig, ax = plot_all(self._model_dir+'log.csv', figax=(fig, ax))
                            fig.canvas.draw()

            final_params = temp_params

            # Save the global step for next warm start
            np.savetxt(self._model_dir+"global_step.txt", np.array([step+1]))

            if plot: # pragma: no cover
                print("Training complete. Close the live plot window to exit.")
                plt.show(block=True)

        # Implementation of mimimize optimizer
        else:

            def objective(circuit_params):
                """Compute the objective for optimization."""
                loss, _ = loss_and_logs(circuit_params, X, Y)
                regul = regularizer(circuit_params)
                return loss + regul

            temp_params = init_params
            for step in range(first_global_step, first_global_step + steps):

                opt = minimize(objective,
                               np.array(temp_params),
                               method=myoptimizer,
                               options={"maxiter": 1, "disp": False})

                temp_params = opt.x

                if step % log_every == 0:
                    temp_loss, temp_log = loss_and_logs(temp_params, X, Y)
                    temp_regularizer = regularizer(temp_params)
                    temp_cost = temp_loss + temp_regularizer
                    if print_log:
                        print("Global step {} | Loss = {} | Regularizer = {} "
                              "".format(step + 1, temp_loss, temp_regularizer),
                              end=" ")
                        for key, value in temp_log.items():
                            print("{} = {} | ".format(key, value), end=" ")
                        print("")

                    with open(self._model_dir + 'log.csv', 'a') as f_log:
                        values_to_write = [step+1, temp_cost, temp_loss, temp_regularizer, '-']

                        for key, value in temp_log.items():
                            values_to_write.append(value)

                        params_to_monitor = temp_params[self._monitored_params_idx]
                        for p in params_to_monitor:
                            values_to_write.append(p)

                        f_log.write(", ".join([str(i) for i in values_to_write]))
                        f_log.write("\n")

                    if plot: # pragma: no cover
                        plt.ion()
                        if step == first_global_step:
                            fig, ax = plot_all(self._model_dir+'log.csv')
                            fig.canvas.set_window_title('Machine Learning Toolbox')
                        else:
                            for a in ax.ravel():
                                a.clear()
                            fig, ax = plot_all(self._model_dir+'log.csv', figax=(fig, ax))
                            fig.canvas.draw()

            final_params = temp_params

            if print_log:
                final_loss, final_log = loss_and_logs(temp_params, X, Y)
                final_regularizer = regularizer(temp_params)
                print("\nFinal loss = {} | Final regularizer = {} \n".format(final_loss, final_regularizer))
                for key, value in final_log.items():
                    print("Final {}: {} ".format(key, value))

        # Replace the global circuit parameters of the model with the final parameters
        for i, w in enumerate(self._circuit_params):
            w['val'] = final_params[i]

        # Save the final params for next warm start
        np.savetxt(self._model_dir+"circuit_params.txt", final_params)

    def run_circuit(self, X=None, outputs_to_predictions=None):
        """Get the outcomes when running the circuit with the current circuit parameters.

        Args:
            X (ndarray): Array of inputs.
            outputs_to_predictions (function): Function of the form ``outputs_to_predictions(outps)`` that takes a
              single output and maps it to a prediction that can be compared to the targets in order to compute
              the accuracy of a classification task. If None, run_circuit will return the outputs only.

        Returns:
            Dictionary: Dictionary of different outcomes. Always contains the key 'outputs' which is the
            (first) argument returned by the circuit function.
        """

        _check_X(X)

        if (X is not None) and (self._hp['task'] in ['optimization', 'unsupervised']):
            raise ValueError("Do not feed input X for {} tasks.".format(self._hp['task']))
        if (X is None) and (self._hp['task'] in ['supervised']):
            raise ValueError("Please provide input X for {} tasks.".format(self._hp['task']))

        circuit = self._hp['circuit']
        task = self._hp['task']
        current_params = [w['val'] for w in self._circuit_params]

        if task == 'optimization' or task == 'unsupervised':
            outps = circuit(params=current_params)
            if isinstance(outps, tuple):
                outps, _ = outps
        else:
            outps = circuit(X=X, params=current_params)
            if isinstance(outps, tuple):
                outps, _ = outps

        outcomes = {'outputs': outps}

        if outputs_to_predictions is not None:
            outcomes['predictions'] = [outputs_to_predictions(outp) for outp in outps]

        return outcomes

    def score_circuit(self, X=None, Y=None, outputs_to_predictions=None):
        """Score the circuit. For unsupervised and supervised learning, the score is computed with regards
        to some input data.

        Args:
            X (ndarray): Array of inputs.
            Y (ndarray): Array of targets.
            outputs_to_predictions (function): Function of the form ``outputs_to_predictions(outps)`` that takes a
              single output and maps it to a prediction that can be compared to the targets in order to compute the accuracy of a classification task.
              If None, no 'accuracy' is added to score metrics.

        Returns:
            Dictionary: Dictionary with score metrics 'cost', 'loss', 'regularization', accuracy (if outputs_to_predictions
            is given) and the logs indicated by custom logging.
        """

        _check_X(X)
        _check_Y(Y, X)

        if (self._hp['task'] == 'optimization') and ((X is not None) or (Y is not None)):
            raise ValueError("Do not feed inputs X or targets Y for optimization tasks.")
        if (self._hp['task'] == 'unsupervised') and (X is None):
            raise ValueError("Please provide inputs X for unsupervised tasks.")
        if (self._hp['task'] == 'supervised') and ((X is None) or (Y is None)):
            raise ValueError("Please provide inputs X and targets Y for supervised tasks.")

        circuit = self._hp["circuit"]
        task = self._hp["task"]
        myloss = self._hp['loss']
        myregularizer = self._hp["regularizer"]
        regul_strength = self._hp["regularization_strength"]

        current_params = np.array([w['val'] for w in self._circuit_params])
        score = {}
        logs = {}

        if task == 'optimization':
            outps = circuit(params=current_params)
            if isinstance(outps, tuple):
                outps, logs = outps
            loss = myloss(circuit_output=outps)

        elif task == 'unsupervised':
            outps = circuit(params=current_params)
            if isinstance(outps, tuple):
                outps, logs = outps
            loss = myloss(circuit_output=outps, X=X)

        elif task == 'supervised':
            outps = circuit(X=X, params=current_params)
            if isinstance(outps, tuple):
                outps, logs = outps
            loss = myloss(circuit_output=outps, targets=Y)
            if outputs_to_predictions is not None:
                predictions = [outputs_to_predictions(outp) for outp in outps]
                accuracy = np.mean(np.array([1 if np.allclose(p, target) else 0 for p, target in zip(predictions, Y)]))
                score['accuracy'] = accuracy

        regularized_params = current_params[self._regularized_params_idx]
        regul = myregularizer(regularized_params)
        cost = loss + regul_strength * regul

        score.update({'cost': cost,
                      'loss': loss,
                      'regularization': regul})
        score.update(logs)

        return score

    def get_circuit_parameters(self, only_print=False):
        """Get the current circuit parameters of the learner.

        Args:
            only_print (boolean): If True, print the variables and return nothing. Defaults to False.

        Returns:
             None or Dictionary: If only_print is False, return a dictionary of variable names and values. Else None.
        """
        current_weights = dict([(w['name'], w['val']) for w in self._circuit_params])
        if only_print:
            for key in current_weights:
                print(key, ": ", current_weights[key])
            print("")
            return None

        return current_weights
