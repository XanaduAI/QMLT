.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _task:


Tasks
=============================================

.. sectionauthor:: Maria Schuld <maria@xanadu.ai>

We define three tasks of variational circuits: 

* optimization, 
* unsupervised learning, and 
* supervised learning. 

All tasks have the goal of optimizing a variational circuit. But while the pure *optimization task* merely intends to find the extremal points of the objective, the goal of learning is to generalise from unlabelled or labelled data. Optimization is therefore a sub-task of learning, while the problem of learning from data exceeds optimization.

To simplify this notion, we define the task depending on the potential data set given with the problem that the QMLT is supposed to solve. If there is no data given, we speak of an optimization task. If the data set is a collection of *inputs* which we assume to be real vectors, we have an unsupervised learning task. If in addition there is a *target label* given for each input, we have a supervised learning task.

|

.. rst-class:: docstable-small

.. table::
   :align: center

   +-----------------------+------------+-----------+
   | Task                  | Inputs     | Targets   |
   +=======================+============+===========+
   | Optimization          | No         | No        |
   +-----------------------+------------+-----------+
   | Unsupervised learning | Yes        | No        |
   +-----------------------+------------+-----------+
   | Supervised learning   | Yes        | Yes       |
   +-----------------------+------------+-----------+

|

Let's go through them one by one.


Optimization
------------

**Optimization** is the most basic application for variational circuits. The circuit parameters are updated with candidates that lead to a better objective until the objective is maximized or minimized, or the maximum number of steps has been reached.

|

.. _fig_opt_var:
.. figure::  _static/opt_var.png
   :align:   center
   :width:   350px

|

Possible applications are

* **Variational quantum eigensolvers** :cite:`peruzzo14`. The output of the circuit is an expectation value with respect to the final state. The objective is to minimize this expectation value. For example, minimizing the energy expectation value, the optimal circuit prepares an approximation to the ground state of a quantum system.
* **Unitary learning**. Here the output of the circuit is a quantum state, and the goal is to learn the circuit that prepares a final state with a certain property. For example, one could maximize the fidelity between the state prepared by the variational circuit and a given target state. One could also use a measure of entanglement as the objective and try to learn a circuit that prepares a maximally entangled state.



Unsupervised learning
----------------------

In **unsupervised learning** the goal is to generalize patterns found in a dataset of input feature vectors.
''Generalizing'' can refer to many different tasks, such as clustering, data compression or generating new data samples.

The data is used to formulate the training objective, which means that the training inputs define the objective
function. A typical objective is maximum likelihood estimation, where one wants to increase the
probability of observing the training data with regards to a model distribution.

|

.. _fig_unsup_var:
.. figure::  _static/unsup_var.png
   :align:   center
   :width:   350px

|

Possible applications are

* **Training of generative models** . A quantum state defines probabilities over basis states (for example Fock states). If we associate these basis states with possible data samples, the quantum state defines a probability distribution over data. Given a set of some data samples drawn from a ''true'' distribution, the goal is to prepare a quantum state that corresponds to a distribution which is as close as possible to this ''true''  distribution. In other words, we have to find a circuit that prepares a quantum state from which measurements sample basis states that are somehow similar to the given data.
* **Quantum clustering**. We want to assign the data samples to n clusters. We can learn a circuit that corresponds to a Hamiltonian such that the potential energy is high in areas of high data density of one class.


Supervised learning
-------------------

In **supervised learning** one wants to generalise the input-output relation of a dataset that contains sample inputs
and target outputs. The variational circuit defines an ansatz of such a input-output function, and the goal is to
learn a circuit which - if fed with new inputs - produced outputs according to the rule with which the data was
produced. This is why here the quantum circuit depends on data inputs and produces outputs that are predictions for
these inputs. The objective compares the predictions with the target outputs and measures how close they are.

|

.. _fig_sup_var:
.. figure::  _static/sup_var.png
   :align:   center
   :width:   350px

|

Possible applications are

* **Quantum models for classification and regression** :cite:`farhi18` :cite:`schuld18cc`: The quantum circuit is an ansatz for a classical machine learning model that is trained to run new inputs.




















