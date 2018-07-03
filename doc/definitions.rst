.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _definitions:

Definitions
===========

.. sectionauthor:: Maria Schuld <maria@xanadu.ai>


The machine learning toolbox uses a number of terms and concepts from machine learning that can have variuous meanings in different textbook. This little glossary gives some pointers of how we use them here.

.. glossary::

	Automatic differentiation
		A programming technique that associates each computational operation with a gradient taken with respect to some user-defined variables. This allows the analytic derivative of a variational circuit output to be retrieved without the need to numerically calculate the gradients on paper. A popular framework that supports automatic differentiation is Tensorflow.

		.. seealso:: :ref:`automatic_training` for more details. Automatic differentiation is supported by :mod:`qmlt.tf`.

	Epoch
		One run through all training data during training. One epoch is equivalent to a number of steps equal to (number of training inputs)/(batch size).

	Circuit parameters
		Trainable parameters of the quantum circuit.

	Cost
		The overall objective. In the QMLT teminology, the cost is the sum of the loss and the regularization. The goal of optimizing a variational circuit is to minimize the loss.

	Hyperparameters
		Configuration settings for the model and the training algorithm

	Learning rate
		Step size in the gradient updates. Can depend on the step or the value of the gradient if the learning rate is adaptive.

	Logging
		Printing out information on the values of variables during training.

	Loss
		The first term in the objective that measures how good a model is.

	Model directory
		Folder in which training logs and model is saved

	Monitoring
		Tagging a variable for visualization by plotting.

	Numerical differentiation
		The use of numerical methods to compute approximations to the gradient, for instance `finite differences <https://en.wikipedia.org/wiki/Finite_difference>`_ methods.

		.. seealso:: :ref:`numerical_differentation` for more details. Numerical differentiation is supported by :mod:`qmlt.numerical`.

	Regularization
		A penalty term in the cost function that depends on the parameters. Can be used to keep certain parameters low, or to limit the model flexibility and prevent overfitting.

	Step
		One iteration of the parameter update during optimization.

	Stochastic gradient descent
		Gradient descent optimization in which a batch of training inputs is sampled in each step. Introduces stochasticity to the training. As a consequence, the cost is not necessarily decreased in each step.

	Warm start
		Starting training with the parameters that the previous training ended with.

