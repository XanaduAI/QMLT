.. _installation:

Installation and Downloads
#################################

Dependencies
============

Before installing the Quantum Machine Learning Toolbox, the following dependencies need to be installed:

* `Python <http://python.org/>`_ >= 3.5
* `matplotlib <https://matplotlib.org/>`_ >= 2.0
* `scikit-learn <http://scikit-learn.org/stable/>`_ >= 0.19
* `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`_ >= 0.7.3

These packages will be installed automatically if using ``pip``, see below for details. For additional information on Strawberry Fields, please see the Strawberry Fields documentation.


Installation
============
.. highlight:: console

Installation of the Xanadu Quantum Machine Learning Toolbox, as well as all required Python packages mentioned above, can be done via pip:
::

    $ python -m pip install qmlt

Make sure you are using the Python 3 version of pip.


Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

If using Ubuntu, they can be installed via a combination of ``apt`` and ``pip``:
::

	$ pip3 install sphinx --user
	$ pip3 install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run the command
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
