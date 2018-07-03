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
Unit tests for qmlt plots
========================================================

.. codeauthor:: Josh Izaac <josh@xanadu.ai>

"""
import sys
import os
import unittest
import importlib

import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

from qmlt.numerical import plot


class TestSquarish(unittest.TestCase):

    def test_1_plot(self):
        res = plot._squareish(1)
        self.assertEqual(res, (1, 1))

    def test_2_plots(self):
        res = plot._squareish(2)
        self.assertEqual(res, (2, 1))

    def test_3_plots(self):
        res = plot._squareish(3)
        self.assertEqual(res, (3, 1))

    def test_4_plots(self):
        res = plot._squareish(4)
        self.assertEqual(res, (2, 2))

    def test_5_plots(self):
        res = plot._squareish(5)
        self.assertEqual(res, (3, 2))

    def test_6_plots(self):
        res = plot._squareish(6)
        self.assertEqual(res, (3, 2))

    def test_7_plots(self):
        res = plot._squareish(7)
        self.assertEqual(res, (3, 3))

    def test_13_plots(self):
        res = plot._squareish(13)
        self.assertEqual(res, (3, 5))


class TestPlot(unittest.TestCase):

    def test_no_axes(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x)
        res = plot._plot(x, y)
        x_plot, y_plot = res.lines[0].get_xydata().T
        self.assertTrue(np.all(x == x_plot))
        self.assertTrue(np.all(y == y_plot))


    def test_axes(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x)
        res = plot._plot(x, y, ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        self.assertTrue(np.all(x == x_plot))
        self.assertTrue(np.all(y == y_plot))

    def test_xlabel(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x)
        res = plot._plot(x, y, ax, xlabel='Test')
        x_label = res.xaxis.get_label().get_text()
        self.assertEqual('Test', x_label)

    def test_ylabel(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x)
        res = plot._plot(x, y, ax, ylabel='Test')
        y_label = res.yaxis.get_label().get_text()
        self.assertEqual('Test', y_label)


class TestMatplotlibImport(unittest.TestCase):

    def test_import_error(self):
        with self.assertRaisesRegex(ImportError, 'To use the plotting'):
            del sys.modules['matplotlib']
            del sys.modules['matplotlib.pyplot']
            importlib.reload(plot)

if __name__ == "__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in [TestSquarish, TestPlot, TestMatplotlibImport]:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

