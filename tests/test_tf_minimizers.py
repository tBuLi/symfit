from __future__ import division, print_function
import pytest
import warnings

import numpy as np
import pickle
import multiprocessing as mp
import tensorflow as tf

from symfit import (
    Variable, Parameter, Fit
)
from symfit.core.minimizers import *
from symfit.core.tf_minimizers import TFBFGS, TFDifferentialEvolution

def test_TFBFGS():
    """
    Compare the results of Tensorflow BFGS vs scipy BFGS
    """
    # Create test data
    xdata = np.linspace(0, 100, 25)  # From 0 to 100 in 100 steps
    a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
    b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
    ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

    # Normal symbolic fit
    a = Parameter('a', value=0.0)
    b = Parameter('b', value=0.0)
    x = Variable('x')
    y = Variable('y')
    model = {y: a * x + b}

    fit = Fit(model, xdata, ydata, minimizer=BFGS)
    fit_result = fit.execute()

    xdata = tf.constant(xdata, dtype=tf.float32)
    ydata = tf.constant(ydata, dtype=tf.float32)
    fit = Fit(model, xdata, ydata, minimizer=TFBFGS)
    tf_fit_result = fit.execute()

    assert pytest.approx(fit_result.value(a), tf_fit_result.value(a))
    assert pytest.approx(fit_result.value(b), tf_fit_result.value(b))

def test_TFDifferentialEvolution():
    """
    Compare the results of Tensorflow BFGS vs scipy BFGS
    """
    # Create test data
    xdata = np.linspace(0, 100, 25)  # From 0 to 100 in 100 steps
    a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
    b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
    ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

    # Normal symbolic fit
    a = Parameter('a', value=1.0, min=0.0, max=100.0)
    b = Parameter('b', value=1.0, min=0.0, max=100.0)
    x = Variable('x')
    y = Variable('y')
    model = {y: a * x + b}

    fit = Fit(model, xdata, ydata, minimizer=DifferentialEvolution)
    fit_result = fit.execute()

    # For TFDifferentialEvolution we need to cast from numpy array to a tensor of
    # tf.float32, and an extra dimension needs to be added to the datasets because tf
    # Will call it with an array for each parameter, of size `population_size`
    xdata = tf.constant(xdata, dtype=tf.float32)[..., None]
    ydata = tf.constant(ydata, dtype=tf.float32)[..., None]
    fit = Fit(model, xdata, ydata, minimizer=TFDifferentialEvolution)
    tf_fit_result = fit.execute()

    assert pytest.approx(fit_result.value(a), tf_fit_result.value(a))
    assert pytest.approx(fit_result.value(b), tf_fit_result.value(b))