"""
Module implementing activation functions like step function, sigmoid function.
"""
import numpy as np


def step_function(x):
    """
    Unit step function.
    0 if x < 0 and 1 if x >= 0

    :param x: input to unit step function
    :return: output of unit step function
    """
    if x >= 0:
        return 1
    else:
        return 0


def sigmoid_function(x):
    """
    Sigmoid function also known as logistic function. S-shaped function bound
    between 0 and 1 which approaches 1 when input is very positive and 0 when
    input is very negative.

    :param x: input to sigmoid function
    :return: output of sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))
