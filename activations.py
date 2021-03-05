"""
Module implementing activation functions like step function, sigmoid function.
"""
import numpy as np


def step(x):
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

def sigmoid(x):
    """
    Sigmoid function also known as logistic function. S-shaped function bound
    between 0 and 1 which approaches 1 when input is very positive and 0 when
    input is very negative.

    :param x: input to sigmoid function
    :return: output of sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    The derivative of the sigmoid function evaluated at x.

    :param x: the point at which to evaluate the derivative
    :return: the derivative
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """
    Hyperbolic tangent
    :param x: input
    :return: output
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    """
    Derivative of hyperbolic tangent
    :param x: input
    :return: output
    """
    return 1 - tanh(x) ** 2


def leaky_relu(x):
    """
    Leaky RELU.

    :param x: input (can be vector form)
    :return: output
    """
    return np.where(x >= 0, x, x * 0.01)


def leaky_relu_derivative(x):
    """
    Derivative of leaky RELU.

    :param x: input
    :return: output
    """
    return np.where(x >= 0, 1, 0.01)


def softmax(a):
    """
    Softmax calculator.

    :param x: X is an array of values representing the output (w * x + b) for
    each of the output neurons.
    :return:
    """
    shiftx = a - np.max(a)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
