import numpy as np
from src.activations import sigmoid_derivative


class LogLikelihood:
    """
    Class implementing log-likelihood cost function
    """
    @staticmethod
    def get_cost(data, network):
        """

        :param data:
        :param predictions:
        :return:
        """

        total = 0
        for x, y in data:
            output = network.feedforward(x)
            total += np.dot(y.reshape(y.shape[0]), np.log(output).reshape(output.shape[0]))
        return - (total / len(data))

    @staticmethod
    def get_delta(a, y, z):
        """
        Method to return delta for the output layer.

        :param a: predicted outcome
        :param y: actual outcome
        :return: delta (error) for output layer
        """
        return a-y


class Quadratic:
    """
    Quadratic cost function. ONLY FOR SIGMOID AT OUTPUT LAYER.
    """
    @staticmethod
    def get_cost(data, network):
        """
        The cost calculated over all the data points

        :param data:
        :param network: the neural network
        :return:
        """
        total = 0
        for x, y in data:
            output = network.feedforward(x)
            total += 0.5 * (np.linalg.norm(output - y) ** 2)
        return - (total / len(data))

    @staticmethod
    def get_delta(a, y, z):
        """
        Method to return delta for the output layer.

        :param a: predicted outcome
        :param y: actual outcome
        :return: delta (error) for output layer
        """
        return (a - y) * sigmoid_derivative(z)

