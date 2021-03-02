"""
Module implementing a multi layer perceptron
"""
import numpy as np
import matplotlib.pyplot as plt
from src.activations import sigmoid, sigmoid_derivative, softmax, \
    tanh_derivative, tanh, leaky_relu_derivative, leaky_relu
from src.cost_functions import LogLikelihood


class Network:
    """
    Class implementing a neural network
    """

    def __init__(self, sizes):
        """
        Initializes neural network. Sizes is a list representing the amount
        of neurons in each layer.

        e.g. sizes = [2, 3 ,4] means the neural network consists 3 layers with
        2 input neurons in the input layer, 3 neurons in the single
        hidden layer and 4 neurons in the output layer.

        :param sizes: A list containing the amount of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.biases = np.array([np.random.randn(x, 1) for x in sizes[1:]],
                               dtype=object)
        self.weights = np.array([np.random.randn(x, y) for (x, y)
                                 in zip(sizes[1:], sizes[:-1])], dtype=object)
        self.cost_function = LogLikelihood

    def feedforward(self, x):
        """
        Method that feeds the input to the neural network passing it
        through all the layers and returns the result output by the output
        layer.

        :param x: The input to the n.n.
        :return: The output of the n.n.
        """
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = sigmoid(np.dot(w, x) + b)
        w, b = self.weights[-1], self.biases[-1]
        x = softmax(np.dot(w, x) + b)
        return x

    def train(self, data, epochs, mini_batch_size, learning_rate,
              validation_data=None):
        """
        Method to train the neural network by learning the weights through
        stochastic gradient descent and backpropagation.

        :param data: The training data as a list of tuples (vector, label)
        :param epochs: The number of epochs to train for
        :param mini_batch_size: The size of the mini batch used for gradient
        descent
        :param learning_rate: The learning rate for gradient descent
        :param validation_data: If supplied will calculate the cost on the
        train and validation data and plot a graph.
        :return: None
        """
        n = len(data)

        if validation_data is not None:
            train_errors = []
            validation_errors = []

        for i in range(epochs):
            if validation_data is not None:
                train_error = self.cost_function.get_cost(data, self)
                train_errors.append(train_error)
                validation_error = self.cost_function.get_cost(validation_data,
                                                               self)
                validation_errors.append(validation_error)

            np.random.shuffle(data)
            mini_batches = [data[j:j + mini_batch_size]
                            for j in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_with_mini_batch(mini_batch, learning_rate)

            print(f"Epoch {i} completed.")

        if validation_data is not None:
            x = list(range(epochs))
            plt.plot(x, train_errors, label=f'training')
            plt.plot(x, validation_errors, label=f'validation')

    def update_with_mini_batch(self, mini_batch, learning_rate):
        """
        Method that takes a mini batch and updates the weights and bias of the
        model based on the data in the mini batch.

        :param mini_batch: random subset of the whole training set
        :param learning_rate: gradient descent learning rate
        :return: None
        """
        bias_gradients = np.array([np.zeros(bias.shape)
                                   for bias in self.biases],  dtype=object)
        weight_gradients = np.array([np.zeros(weight.shape)
                                     for weight in self.weights],
                                    dtype=object)

        for x, y in mini_batch:
            point_bias_gradient, point_weight_gradient =\
                self.backpropagation(x, y)
            bias_gradients += point_bias_gradient
            weight_gradients += point_weight_gradient

        self.biases -= (learning_rate / len(mini_batch)) * bias_gradients
        self.weights -= (learning_rate / len(mini_batch)) * weight_gradients

    def backpropagation(self, x, y):
        """
        Method that applies backpropagation using x as input and returns the
        gradient of the cost function.

        :param x: input point
        :param y: label of x
        :return: gradient of cost function
        """
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        alphas = [x]
        zetas = []
        a = alphas[0]

        # Feed point forward and store all activations and outputs
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            zetas.append(z)
            a = sigmoid(z)
            alphas.append(a)

        w, b = self.weights[-1], self.biases[-1]
        z = np.dot(w, a) + b
        zetas.append(z)
        a = softmax(z)
        alphas.append(a)

        # Calculate cost (delta) for output layer and use it to calculate
        # gradients of the output layer
        delta = self.cost_function.get_delta(alphas[-1], y, zetas[-1])

        bias_gradients[-1] = delta

        weight_gradients[-1] = np.dot(delta, alphas[-2].T)

        # Move back through the network calculating gradients by updating
        # delta
        for i in reversed(range(self.num_layers - 2)):

            weights = self.weights[i + 1]
            delta = np.dot(weights.T, delta) * sigmoid_derivative(zetas[i])

            bias_gradients[i] = delta
            weight_gradients[i] = np.dot(delta, alphas[i].T)

        return np.array(bias_gradients,  dtype=object), \
            np.array(weight_gradients,  dtype=object)

    def evaluate(self, validation_data):
        """
        Validates the network based on the validation data and return
        prediction accuracy.

        :param validation_data: data to validate model on
        :return: The prediction accuracy of the model as amount of correctly
        predicted data points / total data points.
        """
        total = len(validation_data)
        correct = 0

        for x, y in validation_data:
            output = self.feedforward(x)
            if np.argmax(output) == np.argmax(y):
                correct += 1

        return correct / total
