"""
Module implementing a multi layer perceptron
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from activations import sigmoid, sigmoid_derivative, softmax
from cost_functions import LogLikelihood
from matplotlib.ticker import MaxNLocator


def accuracy_score(predictions, targets):
    return np.where(predictions == targets)[0].shape[0] / len(targets)

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
        self.layers = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

        self.weights = [np.random.randn(x, y) for (x, y)
                        in zip(sizes[1:], sizes[:-1])]

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
            train_accs = []
            validation_accs = []

        for i in range(epochs):
            if validation_data is not None:
                train_error = self.cost_function.get_cost(data, self)
                train_acc = accuracy_score(np.array([np.argmax(self.feedforward(x)) + 1 for x, y in data]), np.array([np.argmax(y) + 1 for x, y in data]))
                train_errors.append(train_error)
                train_accs.append(train_acc)

                validation_error = self.cost_function.get_cost(validation_data,
                                                               self)
                validation_acc = accuracy_score(np.array([np.argmax(self.feedforward(x)) + 1 for x, y in validation_data]), np.array([np.argmax(y) + 1 for x, y in validation_data]))
                validation_errors.append(validation_error)
                validation_accs.append(validation_acc)

            np.random.shuffle(data)
            mini_batches = [data[j:j + mini_batch_size]
                            for j in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_with_mini_batch(mini_batch, learning_rate)

            print(f"Epoch {i} completed.")

        if validation_data is not None:
            font = {'family': 'normal',
                    'weight': 'normal',
                    'size': 17}
            plt.rc('font', **font)
            plt.rc('lines', markersize=6)
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            plt.rc("figure", figsize=(6, 6))

            x = list(range(epochs))
            plt.plot(x, train_errors, label=f'training')
            plt.plot(x, validation_errors, label=f'validation')
            ax = plt.gca()
            ax.set_ylabel("Error (MSE)", labelpad=8)
            ax.set_xlabel("Epoch", labelpad=5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title("Error (MSE) over epochs", y=1.05)
            plt.legend()
            plt.show()

            plt.tight_layout()
            # plt.savefig("../doc/plots/validation_vs_training")

            plt.plot(x, train_accs, label=f'training')
            plt.plot(x, validation_accs, label=f'validation')
            ax = plt.gca()
            ax.set_ylabel("Accuracy", labelpad=8)
            ax.set_xlabel("Epochs", labelpad=5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title("Accuracy over epochs", y=1.05)
            plt.legend()
            plt.show()

            plt.tight_layout()

    def update_with_mini_batch(self, mini_batch, learning_rate):
        """
        Method that takes a mini batch and updates the weights and bias of the
        model based on the data in the mini batch.

        :param mini_batch: random subset of the whole training set
        :param learning_rate: gradient descent learning rate
        :return: None
        """
        bias_gradients = [np.zeros(bias.shape)
                          for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape)
                            for weight in self.weights]

        for x, y in mini_batch:
            point_bias_gradient, point_weight_gradient =\
                self.backpropagation(x, y)
            bias_gradients = [bg + pbg for bg, pbg in zip(bias_gradients, point_bias_gradient)]
            weight_gradients = [wg + pwg for wg, pwg in zip(weight_gradients, point_weight_gradient)]

        self.biases = [b - (learning_rate / len(mini_batch)) * bg for b, bg in zip(self.biases, bias_gradients)]
        self.weights = [w - (learning_rate / len(mini_batch)) * wg for w, wg in zip(self.weights, weight_gradients)]

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

        return bias_gradients, weight_gradients

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

    def confusion_matrix(self, validation_data):
        """
        Creates Confusion Matrix based on the validation data and returns
        it.

        :param validation_data: data to validate model on
        :return: The confusion matrix of the model.
        """
        confusion_matrix = np.zeros((7, 7))
        for x, y in validation_data:
            output = self.feedforward(x)
            confusion_matrix[np.argmax(output), np.argmax(y)] += 1

        df_cm = pd.DataFrame(confusion_matrix, index=["Class " + i for i in "1234567"],
                             columns=["Class " + i for i in "1234567"])

        group_c = ["{0: 0.0f}".format(value) for value in confusion_matrix.flatten()]

        normalized = confusion_matrix
        for i in range(len(confusion_matrix.T)):
            normalized[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])


        group_p = ["{0: .2%}".format(value) for value in normalized.flatten()]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_c, group_p)]

        labels = np.asarray(labels).reshape(7, 7)
        plt.figure(figsize=(12, 8))
        sn.heatmap(df_cm, annot=labels, cmap='Blues', fmt='').set(xlabel='Predicted Values', ylabel='Actual Values')

        return confusion_matrix
