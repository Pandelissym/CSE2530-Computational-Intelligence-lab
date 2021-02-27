"""
Module implementing a perceptron.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.activations import step_function


class Perceptron:
    """
    Class implementing a perceptron.
    """

    def __init__(self, data, labels, learning_rate=0.1,
                 convergence_threshold=0.01, max_epochs=10000):
        """
        Initialize perceptron. Each training vector is appended by a 1 and the
        bias is included in the weight vector in order to simplify calculations

        :param data: List of training data
        :param labels: List of corresponding labels
        :param learning_rate: learning rate for learning algorithm
        :param convergence_threshold: threshold to stop learning
        """
        assert len(data) > 0 and len(data[0]) > 0
        assert len(data) == len(labels)

        self.data = np.array([np.append(i, 1) for i in np.array(data)])
        self.labels = np.array(labels)
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.bias = np.random.random()
        self.weights = np.append(np.random.random(len(data[0])), self.bias)

    def train(self):
        """
        Method to train perceptron by learning weights and bias using the
        perceptron algorithm

        :return: None
        """
        converged = False
        epoch = 0

        while not converged and epoch < self.max_epochs:
            epoch += 1
            old_weights = self.weights.copy()

            for k, point in enumerate(self.data):
                output = self.get_prediction(point)
                self._update_weights(point, output, self.labels[k])

            print(f"Epoch {epoch} completed.")
            converged = self._has_converged(old_weights)

        if converged:
            print(f"Converged in {epoch} epochs.")
        elif epoch >= self.max_epochs:
            print(f"Could not converge in {self.max_epochs} epochs.")

    def get_prediction(self, point):
        """
        Method to get prediction for a single data point by doing the dot
        product between the point and the weight vector and then passing it
        through the activation function chosen.

        :param point: input vector
        :return: output of the perceptron
        """
        assert len(point) == len(self.data[0]) \
               or len(point) == len(self.data[0]) - 1

        if len(point) == len(self.data[0]) - 1:
            point = np.append(point, [1])

        output = step_function(np.dot(self.weights, point))
        return output

    def _update_weights(self, point, predicted_output, actual_output):
        """
        Method that updates the weights based on the actual label of the input
        point and the one the model predicted.

        :param point: input vector
        :param predicted_output: output of the perceptron for this input
        :param actual_output: actual label of the input point
        :return: None
        """
        for i in range(len(self.weights)):
            error = actual_output - predicted_output
            delta_w = self.learning_rate * point[i] * error
            self.weights[i] += delta_w

    def _has_converged(self, old_weights):
        """
        Method to check if perceptron has converged by seeing if the weights
        have not changed
        :param old_weights:
        :return:
        """
        outputs = [self.get_prediction(x) for x in self.data]
        n = len(self.data)

        error = 1 / n * np.sum(np.abs(self.labels - outputs))
        return error < self.convergence_threshold

    def plot_decision_boundary(self):
        """
        Method to plot the decision boundary along with the points of the
        trained perceptron.

        :return: None
        """
        assert len(self.data[0]) - 1 == 2

        x_values = self.data[:, 0]
        y_values = self.data[:, 1]
        plt.scatter(x_values, y_values, c=self.labels)

        max_x = np.max(np.array(self.data)[:, 0]) + 1
        min_x = np.min(np.array(self.data)[:, 0]) - 1
        plt.xlim(min_x, max_x)

        x = np.linspace(min_x, max_x)
        y = [(-self.weights[2] - self.weights[0] * i) / self.weights[1]
             for i in x]

        min_y = np.min(self.data[:, 1]) - 1
        max_y = np.max(self.data[:, 1]) + 1
        plt.ylim(min_y, max_y)

        plt.plot(x, y)
        plt.fill_between(x, y, min_y, alpha=0.2)
        plt.fill_between(x, y, max_y, alpha=0.2)

        plt.show()
