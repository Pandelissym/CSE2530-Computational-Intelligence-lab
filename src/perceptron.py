"""
Module implementing a perceptron.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.activations import step
import matplotlib.axes._axes as axes


class Perceptron:
    """
    Class implementing a perceptron.
    """

    def __init__(self, data, labels, learning_rate=0.05,
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

    def train(self, plot_learning_graph=False):
        """
        Method to train perceptron by learning weights and bias using the
        perceptron algorithm

        :param plot_learning_graph: Whether to plot error over epochs
        :return: None
        """
        converged = False
        epoch = 0
        costs = []

        while not converged and epoch < self.max_epochs:

            for k, point in enumerate(self.data):
                output = self.get_prediction(point)
                self._update_weights(point, output, self.labels[k])

            print(f"Epoch {epoch} completed.")
            cost = self._get_cost()
            if plot_learning_graph:
                costs.append(cost)
            converged = self._has_converged(cost)
            epoch += 1

        if converged:
            print(f"Converged in {epoch} epochs.")
        elif epoch >= self.max_epochs:
            print(f"Could not converge in {self.max_epochs} epochs.")

        if plot_learning_graph:
            self._plot_learning_graph(costs, epoch)

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

        output = step(np.dot(self.weights, point))
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

    def _has_converged(self, cost):
        """
        Method to check if perceptron has converged by seeing if the error
        is below a threshold.
        :param cost: The value of the cost
        :return:
        """
        return cost < self.convergence_threshold

    def _get_cost(self):
        """
        Calculate the MSE on the training set

        :return: MSE
        """
        outputs = [self.get_prediction(x) for x in self.data]
        n = len(self.data)
        error = (1 / n) * np.sum(np.square(self.labels - outputs))

        return error

    def plot_decision_boundary(self, save_file=False):
        """
        Method to plot the decision boundary along with the points of the
        trained perceptron.

        :return: None
        """
        assert len(self.data[0]) - 1 == 2

        ax = plt.subplot()  # type: axes.Axes

        colors = ['tab:blue', 'tab:orange']

        x_values = self.data[:, 0]
        y_values = self.data[:, 1]

        # Limit axes to keep only area of interest
        max_x = np.max(np.array(self.data)[:, 0]) + 1
        min_x = np.min(np.array(self.data)[:, 0]) - 1
        ax.set_xlim(min_x, max_x)

        min_y = np.min(self.data[:, 1]) - 1
        max_y = np.max(self.data[:, 1]) + 1
        ax.set_ylim(min_y, max_y)

        # Draw decision boundaries
        xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.01),
                             np.arange(min_y, max_y, 0.01))
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1, r2))
        pred = np.array([self.get_prediction(i) for i in grid])
        zz = pred.reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.2, levels=1, colors=colors)

        # Draw line
        x = np.linspace(min_x, max_x)
        y = [(-self.weights[2] - self.weights[0] * i) / self.weights[1]
             for i in x]
        ax.plot(x, y, color='k', lw='1.2')

        # Draw vector
        index = np.argmin(np.abs(np.array(y) - max_y))
        x1, y1 = x[index], y[index]
        index2 = np.argmin(np.abs(np.array(y) - min_y))
        x2, y2 = x[index2], y[index2]

        ax.quiver((x1 + x2)/2, (y1 + y2)/2, self.weights[0], self.weights[1],
                  scale_units='xy', scale=1)

        # Draw points
        ax.scatter(x_values, y_values, c=[colors[i] for i in self.labels])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title("Decision boundary for OR", y=1.05)
        if save_file:
            plt.savefig("../doc/plots/OR_decision_boundary")
        plt.show()

    def _plot_learning_graph(self, costs: [float], epochs: int, save_file=False):
        """
        Plot the cost versus epochs.

        :param costs: List of cost at each epoch
        :param epochs: The total amount of epochs the model run for
        :return: None
        """
        x = list(range(epochs))
        ax = plt.subplot()  # type:axes.Axes
        ax.set_ylabel("Error (MSE)")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x, costs)
        ax.set_title("Error (MSE) over epochs for OR", y=1.05)
        if save_file:
            plt.savefig("../doc/plots/OR_error_over_epochs")
        plt.show()
