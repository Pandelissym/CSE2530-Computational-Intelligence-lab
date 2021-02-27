"""
Main module.
"""
import numpy as np
from src.perceptron import Perceptron


def main():
    """
    Main method to run.
    """
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 0]

    perceptron = Perceptron(data, labels)
    perceptron.train()
    perceptron.plot_decision_boundary()


if __name__ == "__main__":
    main()
