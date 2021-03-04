"""
Main module.
"""
from src.data_loader import load_data, split_train_test
from src.network import Network
import numpy as np
from src.perceptron import Perceptron
import matplotlib.pyplot as plt


def main():
    """
    Main method to run.
    """
    # data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # labels = [0, 0, 0, 1]
    # perceptron = Perceptron(data, labels)
    # perceptron.train()
    # perceptron.plot_decision_boundary()

    features, targets, unknown = load_data()
    data = list(zip(features, targets))
    training_data, test_data = split_train_test(data, 0.15)
    training_data, validation_data = split_train_test(training_data, 0.15)

    network = Network([10, 30, 7])
    network.train(training_data, 30, 20, 0.07, validation_data)

    # Plots
    accuracy = network.evaluate(test_data)
    confusion_matrix = network.confusion_matrix(test_data)
    print(f"accuracy= {accuracy}")
    plt.suptitle("Sigmoid with softmax and log likelihood. Accuracy = "
                 + str(accuracy))
    plt.suptitle("Confusion Matrix")
    # print(f"Confusion Matrix= {confusion_matrix}")
    plt.legend()
    plt.show()

    # Write predictions of unknown data to Text File
    f = open("../doc/Group_66_classes.txt", "w")
    for x in unknown:
        print(x)
        prediction = np.argmax(network.feedforward(x)) + 1
        f.write(f"{prediction},")
    f.close()

if __name__ == "__main__":
    main()
