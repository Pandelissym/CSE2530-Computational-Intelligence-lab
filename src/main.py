"""
Main module.
"""
from src.data_loader import load_data, split_train_test
from src.network import Network
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

    accuracy = network.evaluate(test_data)
    print(f"accuracy= {accuracy}")

    plt.suptitle("Sigmoid with softmax and log likelihood. Accuracy = "
                 + str(accuracy))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
