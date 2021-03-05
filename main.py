"""
Main module.
"""
from data_loader import *
from matplotlib.ticker import MaxNLocator
from network import Network
from validation import *
import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

# INFO: data folder must be in the parent directory of the data_loader.py file.

def main():
    """
    Main method to run.
    """
    # data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # labels = [0, 1, 1, 0]
    # perceptron = Perceptron(data, labels)
    # perceptron.train(plot_learning_graph=True, save_file=False)
    # perceptron.plot_decision_boundary(save_file=False)

    features, targets, unknown = load_data()
    data = list(zip(features, targets))
    training_data, test_data = split_train_test(data, 0.15)
    training_data, validation_data = split_train_test(training_data, 0.15)

    # Create the network with one hidden layer
    network = Network([10, 30, 7])
    # set to the data to which we are learning to 'data' because we are testing it on the unknown dataset
    network.train(data, 35, 20, 0.07, validation_data)

    # # Plots
    # # Plot the confusion matrix of the network
    # accuracy = network.evaluate(test_data)
    # confusion_matrix = network.confusion_matrix(test_data)
    # print(f"accuracy= {accuracy}")
    # plt.suptitle("Sigmoid with softmax and log likelihood. Accuracy = "
    #              + str(accuracy))
    # plt.suptitle("Confusion Matrix")
    # # print(f"Confusion Matrix= {confusion_matrix}")
    # plt.legend()
    # plt.show()

    # # Plot the performance of different middle layers in cross validation
    # neuron_layouts = {
    #     'amounts': [5, 8, 10, 20, 30, 45],
    #     'scores': []
    # }
    # print('\nk-fold validating...')
    # for amount in neuron_layouts['amounts']:
    #     targets_true = split_train_test(true_targets(), 1 - (1 - 0.15) ** 2)[0]
    #     k_fold_acc = cross_validate(Network([10, amount, 7]), np.array(training_data, dtype=object), targets_true,
    #                                 validation_data=np.array(validation_data) if amount == 30 else None, epochs=24, k=10)
    #     print('\nk_fold_acc: ', k_fold_acc)
    #     neuron_layouts['scores'].append(round(k_fold_acc, 2))
    # plot_results(neuron_layouts)

    # # Write predictions of unknown data to Text File
    # f = open("../doc/Group_66_classes.txt", "w")
    # for x in unknown[:-1]:
    #     prediction = np.argmax(network.feedforward(x)) + 1
    #     f.write(f"{prediction},")
    #
    # prediction = np.argmax(network.feedforward(unknown[-1])) + 1
    # f.write(f"{prediction}")
    # f.close()




if __name__ == "__main__":
    main()
