"""
Module to load the data.
"""
import numpy as np


def load_data():
    """
    Loads the input data.

    :return: (features, targets, unknown)
    """
    with open("./data/features.txt") as file:
        features = [np.array(list(map(float, line.strip('/n').split(',')))).reshape((10, 1))
                    for line in file]

    with open("./data/targets.txt") as file:
        targets = [vectorize_target(int(line.rstrip('\n')), 7) for line in file]

    with open("./data/unknown.txt") as file:
        unknown = [np.array(list(map(float, line.strip('/n').split(',')))).reshape((10,1))
                   for line in file]
    return np.array(features), np.array(targets), np.array(unknown)


def vectorize_target(y, num_classes):
    """
    Local function to vectorize a specific target
    :param y: target to vectorize
    :return: vectorized target
    """
    v_y = np.zeros((num_classes, 1))
    v_y[y - 1] = 1.0
    return v_y


def split_train_test(data, test_size, random=False):
    """
    Method that splits the data into train and test sets.

    :param data: data to split
    :param test_size: number between 0 and 1 indicating the proportion of the data to
    use for testing
    :param random: whether to shuffle the data before splitting.
    :return: (train_data, validate_data)
    """
    assert 0 <= test_size <= 1
    if random:
        np.random.shuffle(data)

    index = int(len(data) * test_size)
    return data[index:], data[:index]


def true_targets():
    return np.genfromtxt('../data/targets.txt', delimiter=',', dtype=int)
