from src.network import Network
import numpy as np
import matplotlib.pyplot as plt


def accuracy_score(predictions, targets):
    return np.where(predictions == targets)[0].shape[0] / targets.shape[0]


def cross_validate(network: Network, X, y, k, validation_data, epochs=30):
    scores = np.empty(k)
    for i in range(k):
        curr_network = Network(network.layers)
        begin_i = (i * len(X)) // k
        end_i = ((i + 1) * len(X)) // k - 1

        X_train = np.concatenate((X[:begin_i], X[end_i:]))
        y_train = np.concatenate((y[:begin_i], y[end_i:]))

        if i == 0:
            curr_network.train(X_train, epochs, 20, 0.07, validation_data)
        else:
            curr_network.train(X_train, epochs, 20, 0.07)

        predictions = [np.argmax(curr_network.feedforward(x)) + 1 for x, y in X[begin_i:end_i]]
        accuracy = accuracy_score(predictions, y[begin_i:end_i])
        scores[i] = accuracy

    print('\nstd: ', np.std(scores))
    print('scores: ', scores)

    return np.mean(scores)


def plot_results(neuron_layouts):
    x = np.arange(len(neuron_layouts['amounts']))  # the label locations
    width = 0.7

    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=[7,5])

    rects = ax.bar(x, neuron_layouts['scores'], width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('K-Fold Cross Validation Scores')
    ax.set_title('Number of Neurons')
    ax.set_xticks(x)
    ax.set_xticklabels(neuron_layouts['amounts'])
    ax.legend()

    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects, ax)

    fig.tight_layout()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    plt.ylim((0.5, 1.1))

    plt.show()
