"""Saves the data from MNIST csv files."""


import pandas as pd
import pickle
import numpy as np


class Image:
    def __init__(self, label, values):
        self.label = label
        self.values = values


def load_csv(path):
    """Saves the data from a csv file."""

    data = pd.read_csv(path)

    images = []
    labels = data.values[:, 0]
    values = data.values[:, 1:] / 255

    for i in range(len(labels)):
        label = np.zeros((10, 1))
        label[labels[i]] = 1.0
        value = values[i]
        value = value.reshape(784, 1)
        images.append(Image(label, value))

    return images


if __name__ == "__main__":
    training_data = load_csv("MNIST/csv/mnist_train.csv")
    testing_data = load_csv("MNIST/csv/mnist_test.csv")

    # Save training data
    with open("src/training_data.pkl", "wb") as f:
        pickle.dump(training_data, f)
    # Save testing data
    with open("src/testing_data.pkl", "wb") as f:
        pickle.dump(testing_data, f)
    