"""Saves the data from MNIST csv files."""


import pandas as pd
import numpy as np

from src.scratch.network import Image


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
    training_data = load_csv("data/MNIST/csv/mnist_train.csv")
    testing_data = load_csv("data/MNIST/csv/mnist_test.csv")

    np.save("data/scratch/training_data.npy", training_data)
    np.save("data/scratch/testing_data.npy", testing_data)

    print("Training and testing data saved.")
