"""Saves the data from MNIST csv files."""


import pandas as pd
import numpy as np
import torch


class Image:
    def __init__(self, label, values):
        self.label = torch.tensor(label, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)


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

    torch.save(training_data, "data/ai/training_data.pt")
    torch.save(testing_data, "data/ai/testing_data.pt")

    print("Training and testing data saved.")