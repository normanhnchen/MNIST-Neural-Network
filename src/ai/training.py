"""Saves the data from MNIST csv files."""


import pandas as pd
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def images_to_tensors(images):
    pixel_rows = []
    targets = []

    for img in images:
        # Flatten 784 pixels into one row (784,)
        pixels = torch.from_numpy(np.asarray(img.values, dtype=np.float32))
        pixel_rows.append(pixels.view(784))

        # Get the digit label
        target = int(np.argmax(img.label))
        targets.append(target)

    # Stack rows into one matrix and move to DEVICE
    inputs = torch.stack(pixel_rows).to(DEVICE)  # (N, 784)
    targets = torch.tensor(targets, dtype=torch.long, device=DEVICE)  # (N,)
    return inputs, targets


if __name__ == "__main__":
    # Load saved data
    training_data = load_csv("data/MNIST/csv/mnist_train.csv")
    testing_data = load_csv("data/MNIST/csv/mnist_test.csv")

    training_inputs, training_targets = images_to_tensors(training_data)
    testing_inputs, testing_targets = images_to_tensors(testing_data)

    torch.save(training_inputs, "data/ai/training_inputs.pt")
    torch.save(training_targets, "data/ai/training_targets.pt")
    torch.save(testing_inputs, "data/ai/testing_inputs.pt")
    torch.save(testing_targets, "data/ai/testing_targets.pt")

    print("Training and testing data saved.")
