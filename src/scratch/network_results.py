"""Loads saved trained neural network and tests it."""


import numpy as np

from src.scratch.network import Network, Image


# Load saved data
# .item() to convert from 0-d array to object
network = np.load("data/scratch/trained_network.npy", allow_pickle=True).item()
testing_data = np.load("data/scratch/testing_data.npy", allow_pickle=True)

test_subset = testing_data[:10001]

res = network.evaluate(test_subset)

print(f"Correct predictions: {res} out of {len(test_subset)}")
print(f"Accuracy: {(res/len(test_subset))*100:.2f}%")
