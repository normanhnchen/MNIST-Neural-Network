"""Loads saved trained neural network and tests it."""


import torch

from src.ai.network import Network, Image


# Load trained network and testing data
network = torch.load("data/ai/trained_network.pt", weights_only=False)
testing_data = torch.load("data/ai/testing_data.pt", weights_only=False)


test_subset = testing_data[:10001]
batch_size = 100

res = network.evaluate(test_subset, batch_size=batch_size)

print(f"Correct predictions: {res} out of {len(test_subset)}")
print(f"Accuracy: {(res/len(test_subset))*100:.2f}%")
