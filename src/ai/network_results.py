"""Loads saved trained neural network and tests it."""


import torch
import torch

from src.ai.network import Network, Image


# Load saved data
network = torch.load("data/ai/trained_network.pt", weights_only=False)
testing_inputs = torch.load("data/ai/testing_inputs.pt", weights_only=False)
testing_targets = torch.load("data/ai/testing_targets.pt", weights_only=False)


test_subset = testing_inputs[:10001]
batch_size = 100

res = network.evaluate(test_subset, testing_targets)

print(f"Correct predictions: {res} out of {len(test_subset)}")
print(f"Accuracy: {(res/len(test_subset))*100:.2f}%")
