"""Loads saved trained neural network and tests it."""


import pickle

from src.scratch.network import Network, Image


# Load saved data
with open("src/scratch/data/trained_network.pkl", "rb") as f:
    network = pickle.load(f)
with open("src/scratch/data/testing_data.pkl", "rb") as f:
    testing_data = pickle.load(f)


test_subset = testing_data[:10001]

res = network.evaluate(test_subset)

print(f"Correct predictions: {res} out of {len(test_subset)}")
print(f"Accuracy: {(res/len(test_subset))*100:.2f}%")
