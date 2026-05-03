"""Loads saved trained neural network and tests it."""


import pickle

from src.network import Network, Image


# Load saved data
with open("src/trained_network.pkl", "rb") as f:
    network = pickle.load(f)
with open("src/testing_data.pkl", "rb") as f:
    testing_data = pickle.load(f)


test_subset = testing_data[:10001]

# Pass the objects directly to the fixed evaluate method
res = network.evaluate(test_subset)

print(f"Correct predictions: {res} out of {len(test_subset)}")
print(f"Accuracy: {(res/len(test_subset))*100:.2f}%")