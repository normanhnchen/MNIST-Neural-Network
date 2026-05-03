"""Initializes the neural network and trains it."""


import pickle

from classes import Network, Image


# Load saved data
with open("training_data.pkl", "rb") as f:
    training_data = pickle.load(f)

# Learning rate
eta = 3
epochs = 30
batch_size = 100

network = Network([784, 30, 10])

network.train(training_data, epochs, batch_size, eta)

with open("trained_network.pkl", "wb") as f:
    pickle.dump(network, f)