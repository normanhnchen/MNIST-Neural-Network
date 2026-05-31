import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.ai.training import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.num_layers = len(sizes) - 1

        # Add fully connected layers connecting each layer in the network
        for i in range(self.num_layers):
            # Add a fully connected layer between the current layer and the next layer
            self.add_module(f"fc{i+1}", nn.Linear(sizes[i], sizes[i+1]))

    def forward(self, inputs):
        """Forward pass through the network."""

        inputs = inputs.to(DEVICE)
        
        # Number of layers in the network (excluding the input layer)
        for i in range(1, self.num_layers + 1):
            # Apply the fully connected layer
            inputs = getattr(self, f"fc{i}")(inputs)
            # Apply the activation function on hidden layers only 
            if i < self.num_layers:
                inputs = torch.sigmoid(inputs)
        
        return inputs

    def SGD(self, training_inputs, training_targets, epochs, batch_size, eta):
        """Stochastic gradient descent."""

        self.to(DEVICE)
        self.train(mode=True)

        optimizer = optim.SGD(self.parameters(), lr=eta/batch_size)
        criterion = nn.CrossEntropyLoss(reduction="sum")

        n = len(training_inputs)

        for epoch in range(epochs):
            # Shuffle the indices of the training data to prevent bias in the order of the data
            perm = torch.randperm(n)

            # Shuffle the training data
            training_inputs = training_inputs[perm]
            training_targets = training_targets[perm]

            # Split the training data into mini-batches
            batches = [training_inputs[j:j+batch_size] for j in range(0, n, batch_size)]
            batches_targets = [training_targets[j:j+batch_size] for j in range(0, n, batch_size)]

            for batch, batch_targets in zip(batches, batches_targets):
                # Reset the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(batch)
                loss = criterion(outputs, batch_targets)

                # Backward propagation: compute gradients
                loss.backward()
                # Update the weights and biases
                optimizer.step()

            print(f"Epoch {epoch} complete")

    def evaluate(self, testing_inputs, testing_targets):
        """Evaluate the network by testing it and getting how many times it gets the correct decision."""

        self.to(DEVICE)
        # Set the network to evaluation mode
        self.train(mode=False)

        correct = 0

        # Disable gradient tracking as evaluation doesn't require it
        # Speeds up processing and memory
        with torch.no_grad():
            # Get the network's decisions
            outputs = self.forward(testing_inputs)
            # Get the maximum number in the output (the neural network's decision)
            guesses = outputs.argmax(dim=1)

            # Count how many times the network gets the correct decision
            correct += (guesses == testing_targets).sum().item()

        return correct


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load saved data
    training_inputs = torch.load("data/ai/training_inputs.pt", weights_only=False)
    training_targets = torch.load("data/ai/training_targets.pt", weights_only=False)
    testing_inputs = torch.load("data/ai/testing_inputs.pt", weights_only=False)
    testing_targets = torch.load("data/ai/testing_targets.pt", weights_only=False)

    eta = 3 # Learning rate
    epochs = 50
    batch_size = 100

    network = Network([784, 256, 256, 10])
    # Train the network

    time_start = time.perf_counter()
    network.SGD(training_inputs, training_targets, epochs, batch_size, eta)
    time_end = time.perf_counter()
    print(f"Time taken to train the network: {time_end - time_start} seconds")

    torch.save(network, "data/ai/trained_network.pt")

    print("Network trained and saved.")
