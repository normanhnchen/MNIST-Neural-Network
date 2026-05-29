import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ai.training import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()

        self.num_inputs = sizes[0]
        self.num_hidden = sizes[1]
        self.num_outputs = sizes[2]

        # Fully connected layers
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)

    def forward(self, inputs):
        """Forward pass through the network."""
        inputs = inputs.to(DEVICE)
        # Apply activation function
        inputs = torch.sigmoid(self.fc1(inputs))
        inputs = self.fc2(inputs)
        return inputs

    def SGD(self, training_data, epochs, batch_size, eta):
        """Stochastic gradient descent."""

        self.to(DEVICE)
        self.train(mode=True)

        optimizer = optim.SGD(self.parameters(), lr=eta/batch_size)
        criterion = nn.CrossEntropyLoss(reduction="sum")

        n = len(training_data)
        for epoch in range(epochs):
            # Shuffle the training data to prevent bias in the order of the data
            random.shuffle(training_data)
            # Split the training data into mini-batches
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]

            for batch in batches:
                # Reset the gradients
                optimizer.zero_grad()

                # Convert the batch of images to tensors
                inputs, targets = images_to_tensors(batch)
                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)

                # Backward propagation: compute gradients
                loss.backward()
                # Update the weights and biases
                optimizer.step()

            print(f"Epoch {epoch} complete")

    def evaluate(self, testing_data, batch_size):
        """Evaluate the network by testing it and getting how many times it gets the correct decision."""

        self.to(DEVICE)
        # Set the network to evaluation mode
        self.train(mode=False)

        correct = 0
        n = len(testing_data)
        batches = [testing_data[j:j+batch_size] for j in range(0, n, batch_size)]

        # Disable gradient tracking as evaluation doesn't require it
        # Speeds up processing and memory
        with torch.no_grad():
            for batch in batches:
                inputs, targets = images_to_tensors(batch)
                outputs = self(inputs)
                # Get the maximum number in the output (the neural network's decision)
                guesses = outputs.argmax(dim=1)

                # Count how many times the network gets the correct decision
                correct += (guesses == targets).sum().item()

        return correct


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load saved data
    training_data = torch.load("data/ai/training_data.pt", weights_only=False)

    eta = 3 # Learning rate
    epochs = 30
    batch_size = 100

    network = Network([784, 16, 10])
    network.SGD(training_data, epochs, batch_size, eta)

    torch.save(network, "data/ai/trained_network.pt")

    print("Network trained and saved.")
    torch.save(network, "data/ai/trained_network.pt")

    print("Network trained and saved.")
