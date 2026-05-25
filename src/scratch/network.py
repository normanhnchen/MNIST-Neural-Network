"""Initializes the neural network and trains it."""


import pickle
import numpy as np
import random


class Image:
    def __init__(self, label, values):
        self.label = label
        self.values = values


class Network:
    def __init__(self, sizes):
        # Sizes of each layer of the network
        self.sizes = np.array(sizes)
        self.num_layers = len(sizes)
        # Activation
        self.a = None
        # Randomize the biases in each layer
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        # Randomize the weights in each layer
        self.w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feed_forward(self, a):
        """Return the output of the network."""

        for w, b in zip(self.w, self.b):
            # Calculate the activation (output of a neuron)
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def backprop(self, input, target):
        """Backpropagate through the entire network."""

        # List to store all gradients of the weights
        nabla_w = [np.zeros(w.shape) for w in self.w]
        # List to store all gradients of the biases
        nabla_b = [np.zeros(b.shape) for b in self.b]

        a = input
        a_list = [input]
        z_list = []
        # Save all activations and z vectors
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            z_list.append(z)
            a_list.append(a)
        
        # Backpropogate at the last layer
        dC_da = self.MSE_deriv(a_list[-1], target)
        da_dz = self.sigmoid_deriv(z_list[-1])
        dC_dz = dC_da * da_dz
        nabla_w[-1] = np.dot(dC_dz, a_list[-2].transpose())
        nabla_b[-1] = dC_dz
        
        # Backpropagate through the rest of the layers
        for l in range(2, self.num_layers):
            z = z_list[-l]
            da_dz = self.sigmoid_deriv(z)
            dC_da = np.dot(self.w[-l+1].transpose(), dC_dz)
            dC_dz = dC_da * da_dz
            dz_db = 1

            dC_db = dC_dz * dz_db
            dC_dw = np.dot(dC_dz, a_list[-l-1].transpose())

            nabla_b[-l] = dC_db
            nabla_w[-l] = dC_dw
        
        return nabla_w, nabla_b
    
    def train(self, training_data, epochs, batch_size, eta):
        """
        Train the network using stochastic gradient descent and
        updating the weights and biases during the process."""

        n = len(training_data)
        for epoch in range(epochs):
            # Shuffle the training data to prevent bias in the order of the data
            random.shuffle(training_data)
            # Split all the training data into mini-batches
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]
            
            for batch in batches:
                sum_nabla_w = [np.zeros(w.shape) for w in self.w]
                sum_nabla_b = [np.zeros(b.shape) for b in self.b]
                
                # Add all gradients from a batch
                for img in batch:
                    curr_nabla_w, curr_nabla_b = self.backprop(img.values, img.label)
                    
                    sum_nabla_w = [sw + nw for sw, nw in zip(sum_nabla_w, curr_nabla_w)]
                    sum_nabla_b = [sb + nb for sb, nb in zip(sum_nabla_b, curr_nabla_b)]
                
                # Update the weights and biases so the network "learns"
                self.w = [w - (eta / len(batch)) * nw for w, nw in zip(self.w, sum_nabla_w)]
                self.b = [b - (eta / len(batch)) * nb for b, nb in zip(self.b, sum_nabla_b)]
                
            print(f"Epoch {epoch} complete")
    
    def sigmoid(self, x):
        """The sigmoid function (logistic curve)."""

        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        """The derivative of the sigmoid function."""

        s = self.sigmoid(x)
        return s * (1 - s)
    
    def MSE(self, a, y):
        """Mean squared error function (loss function)."""

        return 0.5 * np.sum((a - y)**2)

    def MSE_deriv(self, a, y):
        """The derivative of the mean squared error function."""

        return a - y
    
    def evaluate(self, testing_data):
        """
        Evaluate the network by testing it and getting
        how many times it gets the correct decision.
        """

        test_results = []
        for img in testing_data:
            # Get the maximum number in the output (the AI's decision)
            decision = np.argmax(self.feed_forward(img.values))
            test_results.append((decision, np.argmax(img.label)))

        # Get the amount of times the network decides correct
        return sum(int(x == y) for (x, y) in test_results)
    

if __name__ == "__main__":
    # Load saved data
    with open("src/training_data.pkl", "rb") as f:
        training_data = pickle.load(f)

    eta = 3 # Learning rate
    epochs = 30
    batch_size = 100

    network = Network([784, 30, 10])

    network.train(training_data, epochs, batch_size, eta)

    with open("src/trained_network.pkl", "wb") as f:
        pickle.dump(network, f)
