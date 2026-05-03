"""All classes used in the network."""


import numpy as np
import random


class Image:
    def __init__(self, label, values):
        self.label = label
        self.values = values


class Network:
    def __init__(self, sizes):
        self.sizes = np.array(sizes)
        self.num_layers = len(sizes)
        self.a = 0
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feed_forward(self, a):
        for w, b in zip(self.w, self.b):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def backprop(self, input, target):
        nabla_w = [np.zeros(w.shape) for w in self.w]
        nabla_b = [np.zeros(b.shape) for b in self.b]

        a = input
        a_list = [input]
        z_list = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            z_list.append(z)
            a_list.append(a)

        dC_da = self.MSE_deriv(a_list[-1], target)
        da_dz = self.sigmoid_deriv(z_list[-1])
        dC_dz = dC_da * da_dz
        nabla_w[-1] = np.dot(dC_dz, a_list[-2].transpose())
        nabla_b[-1] = dC_dz
        
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
        """Train the network using stochastic gradient descent."""

        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]
            
            for batch in batches:
                sum_nabla_w = [np.zeros(w.shape) for w in self.w]
                sum_nabla_b = [np.zeros(b.shape) for b in self.b]
                
                for img in batch:
                    curr_nabla_w, curr_nabla_b = self.backprop(img.values, img.label)
                    
                    sum_nabla_w = [sw + nw for sw, nw in zip(sum_nabla_w, curr_nabla_w)]
                    sum_nabla_b = [sb + nb for sb, nb in zip(sum_nabla_b, curr_nabla_b)]
                
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
        """Evaluate"""
        # Get the maximum number in the output
        test_results = []
        for img in testing_data:
            prediction = np.argmax(self.feed_forward(img.values))
            test_results.append((prediction, np.argmax(img.label)))

        return sum(int(x == y) for (x, y) in test_results)
