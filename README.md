# MNIST Neural Network

This project mostly follows Michael Nielson's book *Neural Networks and Deep Learning* as a guide. \
Link: http://neuralnetworksanddeeplearning.com/

## Features

 - File data conversions
 - OOP network
 - Sigmoid neuron network
 - Stochastic Gradient Descent

## How to Run

1. Run `convert.py` to convert the MNIST ubtye files into `.csv` files.
2. Run `training.py` to save the data of the `.csv` files in `testing_data.pkl` and `training_data.pkl`.
3. Run `network.py` to train the neural network (or use the pretrained network), and you can change its settings in the code.
4. Run `network_results` to test the trained network and print its results.

## Personal Reflection

I started this project as a stepping stone to my journey through the world of AI and machine learning. Throughout this project, I was able to (eventually) grasp and apply concepts of calculus (which I recently learned in the Calculus 12 course) and to a certain extent - some concepts of linear algebra. I believe that this project was incredibly beneficial to me and I hope to move further to the next steps of neural networks!

## Personal Challenges

 - Backpropogation intuition
 - Applied linear algebra and calculus in stochastic gradient descent

## What I Learned

 - The system of a basic neural network
 - How neural networks "learn"
 - Sigmoid neurons
 - Backpropagation
 - Stochastic gradient descent (and gradient descent)
 - Converting data between and from ubyte and csv files

## Example Networks

Settings:
- Layers: [784, 32, 32, 10]
- Learning rate: 3
- Epochs: 30
- Batch size: 100

### Neural network from scratch

Time taken to train the network: 264.65 sconds

### Neural network using PyTorch

Time taken to train the network: 95.30 seconds
