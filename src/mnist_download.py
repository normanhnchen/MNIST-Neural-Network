"""Downloads MNIST dataset files using torchvision."""


import torchvision.datasets as dset
dataset = dset.MNIST(root="./data", train=True, download=True)
