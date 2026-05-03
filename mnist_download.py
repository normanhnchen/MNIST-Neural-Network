import torchvision.datasets as dset
dataset = dset.MNIST(root='./data', train=True, download=True)
