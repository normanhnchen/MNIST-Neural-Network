"""Converts MNIST ubyte files into csv files."""


def convert(imgs, labels, outfile, n):
    imgf = open(imgs, "rb") # Read binary file
    labelf = open(labels, "rb") # Read binary file
    csvf = open(outfile, "w") # Write to csv file

    # Skip the header of the image file
    imgf.read(16)
    # Skip the header of the label file
    labelf.read(8)

    images = []
    # Read n images and their corresponding labels
    for _ in range(n):
        # Read the label and get the integer value
        image = [labelf.read(1)[0]]
        # Read 784 pixels (28x28) for each image and get the integer value of each pixel
        for _ in range(784):
            pixel = imgf.read(1)
            image.append(pixel[0])
        images.append(image)
    
    # Convert to csv format and write to the output file
    for image in images:
        csvf.write(",".join(str(pixel) for pixel in image) + "\n")
    
    # Close all files
    imgf.close()
    labelf.close()
    csvf.close()

mnist_train_x = "MNIST/raw/train-images-idx3-ubyte"
mnist_train_y = "MNIST/raw/train-labels-idx1-ubyte"
mnist_test_x = "MNIST/raw/t10k-images-idx3-ubyte"
mnist_test_y = "MNIST/raw/t10k-labels-idx1-ubyte"

convert(mnist_train_x, mnist_train_y, "MNIST/csv/mnist_train.csv", 60000)
convert(mnist_test_x, mnist_test_y, "MNIST/csv/mnist_test.csv", 10000)
