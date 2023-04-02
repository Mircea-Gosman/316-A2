import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from colorsys import hls_to_rgb

def exit_with_error(error):
    if error == "syntax":
        print("Invalid Syntax. Expected: fft.py -m mode [-i image]")
    if error == "path": 
        print("Provided image path is incorrect.")
    if error == "type": 
        print("Expected mode to be an integer in range [1, 5]")

    exit(1)


def pad_image(image):
    power_of_2 = lambda x: 2**(math.ceil(math.log2(x)))
    new_dim = lambda x: x if (x & (x-1) == 0) and x != 0 else power_of_2(x)
    
    height = new_dim(image.shape[1])
    width = new_dim(image.shape[0])

    return cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)


def check_CLI():
    args = {
        "mode": 1,
        "image": plt.imread("moonlanding.png").astype(float) # Numpy array
    }

    if len(sys.argv) < 3:
        exit_with_error("syntax")

    # Collect Data
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-m":
            if i == len(sys.argv) - 1:
                exit_with_error("syntax")

            try:
                args["mode"] = int(sys.argv[i + 1])
            except:                
                exit_with_error("type")

        if sys.argv[i] == "-i":
            if i == len(sys.argv) - 1:
                exit_with_error("syntax")

            args["image"] = plt.imread(sys.argv[i + 1]).astype(float)

    if args["image"] is None:
        exit_with_error("path")

    args["image"] = pad_image(args["image"])

    return args


def selection_matrix(shape, factor):
    one_dimension = np.ones(shape[0] * shape[1])
    indices_to_delete = np.random.choice(len(one_dimension), size=int(shape[0] * shape[1] * factor), replace=False)
    one_dimension[indices_to_delete] = 0

    two_dims = np.reshape(one_dimension, (shape[0], shape[1]))

    # Add back channels
    return  np.stack((two_dims,)*shape[-1], axis=-1)


def plot_transform(image, transform):
    # Shift the transform to center it on the plot
    transform = np.fft.fftshift(transform)

    # Plot
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,5))

    ax1.imshow(image, cmap=plt.cm.gray)
    transform_plot = ax2.imshow(np.abs(transform), norm=colors.LogNorm(vmin=5))
    figure.colorbar(transform_plot, ax=ax2, fraction=0.026)

    plt.show()


def plot_images(images, dims):
    _, axes = plt.subplots(*dims, figsize=(15,15))
    plt.gray() 
    
    for i in range(len(images)):        
        axes[i].imshow(np.abs(images[i]))

    plt.show()
