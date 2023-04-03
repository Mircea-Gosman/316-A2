import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from time import time

import fourier_operations as Fourier

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
                if args["mode"] > 5 or args["mode"] < 0:
                    raise ValueError("Mode should be an integer between 1 and 5 inclusive.")
            except:                
                exit_with_error("type")

        if sys.argv[i] == "-i":
            if i == len(sys.argv) - 1:
                exit_with_error("syntax")

            try:
                args["image"] = plt.imread(sys.argv[i + 1]).astype(float)
            except:
                exit_with_error("path")

    if args["image"] is None:
        exit_with_error("path")

    args["image"] = pad_image(args["image"])

    return args


def selection_matrix(shape, factor):
    one_dimension = np.ones(shape[0] * shape[1])
    indices_to_delete = np.random.choice(len(one_dimension), size=int(shape[0] * shape[1] * factor), replace=False)
    one_dimension[indices_to_delete] = 0
    return  np.reshape(one_dimension, (shape[0], shape[1]))


# Source: https://stackoverflow.com/a/38884051
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    n = int(n)
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# Loop 10 times:
    # Create 2D arrays from sizes [2^5, 2^10] with random values
    # Record naive transform runtime for each
    # Record fast transform runtime for each
def runtime(num_iterations, sizes):
    times = np.empty((2, len(sizes), num_iterations)) # [ naive[size][iteration], fast[size][iteration] ]

    for i in range(num_iterations):
        for s in range(len(sizes)):            
            for t in range(len(times)):
                signal = np.random.rand(sizes[s], sizes[s])
                start = time()
                _ = Fourier.fast_transform(signal) if t == 1 else Fourier.normal_transform(signal)
                times[t][s][i] = time() - start

    return np.array(times)


def plot_transform(image, transform):
    # Shift the transform to center it on the plot
    transform = np.fft.fftshift(transform)

    # Plot
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,5))
    ax1.title.set_text(f"Original Image")
    ax2.title.set_text(f"Shifted Fast Fourier Transform Result")

    ax1.imshow(image, cmap=plt.cm.gray)
    transform_plot = ax2.imshow(np.abs(transform), norm=colors.LogNorm(vmin=5)) # Expected to be vmin = 0 and vmax = 2pi, but these throw errors.
    figure.colorbar(transform_plot, ax=ax2, fraction=0.026)

    plt.show()


def plot_images(images, dims):
    _, axes = plt.subplots(*dims, figsize=(15,15))
    axes = np.array(axes).flatten()
    plt.gray() 

    for i in range(len(images)):        
        axes[i].imshow(np.abs(images[i]))
        axes[i].title.set_text(f"Image {i + 1}")

    plt.show()


# Plot: x -> problem size, y -> corresponding runtime mean
# Include error bars to be twice the standard deviation 
def plot_statistics(sizes, means, std_devs, labels):
    for i in range(len(labels)): 
        plt.errorbar(sizes, means[i], yerr= 2*std_devs[i], label = labels[i])
    
    plt.title('FFT & Naive Transform Runtimes as a function of problem size', fontsize=12)
    plt.xlabel('Problem Unliateral Size (pixels)', fontsize=10)
    plt.ylabel('Runtime (s)', fontsize=10)
    plt.legend()
    plt.show()