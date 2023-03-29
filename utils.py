import sys
import math
import cv2
import numpy as np
from numpy import pi
from colorsys import hls_to_rgb

def exit_with_error(error):
    if error == "syntax":
        print("Invalid Syntax. Expected: fft.py -m mode [-i image]")
    if error == "path": 
        print("Provided image path is incorrect.")
    if error == "type": 
        print("Expected mode to be an integer in range [1, 4]")

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
        "image": cv2.imread("moonlanding.png") # Numpy array
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

            args["image"] = cv2.imread(sys.argv[i + 1])

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


# Source: https://stackoverflow.com/questions/17044052/how-to-create-a-phase-plot-for-a-2d-array-of-complex-numbers-with-matplotlib
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0)

    return c
