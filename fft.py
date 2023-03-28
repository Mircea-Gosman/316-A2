import matplotlib.pyplot as plt
import matplotlib.colors as colors
import utils as Utils
from fourier_operations import Fourier
from fourier_operations import create_index_map
import numpy as np
import math
from time import time

def run_fast_mode(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

    # Display original image & Log scale transform
    n = np.array([[ [i] * f_transform.shape[2] for i  in range(f_transform.shape[1])] for j in range (f_transform.shape[0])])    
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,5))
    ax1.imshow(image)
    ax2.scatter(n[:,:,0] , f_transform[:,:,0], norm=colors.LogNorm())
    # ax3.scatter(n[:,:,0], np.fft.fft2(image[:,:,0]), norm=colors.LogNorm())
    # plt.yscale('log')
    plt.show()
# frequency: https://www.mathworks.com/help/matlab/math/fourier-transforms.html

def denoise(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

    # Set high frequencies to zero, CLI print number of non zero values left [ high frequencies -> 2pi/N * k not close to 0]
    HIGH_FREQ_THRESH = math.pi * 1./2  # TODO: experiment with that number
    frequencies = create_index_map(f_transform, 1, 0) * 2. * math.pi / f_transform.shape[1]
    f_transform[frequencies < HIGH_FREQ_THRESH] = 0
    f_transform[frequencies > (2. * math.pi - HIGH_FREQ_THRESH)] = 0

    # Perform inverse transform from Fourier
    inverse_transform = fourier.fast_transform(f_transform, inverse=True)

    # Display original image & denoised image
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,5))
    ax1.imshow(image)
    ax2.imshow(inverse_transform)
    plt.show()


def compress(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

    # Save matrix of coefficients to csv 
    np.savetxt("compression_fourier_transform.csv", f_transform, delimiter=",")
    
    # Set some coefficients to zero (6 different amounts of compression: {0, ... , 95%}), TODO: should experiment with selection scheme
    compression_factors = [ 0, 0.19, 0.38, 0.57, 0.76, 0.95 ]
    transforms = [ np.copy(f_transform) * Utils.selection_matrix(f_transform.shape, compression_factors[i]) for i in range(6) ]

    # [? Save the 6 resulting matrices of coefficients to csv ?] - (I can't tell what they mean on p.4)
    # CLI Print number of non zero coeffs left in each image
    for i in range(6):
        original_size = transforms[i].shape[0] * transforms[i].shape[1]
        print(f'Image {i} is using {transforms[i].shape[0] * (1 - compression_factors[i])} out of {original_size}')

    # Inverse the 6 resulting transforms
    inverse_transforms = [ fourier.fast_transform(transforms[i], inverse=True) for i in range(6) ]

    # Display the 6 images
    figure, axes = plt.subplots(2, 3, figsize=(5,5))
    for i in range(axes):
        axes[i].imshow(inverse_transforms[i])

    plt.show()


def plot(fourier):
    num_iterations = 10
    sizes = [2**5, 2**6, 2**7, 2**8,2**9,2**10]
    times = np.empty((2, len(sizes), num_iterations)) # [ naive[size][iteration], fast[size][iteration] ]

    # Loop 10 times:
        # Create 2D arrays from sizes [2^5, 2^10] with random values
        # Record naive transform runtime for each
        # Record fast transform runtime for each
    for i in range(num_iterations):
        for s in range(len(sizes)):            
            for t in range(len(times)):
                signal = np.random.rand(sizes[s], sizes[s], 3)
                start = time()
                _ = fourier.fast_transform(signal) if t == 1 else fourier.normal_transform(signal)
                times[t][s][i] = time() - start
                print(f'{t} {s} {i}')

    # Record mean and standard deviation per problem size
    np_times = np.array(times)
    means = np.average(np_times, axis=-1)
    standard_deviations = np.std(np_times, axis=-1)

    # Plot: x -> problem size, y -> corresponding runtime mean
    # Include error bars to be twice the standard deviation 
    plt.errorbar(sizes, means[0], yerr= 2*standard_deviations[0], label = "Naive")
    plt.errorbar(sizes, means[1], yerr= 2*standard_deviations[1], label = "Fast")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = Utils.check_CLI()
    fourier = Fourier()

    if args["mode"] == 1:
        run_fast_mode(args["image"], fourier)
    if args["mode"] == 2:
        denoise(args["image"], fourier)
    if args["mode"] == 3:
        compress(args["image"], fourier)
    if args["mode"] == 4:
        plot(fourier)
