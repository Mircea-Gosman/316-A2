import utils as Utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import fourier_operations as Fourier
from fourier_operations import create_index_map
import numpy as np
import math
from time import time

def run_fast_mode(image):
    f_transform = Fourier.fast_transform(image)
    Utils.plot_transform(image, f_transform)


def denoise(image, HIGH_FREQ_THRESH=[np.pi * 0.09,  np.pi * 0.09]):
    f_transform        = Fourier.fast_transform(image)     
    filtered_transform = Fourier.filter_frequencies(f_transform, HIGH_FREQ_THRESH)
    inverse_transform  = Fourier.fast_transform(filtered_transform, inverse=True)
    Utils.plot_images([image, inverse_transform], [1, 2])


def compress(image):
    # Perform transform from Fourier
    f_transform = Fourier.fast_transform(image)

    # Save matrix of coefficients to csv 
    np.savetxt("compression_fourier_transform.csv", f_transform[:, :, 0], delimiter=",") # Saving only 1 channel for demo purposes
    
    # Set some coefficients to zero (6 different amounts of compression: {0, ... , 95%}), TODO: should experiment with selection scheme
    compression_factors = [ 0, 0.19, 0.38, 0.57, 0.76, 0.95 ]
    transforms = [ np.copy(f_transform) * Utils.selection_matrix(f_transform.shape, compression_factors[i]) for i in range(6) ]

    # [? Save the 6 resulting matrices of coefficients to csv ?] - (I can't tell what they mean on p.4)
    # CLI Print number of non zero coeffs left in each image
    for i in range(6):
        original_size = transforms[i].shape[0] * transforms[i].shape[1]
        print(f'Image {i} is using {int(original_size * (1 - compression_factors[i]))} out of {original_size}')

    # Inverse each of the 6 resulting transforms
    inverse_transforms = [ Fourier.fast_transform(transforms[i], inverse=True) for i in range(6) ]

    # Display the 6 images
    figure, axes = plt.subplots(2, 3, figsize=(5,5))
    for i in range(len(axes)):
        axes[i].imshow(inverse_transforms[i].real[:,:,0], norm=colors.LogNorm())

    plt.show()


def plot():
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
                _ = Fourier.fast_transform(signal) if t == 1 else Fourier.normal_transform(signal)
                times[t][s][i] = time() - start

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


def accuracy(image, TOLERANCE=-10):
    # Smaller Toy Data
    # image = np.array([
    #     [[1, 1, 1], [20,20,20] , [5,5,5], [1, 1, 1], [20,20,20] , [5,5,5], [10, 10, 10] , [7,7,7]],
    #     [[2, 2, 2], [10, 10, 10] , [7,7,7], [1, 1, 1], [20,20,20] , [5,5,5], [10, 10, 10] , [7,7,7]],
    # ])

    # Results
    naive_transform = Fourier.normal_transform(image)[:,:,0]
    fast_transform = Fourier.fast_transform(image)
    inverse_fast_transform = Fourier.fast_transform(naive_transform, inverse=True)

    # Numpy Results
    print("Checking Numpy Transforms...")
    np_fft = np.fft.fft2(image, axes=(0, 1))[:,:,0]
    np_ifft = np.fft.ifft2(np_fft, axes=(0, 1))
    
    # RMSs & Tolerances
    rms = lambda y, z: np.sqrt(np.mean((y - z)**2))
    naive_tol = np.allclose(naive_transform, np_fft, rtol=0, atol=np.exp(TOLERANCE))
    fast_tol = np.allclose(fast_transform, np_fft, rtol=0, atol=np.exp(TOLERANCE))
    fast_inv_tol = np.allclose(naive_transform, np_fft, rtol=0, atol=np.exp(TOLERANCE))

    # Display
    print("Root mean squared errors between our transforms & Numpy's:")
    print(f"\tNaive transform is\t\t{rms(naive_transform, np_fft)}\t\t| Within 10^{TOLERANCE} tolerance:\t{naive_tol}")
    print(f"\tFast transform is\t\t{rms(fast_transform, np_fft)}\t\t| Within 10^{TOLERANCE} tolerance:\t{fast_tol}")
    print(f"\tFast inverse transform is\t{rms(inverse_fast_transform, np_ifft)}\t\t| Within 10^{TOLERANCE} tolerance:\t{fast_inv_tol}")

    
if __name__ == "__main__":
    args = Utils.check_CLI()

    if args["mode"] == 1:
        run_fast_mode(args["image"])
    if args["mode"] == 2:
        denoise(args["image"])
    if args["mode"] == 3:
        compress(args["image"])
    if args["mode"] == 4:
        plot()
    if args["mode"] == 5:
        accuracy(args["image"])
