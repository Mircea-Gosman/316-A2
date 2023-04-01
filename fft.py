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
    print(image.shape)                          # (512, 1024, 3) TODO: why is it not the same
    f_transform = fourier.fast_transform(image) #(1024, 512, 3) 

    # Display original image & Log scale transform 
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5,5))
    ax1.imshow(image)
    print(f_transform.real.shape)
    ax2.imshow(f_transform[:,:,0].real, norm=colors.LogNorm())
    ax3.imshow(np.fft.fft2(image[:,:,0]).real, norm=colors.LogNorm())
    plt.show()
# frequency: https://www.mathworks.com/help/matlab/math/fourier-transforms.html

def denoise(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

    # Set high frequencies to zero, CLI print number of non zero values left [ high frequencies -> 2pi/N * k not close to 0]
    HIGH_FREQ_THRESH = math.pi * 0.05  # TODO: experiment with that number
    frequencies = create_index_map(f_transform, 1, 0) * 2. * math.pi / f_transform.shape[1]
    f_transform[frequencies < HIGH_FREQ_THRESH] = 0
    f_transform[frequencies > (2. * math.pi - HIGH_FREQ_THRESH)] = 0

    # Perform inverse transform from Fourier
    inverse_transform = fourier.fast_transform(f_transform, inverse=True)

    # Display original image & denoised image
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,5))
    ax1.imshow(image)
    ax2.imshow(inverse_transform.real[:,:,0], norm=colors.LogNorm())
    plt.show()


def compress(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

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
    inverse_transforms = [ fourier.fast_transform(transforms[i], inverse=True) for i in range(6) ]

    # Display the 6 images
    figure, axes = plt.subplots(2, 3, figsize=(5,5))
    for i in range(len(axes)):
        axes[i].imshow(inverse_transforms[i].real[:,:,0], norm=colors.LogNorm())

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


def accuracy(image, fourier):
    image = np.array([
        [[1, 1, 1], [20,20,20] , [5,5,5], [1, 1, 1], [20,20,20] , [5,5,5], [10, 10, 10] , [7,7,7]],
        [[2, 2, 2], [10, 10, 10] , [7,7,7], [1, 1, 1], [20,20,20] , [5,5,5], [10, 10, 10] , [7,7,7]],
    ])

    # Ours
    # naive_transform = fourier.normal_transform(image)
    print('ttt')
    np_fft = np.fft.fft2(image, axes=(0, 1))
    print('tttt')
    print('t')
    fast_transform = fourier.fast_transform(image)
    print('tt')
    # inverse_fast_transform = fourier.fast_transform(image, inverse=True)
    # print(naive_transform[:,:,0])
    # print("----")
    # print(fast_transform[:,:,0])
    # print("----")

    # Numpy
    print('ttt')
    np_fft = np.fft.fft2(image, axes=(0, 1))
    print('tttt')
    np_ifft = np.fft.ifft2(np_fft)
    # print(np_fft[:,:,0])
    # RMSs
    rms = lambda y, z: np.sqrt(np.mean((y - z)**2))
    print("Root mean squared errors between our transforms & Numpy's:")
    # print(f"\tNaive transform is\t{rms(naive_transform, np_fft)}")
    print(f"\tFast transform is\t{rms(fast_transform, np_fft)}")
    # print(f"\tFast transform is\t{((naive_transform - np_fft)**2).mean()}")
    # print(f"\tFast transform is\t{((naive_transform - fast_transform)**2).mean()}")
    # print(f"\tFast inverse transform is\t{((inverse_fast_transform - np_ifft)**2).mean()}")

    print(np.allclose(fast_transform, np_fft, rtol=0, atol=np.exp(-15)))
    # print(np.allclose(naive_transform, np_fft, rtol=0, atol=np.exp(-15)))
    # print(np.allclose(naive_transform, fast_transform, rtol=0, atol=np.exp(-15)))
    
def simple_accuracy_manual_example():
    image = np.array([
        [[1, 1, 1], [20,20,20]],
        [[2, 2, 2], [10, 10, 10]]
    ])

    # print(-1j * 2 * np.pi * 1 / 2)

    print(1* np.exp(0j * 0) + 20* np.exp(0j * 1))
    print(1* np.exp(-3.141592653589793j * 0) + 20* np.exp(-3.141592653589793j * 1))


    print(2* np.exp(0j * 0) + 10* np.exp(0j * 1))
    print(2* np.exp(-3.141592653589793j * 0) + 10* np.exp(-3.141592653589793j * 1))

    print("----")
    row1 = [ 1, 20]
    row2 = [2, 10]
    print(np.fft.fft(row1))
    print(np.fft.fft(row2))
    
    print("----")
    print("----")
    print((21+0j) * np.exp(0j * 0) + (12+0j) * np.exp(0j * 1))
    print((-19-2.4492935982947065e-15j) * np.exp(0j * 0) + (-8-1.2246467991473533e-15j) * np.exp(0j * 1))
    print((21+0j) * np.exp(-3.141592653589793j * 0) + (12+0j) * np.exp(-3.141592653589793j * 1))
    print((-19-2.4492935982947065e-15j) * np.exp(-3.141592653589793j * 0) + (-8-1.2246467991473533e-15j) * np.exp(-3.141592653589793j * 1))
    print("----")
    row1 = [ 21.+0.j,  12.+0.j]
    row2 = [ -19.+0.j, -8.+0.j]
    print(np.fft.fft(row1))
    print(np.fft.fft(row2))
    print("----")
    print("----")
    mat = [
        [1, 20],
        [2, 10]
    ]
    
    print(np.fft.fft2(mat))

    return

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
    if args["mode"] == 5:
        accuracy(args["image"], fourier)
