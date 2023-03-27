import matplotlib.pyplot as plt
import matplotlib.colors as colors
import utils as Utils
from fourier_operations import Fourier
import numpy as np

def run_fast_mode(image, fourier):
    # Perform transform from Fourier
    f_transform = fourier.fast_transform(image)

    # Display original image & Log scale transform
    n = np.array([[ [i] * f_transform.shape[2] for i  in range(f_transform.shape[1])] for j in range (f_transform.shape[0])])    
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5,5))
    ax1.imshow(image)
    ax2.scatter(n , f_transform, norm=colors.LogNorm())
    ax3.scatter(n , np.fft.fft2(image), norm=colors.LogNorm())
    plt.show()


def denoise(image, fourier):
    # Perform transform from Fourier
    # Set high frequencies to zero, CLI print number of non zero values left
    # Perform inverse transform from Fourier
    # Display original image
    # Display denoised image
    pass


def compress(image, fourier):
    # Perform transform from Fourier
    # Save matrix of coefficients to csv 
    # Set some coefficients to zero (6 different amounts of compression: {0, ... , 95%})
    # [? Save the 6 resulting matrices of coefficients to csv ?] - (I can't tell what they mean on p.6)
    # CLI Print number of non zero coeffs left in each image
    # Inverse the 6 resulting transforms
    # Display the 6 images
    pass


def plot(image, fourier):
    # Loop 10 times:
        # Create 2D arrays from sizes [2^5, 2^10] with random values
        # Record naive transform runtime for each
        # Record fast transform runtime for each

    # Record mean and standard deviation per problem size
    # Plot: x -> problem size, y -> corresponding runtime mean
    # Include error bars to be (twice?) the standard deviation 
    pass


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
        plot(args["image"], fourier)
