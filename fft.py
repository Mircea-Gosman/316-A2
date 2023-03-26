import utils as Utils
from fourier_operations import Fourier

def run_fast_mode(image, fourier):
    # Perform transform from Fourier
    initial_image = image
    f_transform = fourier.fast_transform(image)

    # Display original image & Log scale transform
    pass


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
