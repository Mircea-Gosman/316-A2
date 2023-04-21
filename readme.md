<h1 align='center'>Fourier Transform Image Denoising</h1>
<h3 align='center'>Implemented within the Scope of ECSE 316 at McGill University</h3>

## Authors
* Mircea Gosman - mirceagosman@gmail.com <br>
* Abdelmadjid Kamli 

## Abstract
Numpy's Fast Fourier Transform (FFT) is rapid, but how does it work internally?
From scratch, this project implements the [Cooley-Tuckey algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm): a divide-and-conquer approach to the FFT. We also look into its use in the 2D plane by applying the algorithm to noisy images.

## Usage Guide
The code enables the user to navigate and test our reported results. It may:
* Display the frequency domain of an image by [taking its FFT](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html#tricks-in-fft)
* Denoise an image by taking its FFT, thresholding high frequencies, and inverting the FFT (IFFT)
* Use denoising to compress the image
* Report accuracy measures against Numpy's built-in implementation of the [2D-FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)
* Graph our methodology to obtain the Cooley-Tuckey recursive threshold for our machine.


### Syntax
`python fft.py [-m mode] [-i image]`

**image (optional):** <br>
Path to an image to process. Defaults to `moonlanding.png`

**mode (optional):** <br>
- [1] (Default) for fast mode where the image is converted into its FFT form and displayed
- [2] for denoising 
- [3] for compressing and saving the created transforms
- [4] for plotting the runtime graphs 
- [5] For plotting accuracy
- [6] for plotting accuracy & runtime of various FFT splitting thresholds