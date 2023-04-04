# Fourier Transforms
## ECSE 316 Assignment 2

### Group Members
Abdelmadjid Kamli (260984339)
Mircea Gosman     (260983354)

### Calling Syntax
python fft.py [-m mode] [-i image]

image (optional):
Path to an image to process. Defaults to `moonlanding.png`

mode (optional):
- [1] (Default) for fast mode where the image is converted into its FFT form and displayed
- [2] for denoising 
- [3] for compressing and saving the created transforms
- [4] for plotting the runtime graphs 
- [5] For plotting accuracy
- [6] for plotting accuracy & runtime of various FFT splitting thresholds