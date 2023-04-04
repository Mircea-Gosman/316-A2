import numpy as np
import utils as Utils

create_index_map = lambda signal, w, h: np.concatenate([[np.mgrid[0:signal.shape[w], 0:signal.shape[-1]][0]]] * signal.shape[h], axis=0)
X_k = lambda f, index, k, j_coef=-1, axis=1: np.sum(f*np.exp( (j_coef * 1j * 2 * np.pi * k * index) / f.shape[axis]), axis=axis)

def normal_transform(signal):
    print("Taking Normal Transform...")
    signal = signal if len(signal.shape) <= 2 else signal[:,:, 0]
    one_dimension_transform = lambda f, index: np.array([ X_k(f, index, k) for k in range(f.shape[1])])

    one_dim_result_transposed = one_dimension_transform(signal, create_index_map(signal, 1, 0)[:,:, 0])
    two_dim_result = one_dimension_transform(one_dim_result_transposed, create_index_map(signal, 0, 1)[:,:, 0])

    return two_dim_result   


def fast_transform(signal, FFT_REC_THRESHOLD=32, inverse=False):
    annoucement = "Taking FFT..." if not inverse else "Taking Inverse..."
    print(annoucement)

    j_coef = 1 if inverse else -1
    signal = signal if len(signal.shape) <= 2 else signal[:,:, 0]
    d1 = np.empty(signal.shape,dtype = 'complex_')
    d2 = np.empty(signal.shape,dtype = 'complex_')
    
    for n in range(signal.shape[0]):
        d1[n,:] = fft(signal[n,:], j_coef, FFT_REC_THRESHOLD)

    for m in range(signal.shape[1]):
        d2[:, m] = fft(d1[:,m], j_coef, FFT_REC_THRESHOLD)

    return d2 if not inverse else d2 / (signal.shape[0] * signal.shape[1])


def fft(signal, j_coeff, FFT_REC_THRESHOLD):
    if signal.shape[0] == 1 or signal.shape[0] < FFT_REC_THRESHOLD:
        return np.array([ X_k(signal, np.arange(signal.shape[0]), k, j_coeff, 0) for k in range(signal.shape[0]) ])
    
    coeffs = np.exp(j_coeff * 2j * np.pi * np.arange(signal.shape[0]) / signal.shape[0])
    x_even = fft(signal[::2], j_coeff, FFT_REC_THRESHOLD)
    x_odd = fft(signal[1::2], j_coeff, FFT_REC_THRESHOLD)

    # Book Reference for Concatenation use case: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html#tricks-in-fft
    return np.concatenate([x_even + coeffs[:signal.shape[0]//2] * x_odd, x_even + coeffs[signal.shape[0]//2:] * x_odd])


## Set some frequencies to zero, CLI print number of non zero values left [ high frequencies -> 2pi/N * k not close to 0]
def filter_frequencies(signal, thresholds, scheme="high_frequency", verbose=True):
    if verbose:
        print(f"Filtering frequencies over {thresholds}...")
    dims = [ [1, 0], [0,1] ]

    for i in range(len(dims)):
        frequencies = create_index_map(signal, *dims[i])[:,:,0] * 2. * np.pi / signal.shape[dims[i][0]]
        frequencies = frequencies if i == 0 else frequencies.T

        if scheme == "high_frequency":
            f_close_to_0, f_close_to_2pi  = np.copy(signal), np.copy(signal)

            f_close_to_0[frequencies > thresholds[i]] = 0
            f_close_to_2pi[frequencies < (2. * np.pi - thresholds[i])] = 0

            signal = f_close_to_0 + f_close_to_2pi
        if scheme == "low_frequency":
            signal[frequencies < thresholds[i]] = 0
            signal[frequencies > (2. * np.pi - thresholds[i])] = 0
        
    remaining_coef_count = np.count_nonzero(signal != 0)

    if verbose:
        print(f"Filtering has left {remaining_coef_count} non-zero coefficients out of {signal.size}. ({remaining_coef_count/signal.size} of total)")

    return signal


# Set some coefficients to zero (6 different amounts of compression)
def compress(signal, factors, scheme="threshold"):
    if scheme == "random":
        transforms = [ np.copy(signal) * Utils.selection_matrix(signal.shape, factors[i]) for i in range(len(factors)) ]
    else:
        transforms = []
        
        for i in range(len(factors)):
            transform = np.copy(signal)

            if scheme == "threshold":
                quantity_to_remove = np.floor(signal.size*factors[i])
                if quantity_to_remove != 0:
                    indices = Utils.largest_indices(transform, quantity_to_remove)
                    transform[indices] = 0    
            else:
                # Frequencies schemes (one of low_frequency or high_frequency)
                transform = filter_frequencies(transform, [factors[i] * np.pi, factors[i] * np.pi], scheme, verbose=False)
            
            transforms.append(np.copy(transform))

        transforms = np.array(transforms)        

    if scheme == "high_frequency":
        transforms = np.flip(transforms, axis=0)

    # Save matrices of coefficients to csv & Print used coefficient count
    np.savetxt(f"./compression_data/non-compressed_fourier_transform.csv", np.zeros(transforms[i].shape, dtype="complex"), delimiter=",") # Minimum Possible
    for i in range(len(transforms)):
        np.savetxt(
            f"./compression_data/compressed_fourier_transform-{factors[i]}%.csv", 
            transforms[i], 
            delimiter=","
        )
        leftovers = np.count_nonzero(transforms[i] != 0)
        print(f'Image {i + 1} is using {leftovers} out of {transforms[i].size} coefficients.\t({leftovers/transforms[i].size} of total)')

    return transforms