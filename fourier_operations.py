import numpy as np

FFT_REC_THRESHOLD = 10 # TODO: Experiment on that number

create_index_map = lambda signal, w, h: np.concatenate([[np.mgrid[0:signal.shape[w], 0:signal.shape[-1]][0]]] * signal.shape[h], axis=0)
X_k = lambda f, index, k, j_coef=-1, axis=1: np.sum(f*np.exp( (j_coef * 1j * 2 * np.pi * k * index) / f.shape[axis]), axis=axis)

def normal_transform(signal): # signal shape is img.height, img.width, channels
    print("Taking Normal Transform...")
    one_dimension_transform = lambda f, index: np.array([ X_k(f, index, k) for k in range(f.shape[1])])

    one_dim_result_transposed = one_dimension_transform(signal, create_index_map(signal, 1, 0))
    two_dim_result = one_dimension_transform(one_dim_result_transposed, create_index_map(signal, 0, 1))

    return two_dim_result   


def fast_transform(signal, inverse=False):
    annoucement = "Taking FFT..." if not inverse else "Taking Inverse..."
    print(annoucement)

    j_coef = 1 if inverse else -1
    signal = signal if len(signal.shape) <= 2 else signal[:,:, 0]
    d1 = np.empty(signal.shape,dtype = 'complex_')
    d2 = np.empty(signal.shape,dtype = 'complex_')
    
    for n in range(signal.shape[0]):
        d1[n,:] = fft(signal[n,:], j_coef)

    for m in range(signal.shape[1]):
        d2[:, m] = fft(d1[:,m], j_coef)

    return d2 if not inverse else d2 / (signal.shape[0] * signal.shape[1])


def fft(signal, j_coeff):
    if signal.shape[0] == 1 or signal.shape[0] < FFT_REC_THRESHOLD:
        return np.array([ X_k(signal, np.arange(signal.shape[0]), k, j_coeff, 0) for k in range(signal.shape[0]) ])
    
    coeffs = np.exp(j_coeff * 2j * np.pi * np.arange(signal.shape[0]) / signal.shape[0])
    x_even = fft(signal[::2], j_coeff)
    x_odd = fft(signal[1::2], j_coeff)

    # Book Reference for Concatenation use case: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html#tricks-in-fft
    return np.concatenate([x_even + coeffs[:signal.shape[0]//2] * x_odd, x_even + coeffs[signal.shape[0]//2:] * x_odd])


## Set high frequencies to zero, CLI print number of non zero values left [ high frequencies -> 2pi/N * k not close to 0]
def filter_frequencies(signal, thresholds):
    print(f"Filtering frequencies over {thresholds}...")
    dims = [ [1, 0], [0,1] ]

    for i in range(len(dims)):
        frequencies = create_index_map(signal, *dims[i])[:,:,0] * 2. * np.pi / signal.shape[1]
        frequencies = frequencies if i == 0 else frequencies.T

        f_close_to_0, f_close_to_2pi  = np.copy(signal), np.copy(signal)

        f_close_to_0[frequencies > thresholds[i]] = 0
        f_close_to_2pi[frequencies < (2. * np.pi - thresholds[i])] = 0

        signal = f_close_to_0 + f_close_to_2pi
        
    remaining_coef_count = np.count_nonzero(signal != 0)
    print(f"Filtering has left {remaining_coef_count} non-zero coefficients. ({remaining_coef_count/ (signal.size)} of total)")

    return signal