import numpy as np

FFT_REC_THRESHOLD = 10 # TODO: Experiment on that number

create_index_map = lambda signal, w, h: np.array([[ [i] * signal.shape[2] for i in range(signal.shape[w]) ] for j in range(signal.shape[h]) ])
X_k = lambda f, index, k, j_coef=-1: np.sum(f*np.exp( (j_coef * 1j * 2 * np.pi * k * index) / f.shape[1]), axis=1)
#create_coefs = lambda shape, k, div=2, j_coef=-1: np.stack((np.exp(j_coef * 1j * 2 * np.pi * np.arange(0, shape[0] // div) * k / (shape[0] // div)),) *shape[-1], axis=-1)
create_coefs = lambda shape, k, div=2, j_coef=-1: np.exp(j_coef * 1j * 2 * np.pi * np.arange(0, shape[0] // div) * k / (shape[0] // div))

X_k_2 = lambda f, index, k, j_coef=-1: np.sum(f*np.exp( (j_coef * 1j * 2 * np.pi * k * index) / f.shape[0]), axis=0)

class Fourier:
    def normal_transform(self, signal): # signal shape is img.height, img.width, channels
        one_dimension_transform = lambda f, index: np.array([ X_k(f, index, k) for k in range(f.shape[1])])

        index_map_M = create_index_map(signal, 1, 0)
        index_map_N_transposed = create_index_map(signal, 0, 1)

        one_dim_result_transposed = one_dimension_transform(signal, index_map_M)
        two_dim_result = one_dimension_transform(one_dim_result_transposed, index_map_N_transposed)

        return two_dim_result


    def fast_transform(self, signal, inverse=False):
        j_coef = 1 if inverse else -1
        middle_coef = lambda k, N: np.exp(j_coef * 1j * 2 * np.pi * k / N)
        
        # one_dim_res = np.array([ [ self._split(row, j_coef, k, middle_coef, create_coefs(row.shape, k, 1, j_coef)) for k in range(signal.shape[1])] for row in signal])
        # two_dim_res_T = np.array([ [ self._split(row, j_coef, k, middle_coef, create_coefs(row.shape, k, 1, j_coef)) for k in range(signal.shape[0])] for row in one_dim_res.transpose(1,0,2)])
        one_dim_res = np.apply_along_axis(lambda row: np.array([self._split(row, j_coef, k, middle_coef, create_coefs(row.shape, k, 1, j_coef)) for k in range(row.shape[0])]), axis=1, arr=signal)
        two_dim_res_T = np.apply_along_axis(lambda row: np.array([self._split(row, j_coef, k, middle_coef, create_coefs(row.shape, k, 1, j_coef)) for k in range(row.shape[0])]), axis=1, arr=one_dim_res.transpose(1,0,2))

        return two_dim_res_T.transpose(1,0,2)


    def _split(self, d1_signal, j_coef, k, middle_coef, coefs):
        if d1_signal.shape[0] <= FFT_REC_THRESHOLD or d1_signal.shape[0] == 1:
            return np.sum(d1_signal * coefs, axis=0)
        
        coefs = create_coefs(d1_signal.shape, k, 2, j_coef)

        odd = self._split(d1_signal[1::2], j_coef, k, middle_coef, coefs)
        even = self._split(d1_signal[::2], j_coef, k, middle_coef, coefs)

        return even + middle_coef(k, d1_signal.shape[0]) * odd
    

    def fft2(self, x, inverse=False):
        j_coef = 1 if inverse else -1
        d1 = np.empty(x.shape,dtype = 'complex_')
        d2 = np.empty(x.shape,dtype = 'complex_')
        
        for n in range(x.shape[0]):
            d1[n,:] = self.fft(x[n,:], j_coef)

        for m in range(x.shape[1]):
            d2[:, m] = self.fft(d1[:,m], j_coef)

        return d2


    def fft(self, x, j_coeff):
        if x.shape[0] == 1 or x.shape[0] < FFT_REC_THRESHOLD:
            return np.array([ X_k_2(x, np.arange(x.shape[0]), k, j_coeff) for k in range(x.shape[0]) ])
        
        coeffs = np.exp(j_coeff * 2j * np.pi * np.arange(x.shape[0]) / x.shape[0])
        x_even = self.fft(x[::2], j_coeff)
        x_odd = self.fft(x[1::2], j_coeff)

        return np.concatenate([x_even + coeffs[:x.shape[0]//2] * x_odd, x_even + coeffs[x.shape[0]//2:] * x_odd])