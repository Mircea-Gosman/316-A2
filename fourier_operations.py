import numpy as np
import math

FFT_REC_THRESHOLD = 25 # TODO: Experiment on that number

create_index_map = lambda signal, w, h: np.array([[ [i] * signal.shape[2] for i in range(signal.shape[w]) ] for j in range(signal.shape[h]) ])

class Fourier:
    def __init__(self):
        self.memo = {}

    def normal_transform(self, signal): # signal shape is img.height, img.width, channels
        one_dimension_transform = lambda f, index: np.array([ np.sum(f * math.e**( (-2j * math.pi * k * index) / f.shape[1]), axis=1) for k in range(f.shape[1])])

        index_map_M = create_index_map(signal, 1, 0)
        index_map_N = create_index_map(signal, 0, 1)

        one_dim_result_transposed = one_dimension_transform(signal, index_map_M)
        two_dim_result = one_dimension_transform(one_dim_result_transposed, index_map_N)

        return two_dim_result


    def _split(self, signal, index, k, coef):
        signal_key = np.array2string(index)

        if signal_key in self.memo:
            return self.memo[signal_key]

        if signal.shape[1] < FFT_REC_THRESHOLD:
            res = np.sum(signal * math.e**( (-2j * math.pi * k * index) / signal.shape[1]), axis=1)
            self.memo[signal_key] = res
            return res

        odd_index = index[:, 1::2]
        even_index = index[:, ::2]
        odd = signal[:, 1::2]
        even = signal[:, ::2]
        
        return self._split(even, even_index, k, coef) + coef(k, signal.shape[1]) * self._split(odd, odd_index, k, coef) 


    def fast_transform(self, signal):
        coef = lambda k, N: math.e** (-2j * math.pi * k / N)

        index_map_M = create_index_map(signal, 1, 0)
        index_map_N = create_index_map(signal, 0, 1)

        self.memo = {}
        one_dim_result_transposed = np.array([ self._split(signal, index_map_M, k, coef) for k in range(signal.shape[1]) ])
        self.memo = {}
        two_dim_result = np.array([ self._split(one_dim_result_transposed, index_map_N, k, coef) for k in range(one_dim_result_transposed.shape[1]) ])
        self.memo = {}

        return two_dim_result