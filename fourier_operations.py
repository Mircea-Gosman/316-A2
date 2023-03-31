import numpy as np

FFT_REC_THRESHOLD = 50 # TODO: Experiment on that number

create_index_map = lambda signal, w, h: np.array([[ [i] * signal.shape[2] for i in range(signal.shape[w]) ] for j in range(signal.shape[h]) ])
X_k = lambda f, index, k: np.sum(f*np.exp( (-1j * 2 * np.pi * k * index) / f.shape[1]), axis=1)

class Fourier:
    def normal_transform(self, signal): # signal shape is img.height, img.width, channels
        one_dimension_transform = lambda f, index: np.array([ X_k(f, index, k) for k in range(f.shape[1])])

        index_map_M = create_index_map(signal, 1, 0)
        index_map_N_transposed = create_index_map(signal, 0, 1)

        one_dim_result_transposed = one_dimension_transform(signal, index_map_M)
        two_dim_result = one_dimension_transform(one_dim_result_transposed, index_map_N_transposed)

        return two_dim_result


    def fast_transform(self, signal, inverse=False):
        coef = lambda k, N: np.exp(-1j * 2 * np.pi * k / N)
        op = X_k

        # Basically removing a minus
        if inverse:
            op = lambda f, index, k:  np.sum(f * np.exp( (1j * 2 * np.pi * k * index) / f.shape[1]), axis=1) 
            coef = lambda k, N: np.exp(1j * 2 * np.pi * k / N)


        self.memo = {}
        one_dim_result_transposed = np.array([ self._split(signal, k, coef, op) for k in range(signal.shape[1]) ])#.transpose(1,0,2)
        self.memo = {}
        print("q")
        two_dim_result = np.array([ self._split(one_dim_result_transposed, k, coef, op) for k in range(one_dim_result_transposed.shape[1]) ])
        self.memo = {}
        
        return two_dim_result if not inverse else two_dim_result / (signal.shape[0] * signal.shape[1])
    

    def _split(self, signal, k, coef, op):
        signal_key = str(k) + ":" + np.array2string(signal)

        if signal_key in self.memo:
            return self.memo[signal_key]

        if signal.shape[1] <= FFT_REC_THRESHOLD or signal.shape[1] == 1:
            index = np.stack((np.mgrid[0:signal.shape[1], 0:signal.shape[0]][0].T,) *signal.shape[-1], axis=-1)
            self.memo[signal_key] = op(signal, index, k)
            return self.memo[signal_key]

        odd = signal[:, 1::2]
        even = signal[:, ::2]
        
        return self._split(even, k, coef, op) + coef(k, signal.shape[1]) * self._split(odd, k, coef, op)