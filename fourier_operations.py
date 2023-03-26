import numpy as np
import math

create_index_map = lambda signal, w, h: np.array([[ [i] * signal.shape[2] for i in range(signal.shape[w]) ] for j in range(signal.shape[h]) ])

def normal_transform(signal): # signal shape is img.height, img.width, channels
    one_dimension_transform = lambda f, index: np.array([ np.sum(f * math.e**( (-2j * math.pi * k * index) / f.shape[0]), axis=1) for k in range(f.shape[1])])

    index_map_M = create_index_map(signal, 1, 0)
    index_map_N = create_index_map(signal, 0, 1)

    one_dim_result_transposed = one_dimension_transform(signal, index_map_M)
    two_dim_result = one_dimension_transform(one_dim_result_transposed, index_map_N)

    return two_dim_result


def fast_transform(signal):
    pass