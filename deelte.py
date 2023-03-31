import numpy as np
import math

create_index_map = lambda signal, w, h: np.array([[ [i] * signal.shape[2] for i in range(signal.shape[w]) ] for j in range(signal.shape[h]) ])

f = np.array([
	[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
	[[1, 1, 1], [2, 2, 2], [3, 3, 3]]
])

index_map_M = create_index_map(f, 1, 0)
index_map_N = create_index_map(f, 0, 1)

print("--------------------------------")
print(index_map_N.transpose(1,0,2))
print("--------------------------------")


X_k = lambda f, index, k: np.sum(f * math.e**( (-1j * 2 * math.pi * k * index) / f.shape[1]), axis=1)
X_t = lambda f, index, k: np.sum(np.multiply(f, index * k), axis=1)

print("--------------------------------")
a = np.array([ X_t(f, index_map_M, k) for k in range(f.shape[1])]).transpose(1,0,2)
print(a)
print("--------------------------------")
print(np.array([ X_t(a, index_map_N.transpose(1,0,2), k) for k in range(a.shape[1])]).transpose(1,0,2))

#[
#	[[ 0  0  0] [ 0  0  0]]
#	[[ 8  8  8] [ 8  8  8]]
#   [[16 16 16] [16 16 16]]
#]

#[
#	[[0 0 0]  [1 1 1]]
# 	[[0 0 0]  [1 1 1]]
# 	[[0 0 0]  [1 1 1]]
#]
# Now:

#[
#	[[0 0 0] [0 0 0]  [0 0 0]]
# 	[[1 1 1] [1 1 1]  [1 1 1]]
#]

#[
#	[[ 0  0  0] [ 8  8  8] [16 16 16]]
# 	[[ 0  0  0] [ 8  8  8] [16 16 16]]
#]

[
	[[ 0  0  0]  [ 0  0  0]]
 	[[ 0  0  0]  [24 24 24]]
	[[ 0  0  0]  [48 48 48]]
]
[
	[[ 0  0  0] [ 0  0  0]  [ 0  0  0]]
 	[[ 0  0  0] [24 24 24]  [48 48 48]]
]