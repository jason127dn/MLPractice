import numpy as np
np1 = np.array([1, 2, 3])
np2 = np.array([3, 4, 5])

print(np1 + np2)
print(np1.ndim, np1.shape, np1.dtype)

np3 = np.array([1, 2, 3, 4, 5, 6])
np3 = np3.reshape([2, 3])

print(np3.ndim, np3.shape, np3.dtype)

