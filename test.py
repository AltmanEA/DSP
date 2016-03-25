import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft
from eafft import fft64s, fft4, fft16, bitRevers
from myfft import crfft

# test bitrevers
print(bitRevers(np.array([x for x in range(0, 16)])))

# test fft
# x = np.array([x for x in range(0, 16)])
# y = fft16(x)
# y1 = fft(x)
# print(max(y-y1))


# x = np.array([x for x in range(0, 64)], dtype=complex)
# s = np.array([1 for x in range(0, 64)])
# print(fft(x[0:16]))
# print(fft(x[16:32]))
# y = fft64s(x, s)
# y1 = fft(x)
# print(max(y-y1))


# test crfft
# n = 32
# x = np.array([np.sin(2*np.pi*x/n) for x in range(0, n)])
# x = np.array([x for x in range(0, n)])
# plt.plot(x)
# plt.show()
# y = crfft(x)
# print(max(y-np.fft.fft(x)))
