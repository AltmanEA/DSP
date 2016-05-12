import numpy as np

from numpy.fft import fft
from classic_fft import fft4, fft16, fft64, s_gen, crfft, ncpsrfft

# test fft
from eafft import fft64s, fft256

x = np.array([x for x in range(256)], dtype=complex)
y1 = fft(x)
y = fft256(x)

print(y1)
print(y)
print(max(abs(y-y1)))



