from numpy import array, zeros
from numpy.fft import fft, ifft

n = 8
x = array([2*i+3 for i in range(n)], dtype=complex)
xi = zeros(x.shape, dtype=complex)
xi[0] = x[0]
xi[1:n] = x[n-1:0:-1]
y = fft(x)
yi = ifft(xi)*8

print(y- yi)
