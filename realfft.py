from random import random

from numpy import real, imag
from numpy.core.umath import pi
from numpy.fft import fft
from numpy.ma import array, exp, zeros


def real_fft(data):
    n = len(data)
    result = zeros(n, dtype=complex)
    re_x = [data[2*i] for i in range(n // 2)]
    im_x = [data[2*i+1] for i in range(n // 2)]
    x = [re_x[i] + 1j*im_x[i] for i in range(n // 2)]
    y = fft(x)
    c = real(y)
    d = imag(y)
    e = zeros(n // 2, dtype=float)
    f = zeros(n // 2, dtype=float)
    g = zeros(n // 2, dtype=float)
    h = zeros(n // 2, dtype=float)
    s = zeros(n // 2, dtype=complex)
    t = zeros(n // 2, dtype=complex)
    s[0] = c[0]
    t[0] = d[0]
    for i in range(1, n//2):
        e[i] = (c[i] + c[n//2 - i])/2
        f[i] = (d[i] - d[n//2 - i])/2
        h[i] = -(c[i] - c[n//2 - i])/2
        g[i] = (d[i] + d[n//2 - i]) / 2
        s[i] = e[i] + 1j * f[i]
        t[i] = g[i] + 1j * h[i]
    w = array([exp(-1j*2*pi*i/n) for i in range(n//2)])
    r = t * w
    for i in range(n//2):
        result[i] = s[i] + r[i]
        result[i+n//2] = s[i] - r[i]
    return result


#test
x = [random() for i in range(128)]
y = real_fft(x)
print(max(y-fft(x)))