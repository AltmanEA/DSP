import numpy as np


def crfft(x):
    n = len(x)
    if n == 1:
        return x
    if n == 2:
        return [x[0]+x[1], x[0]-x[1]]

    w1 = np.array([np.exp(-2j*np.pi*i/n) for i in range(0, n//4)])
    w2 = np.array([np.exp(2j*np.pi*i/n) for i in range(0, n//4)])

    u = crfft(x[0:n:2])
    z1 = crfft(x[1:n:4]) * w1
    z2 = crfft(np.roll(x[3:n:4], 1)) * w2

    y = np.zeros(n, dtype=complex)
    y[0:n//4] = u[0:n//4] + z1[0:n//4] + z2[0:n//4]
    y[n//2:3*n//4] = u[0:n//4] - z1[0:n//4] - z2[0:n//4]
    y[n//4:n//2] = u[n//4:n//2] - 1j*(z1[0:n//4] - z2[0:n//4])
    y[3*n//4:n] = u[n//4:n//2] + 1j*(z1[0:n//4] - z2[0:n//4])

    return y
