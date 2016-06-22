from numpy import zeros, array, exp, pi, log2
from numpy.fft import fft


def fft4_normal(x):
    t0 = x[0] + x[2]
    t1 = x[0] - x[2]
    t2 = x[1] + x[3]
    t3 = 1j * (x[1] - x[3])
    x[0] = t0 + t2
    x[1] = t1 - t3
    x[2] = t0 - t2
    x[3] = t1 + t3
    return x


def bit_revers(x):
    n = x.size
    y = array(x)
    bits = int(log2(n))
    for i in range(n):
        y[i] = x[bit_reverse_index(i, bits)]
    return y


def bit_reverse_index(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1:
            result |= 1 << (n - 1 - i)
    return result


def fft_r4(x, fft4):
    n = x.size
    if n == 4:
        return fft4(x)
    w = array([[exp(-2j*pi*i*j/n) for i in range(4)] for j in range(n//4)])

    for i in range(4):
        fft_r4(x[i:n:4], fft4)
    x = x*w.flatten()
    for i in range(n//4):
        fft4(x[i*4:i*4+4])

    return x


gx = array([x for x in range(16)], dtype=complex)
y1 = fft(gx)
y = fft_r4(gx, fft4_normal)
print(y1)
print(y)
print(max(abs(y-y1)))
