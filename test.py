# check some things
from numpy import array, exp, pi, zeros, dot
from numpy.fft import fft, ifft
from numpy.linalg import linalg


def fft16(x):
    for i in range(4):
        x[i:16:4] = fft(x[i:16:4])

    w = array([[exp(-2j*pi*i*j/16) for i in range(4)] for j in range(4)])
    x = x * w.flatten()

    for i in range(4):
        x[i*4:(i+1)*4] = fft(x[i*4:(i+1)*4])

    x = x.reshape((4, 4)).transpose().flatten()
    return x


def fft16_back(x):
    x = x.reshape((4, 4)).transpose().flatten()
    for i in range(4):
        x[i*4:(i+1)*4] = ifft(x[i*4:(i+1)*4])*4

    w = array([[exp(2j*pi*i*j/16) for i in range(4)] for j in range(4)])
    x = x * w.flatten()

    for i in range(4):
        x[i:16:4] = ifft(x[i:16:4])*4

    return x


def fft128(x):
    for i in range(32):
        x[i:128:32] = fft(x[i:128:32])

    w = array([[exp(-2j*pi*i*j/128) for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    for i in range(4):
        x[i*32:(i+1)*32] = fft(x[i*32:(i+1)*32])

    x = x.reshape((4, 32)).transpose().flatten()
    return x


def fft128_back(x):
    x = x.reshape((4, 32)).transpose().flatten()

    for i in range(4):
        x[i*32:(i+1)*32] = ifft(x[i*32:(i+1)*32])*32

    w = array([[exp(-2j*pi*i*j/128) for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    for i in range(32):
        x[i:128:32] = ifft(x[i:128:32])*4

    return x


# test function
def test_func(func, size, full_list=False):
    x = array([x for x in range(size)], dtype=complex)
    python_fft = fft(x)
    y = func(x)
    if full_list:
        print(python_fft)
        print(y)
    print(max(abs(y-python_fft)))

# test
# test_func(fft16_back, 16, True)

w  = array([[exp(-2j*pi*i*j/8) for i in range(8)] for j in range(8)])
w1 = linalg.inv(w)
x = dot(w1, w)
print(w)