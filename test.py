# check some things
from numpy import array, exp, pi, zeros
from numpy.fft import fft, ifft


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
        x[i*4:(i+1)*4] = fft(x[i*4:(i+1)*4])

    w = array([[exp(-2j*pi*i*j/16) for i in range(4)] for j in range(4)])
    x = x * w.flatten()

    for i in range(4):
        x[i:16:4] = fft(x[i:16:4])

    return x


def fft128_4_32(x):
    for i in range(32):
        x[i:128:32] = fft(x[i:128:32])

    w = array([[exp(-2j*pi*i*j/128) for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    for i in range(4):
        x[i*32:(i+1)*32] = fft(x[i*32:(i+1)*32])

    x = x.reshape((4, 32)).transpose().flatten()
    return x


def fft128(x):
    w = array([[exp(-2j*pi*i*j/32) for i in range(4)] for j in range(8)])
    for i in range(4):
        tmp = x[i:128:4]

        for j in range(4):
            tmp[j:32:4] = fft(tmp[j:32:4])
        tmp = tmp*w.flatten()
        for j in range(8):
            tmp[j*4:(j+1)*4] = fft(tmp[j*4:(j+1)*4])

        x[i:128:4] = tmp.reshape((8, 4)).transpose().flatten()

    w = array([[exp(-2j*pi*i*j/128) for i in range(4)] for j in range(32)])
    x = x*w.flatten()

    for i in range(32):
        x[i*4:(i+1)*4] = fft(x[i*4:(i+1)*4])

    x = x.reshape((32, 4)).transpose().flatten()
    return x


def fft128_back(x):
    x = x.reshape((32, 4)).transpose().flatten()

    for i in range(4):
        x[i*32:(i+1)*32] = fft(x[i*32:(i+1)*32])

    w = array([[exp(-2j*pi*i*j/128) for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    for i in range(32):
        x[i:128:32] = fft(x[i:128:32])

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
test_func(fft128, 128)