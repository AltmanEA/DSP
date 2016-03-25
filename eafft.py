from numpy import array, zeros, matrix, pi, exp, log2


# fft 4 point
def fft4(x):
    y = zeros(4, dtype=complex)
    t0 = x[0] + x[2]
    t1 = x[0] - x[2]
    t2 = x[1] + x[3]
    t3 = 1j * (x[1] - x[3])
    y[0] = t0 + t2
    y[1] = t1 - t3
    y[2] = t0 - t2
    y[3] = t1 + t3
    return y


def fft16(x):
    for i in range(0, 4):
        x[i * 4:i * 4 + 4] = fft4(x[i * 4:i * 4 + 4])

    w = array(exp(-2j * pi * matrix(
        [[i * j for i in range(0, 4)] for j in range(0, 4)]) / 16)).flatten()
    x = x * w

    for j in range(0, 4):
        x[j:16:4] = fft4(x[j:16:4])

    return x


# scaled (s) fft for 64 point
def fft64s(x, s):
    # stage 1, 16 fft4
    for i in range(0, 16):
        x[i * 4:i * 4 + 4] = fft4(x[i * 4:i * 4 + 4])

    # stage 2, 4 fft16
    w = array(exp(-2j * pi * matrix(
        [[i * j for i in range(0, 4)] for j in range(0, 4)]) / 16)).flatten()
    for i in range(0, 4):
        x[i * 16:(i + 1) * 16] = x[i * 16:(i + 1) * 16] * w
        for j in range(0, 4):
            x[i * 16 + j:(i + 1) * 16:4] = fft4(x[i * 16 + j:(i + 1) * 16:4])

    # stage 3, 1 fft64
    w = array(exp(-2j * pi * matrix(
        [[i * j for i in range(0, 16)] for j in range(0, 4)]) / 64).reshape((1, 64)))
    x = (x * w).reshape(64)
    for i in range(0, 16):
        x[i:64:16] = fft4(x[i:64:16])

    return x


def bitRevers(x):
    n = x.size
    y = array(x)
    bits = int(log2(n))
    for i in range(n):
        y[i] = x[bitReverseIndex(i, bits)]
    return y

def bitReverseIndex(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result



