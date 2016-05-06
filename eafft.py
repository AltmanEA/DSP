from numpy import array, zeros, matrix, pi, exp, log2, dtype


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
    y = zeros(16, dtype=complex)
    for i in range(4):
        x[i:16:4] = fft4(x[i:16:4])

    w = array(exp(-2j * pi * matrix(
        [[i * j for i in range(4)] for j in range(4)]) / 16))
    x = x * w.flatten()

    for i in range(4):
        y[i:16:4] = fft4(x[i*4:i*4+4])

    return y


def fft64(x):

    for k in range(4):
        z = array(x[k:64:4])

        for i in range(4):
            z[i:16:4] = fft4(z[i:16:4])

        w = array(exp(-2j * pi * matrix(
            [[i * j for i in range(4)] for j in range(4)]) / 16))
        z = z * w.flatten()

        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft4(z[i * 4:i * 4 + 4])

        x[k:64:4] = y

    w = array(exp(-2j * pi * matrix(
            [[i * j for i in range(4)] for j in range(16)]) / 64))
    x = x * w.flatten()

    y = zeros(64, dtype=complex)
    for i in range(16):
        y[i:64:16] = fft4(x[i*4:i*4+4])

    return y


# scaled (s) fft for 64 point
def fft64s(x, s):
    for k in range(4):
        z = array(x[k:64:4])

        for i in range(4):
            z[i:16:4] = fft4(z[i:16:4])

        w = array(exp(-2j * pi * matrix(
            [[i * j for i in range(4)] for j in range(4)]) / 16))
        z = z * w.flatten()

        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft4(z[i * 4:i * 4 + 4])

        x[k:64:4] = y

    w = array(exp(-2j * pi * matrix(
        [[i * j for i in range(4)] for j in range(16)]) / 64))
    x = x * w.flatten()

    y = zeros(64, dtype=complex)
    for i in range(16):
        y[i:64:16] = fft4(x[i * 4:i * 4 + 4])

    return y


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



