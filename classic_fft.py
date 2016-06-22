from numpy import array, zeros, matrix, pi, exp, log2, cos, sin, fmax, sqrt, hstack


#
# recursive function
#

def fftr2(x):
    n = x.size
    if n == 1:
        return x
    w = array([exp(-1j*2*pi*i/n) for i in range(n//2)])
    y0 = fftr2(x[0:n:2])
    y1 = fftr2(x[1:n:2])*w
    return hstack((y0 + y1, y0 - y1))


#
# radix-4
#

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
    w = array([[exp(-2j*pi*i*j/16) for i in range(4)] for j in range(4)])
    x = x * w.flatten()
    for i in range(4):
        y[i:16:4] = fft4(x[i * 4:i * 4 + 4])
    return y


def fft64(x):
    for k in range(4):
        z = array(x[k:64:4])
        for i in range(4):
            z[i:16:4] = fft4(z[i:16:4])
        w = array([[exp(-2j*pi*i*j/16) for i in range(4)] for j in range(4)])
        z = z * w.flatten()
        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft4(z[i * 4:i * 4 + 4])
        x[k:64:4] = y
    w = array([[exp(-2j*pi*i*j/64) for i in range(4)] for j in range(16)])
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


#
# Conjugate-pair Split-Radix
#

def shift_right(x):
    n = x.size
    y = zeros(n, dtype=complex)
    y[0] = x[n-1]
    y[1:n] = x[0:n-1]
    return y


def split_radix_butt(u, z, z1, n):
    y = zeros(n, dtype=complex)
    for i in range(int(n/4)):
        y[i]       = (u[i] + (z[i] + z1[i]))
        y[i+n/2]   = (u[i] - (z[i] + z1[i]))
        y[i+n/4]   = (u[i+n/4] - 1j*(z[i] - z1[i]))
        y[i+3*n/4] = (u[i+n/4] + 1j*(z[i] - z1[i]))
    return y


def crfft(x):
    n = x.size
    if n == 1:
        return x
    if n == 2:
        return array([x[0] + x[1], x[0] - x[1]])
    w1 = array([exp(-1j*2*pi*i/n) for i in range(n//4)])
    w2 = array([exp(1j*2*pi*i/n) for i in range(n//4)])
    u = crfft(x[0:n:2])
    z = crfft(x[1:n:4]) * w1
    z1 = crfft(shift_right(x[3:n:4])) * w2
    return split_radix_butt(u, z, z1, n)


#
# New Conjugate-pair Split-Radix
#

def s_gen(n):
    if n < 5:
        return zeros(n) + 1
    result = zeros(n, dtype=float)
    q = int(n / 4)
    tmp = array(range(q))
    result[0:q] = fmax(cos(2 * pi * tmp / n), sin(2 * pi * tmp / n))
    result[q:2 * q] = result[0:q]
    result[2 * q:4 * q] = result[0:2 * q]
    return result


def get_sizes(x):
    n = x.size
    n1 = int(n/4)
    n2 = int(n/2)
    n3 = int(3*n/4)
    return n, n1, n2, n3


def get_w(n, scale):
    tmp = array(range(n//4))
    w1 = exp(-1j*2*pi*tmp/n)*scale
    w2 = exp(1j*2*pi*tmp/n)*scale
    return w1, w2


def get_scales(n):
    s = s_gen(n)
    s1_4 = s_gen(n//4)
    s2 = s_gen(2 * n)
    s4 = s_gen(4 * n)
    t = s1_4 / s[0:n//4]
    return t, s, s1_4, s2, s4


def ncpsrffts4(x):
    n, n1, n2, n3 = get_sizes(x)
    if n == 1:
        return x
    if n == 2:
        return array([x[0] + x[1], (x[0] - x[1])*sqrt(2)])
    t, s, s1_4, s2, s4 = get_scales(n)
    w1, w2 = get_w(n, t)

    u = ncpsrffts2(x[0:n:2])
    z = ncpsrffts(x[1:n:4]) * w1
    z1 = ncpsrffts(shift_right(x[3:n:4])) * w2

    y = zeros(n, dtype=complex)
    for i in range(n//4):
        y[i]       = (u[i] + (z[i] + z1[i]))*s[i]/s4[i]
        y[i+n/2]   = (u[i] - (z[i] + z1[i]))*s[i]/s4[i+n/2]
        y[i+n/4]   = (u[i+n/4] - 1j*(z[i] - z1[i]))*s[i]/s4[i+n/4]
        y[i+3*n/4] = (u[i+n/4] + 1j*(z[i] - z1[i]))*s[i]/s4[i+3*n/4]
    return y


def ncpsrffts2(x):
    n, n1, n2, n3 = get_sizes(x)
    if n == 1:
        return x
    if n == 2:
        return array([x[0] + x[1], x[0] - x[1]])
    t, s, s1_4, s2, s4 = get_scales(n)
    w1, w2 = get_w(n, t)
    u = ncpsrffts4(x[0:n:2])
    z = ncpsrffts(x[1:n:4]) * w1
    z1 = ncpsrffts(shift_right(x[3:n:4])) * w2

    y = zeros(n, dtype=complex)
    for i in range(n//4):
        y[i]       = u[i] + (z[i] + z1[i])*s[i]/s2[i]
        y[i+n/2]   = u[i] - (z[i] + z1[i])*s[i]/s2[i]
        y[i+n/4]   = u[i+n/4] - 1j*(z[i] - z1[i])*s[i]/s2[i+n/4]
        y[i+3*n/4] = u[i+n/4] + 1j*(z[i] - z1[i])*s[i]/s2[i+n/4]
    return y


def ncpsrffts(x):
    n, n1, n2, n3 = get_sizes(x)
    if n == 1:
        return x
    if n == 2:
        return array([x[0] + x[1], x[0] - x[1]])
    t, s, s1_4, s2, s4 = get_scales(n)
    w1, w2 = get_w(n, t)
    u = ncpsrffts2(x[0:n:2])
    z = ncpsrffts(x[1:n:4]) * w1
    z1 = ncpsrffts(shift_right(x[3:n:4])) * w2
    return split_radix_butt(u, z, z1, n)


def ncpsrfft(x):
    n, n1, n2, n3 = get_sizes(x)
    if n < 64:
        return crfft(x)
    t, s, s1_4, s2, s4 = get_scales(n)
    w1, w2 = get_w(n, s1_4)
    u = ncpsrfft(x[0:n:2])
    z = ncpsrffts(x[1:n:4]) * w1
    z1 = ncpsrffts(shift_right(x[3:n:4])) * w2
    return split_radix_butt(u, z, z1, n)
