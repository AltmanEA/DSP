from numpy import array, zeros, matrix, pi, exp, cos
from classic_fft import fft4, crfft, shift_right, split_radix_butt, ncpsrfft


# scaled (s) fft for 64 point
def fft64s(x, s):
    for k in range(4):
        z = array(x[k:64:4])
        for i in range(4):
            z[i:16:4] = fft4(z[i:16:4])
        w = array([[exp(-2j*pi*i*j/16)/even_koef(j) for i in range(4)] for j in range(4)])
        z = z * w.flatten()

        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft4(z[i * 4:i * 4 + 4])
        x[k:64:4] = y

    w = array([[exp(-2j*pi*i*j/64)*even_koef(j)/s[j] for i in range(4)] for j in range(16)])
    x = x * w.flatten()
    y = zeros(64, dtype=complex)
    for i in range(16):
        y[i:64:16] = fft4(x[i * 4:i * 4 + 4])
    return y


def even_koef(elem):
    if elem % 2 == 0:
        return 1
    return cos(pi/8)


def fft256(x):
    n = 256
    s1 = array([exp(1j*2*pi*i/n) for i in range(16)])/cos(pi/8)
    s2 = array([exp(-1j*2*pi*i/n) for i in range(16)])/cos(pi/8)
    s1[0:64:8] = 1
    s2[0:64:8] = 1
    w1 = array([exp(-1j*2*pi*i/n)*s1[i%16] for i in range(64)])
    w2 = array([exp(1j*2*pi*i/n)*s2[i%16] for i in range(64)])
    u = ncpsrfft(x[0:n:2])
    z = fft64s(x[1:n:4], s1) * w1
    z1 = fft64s(shift_right(x[3:n:4]), s2) * w2
    return split_radix_butt(u, z, z1, n)

