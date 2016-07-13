from numpy import array, zeros, pi, exp, cos, hstack
from numpy.fft import fft, ifft
from classic_fft import split_radix_butt, ncpsrfft, shift_right


#
#       Algorithm 1
#

# scaled (s) fft for 64 point
def fft64s(x, s):
    for k in range(4):
        z = array(x[k:64:4])
        for i in range(4):
            z[i:16:4] = fft(z[i:16:4])
        w = array([[exp(-2j*pi*i*j/16)/even_koef(j) for i in range(4)] for j in range(4)])
        z = z * w.flatten()

        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft(z[i*4:i*4+4])
        x[k:64:4] = y

    w = array([[exp(-2j*pi*i*j/64)*even_koef(j)/s[j] for i in range(4)] for j in range(16)])
    x = x * w.flatten()
    y = zeros(64, dtype=complex)
    for i in range(16):
        y[i:64:16] = fft(x[i*4:i*4+4])
    return y


def even_koef(elem):
    if elem % 2 == 0:
        return 1
    return cos(pi / 8)


# new fft 256 point
def fft256_a1(x):
    n = 256
    s1 = array([exp(1j*2*pi*i/n) for i in range(16)]) / cos(pi/8)
    s2 = array([exp(-1j*2*pi*i/n) for i in range(16)]) / cos(pi/8)
    s1[0:64:8] = 1
    s2[0:64:8] = 1
    w1 = array([exp(-1j*2*pi*i/n)*s1[i%16] for i in range(64)])
    w2 = array([exp(1j*2*pi*i/n)*s2[i%16] for i in range(64)])
    u = ncpsrfft(x[0:n:2])
    z = fft64s(x[1:n:4], s1)*w1
    z1 = fft64s(shift_right(x[3:n:4]), s2)*w2
    return split_radix_butt(u, z, z1, n)


#
#       Algorithm 2
#

def fft16_m1(xl):
    for i in range(4):
        xl[i:16:4] = fft(xl[i:16:4])

    wl = array([[exp(-2j * pi * i * j / 16) for i in range(4)] for j in range(4)])
    xl = xl * wl.flatten()

    for i in range(4):
        t0 = xl[i * 4] + xl[i * 4 + 2]
        t1 = xl[i * 4] - xl[i * 4 + 2]
        t2 = xl[i * 4 + 1] + xl[i * 4 + 3]
        t3 = 1j * (xl[i * 4 + 1] - xl[i * 4 + 3])
        xl[i * 4] = t0 + t2
        xl[i * 4 + 1] = t1 - t3
        xl[i * 4 + 2] = t0 - t2
        xl[i * 4 + 3] = t1 + t3

    xl = xl.reshape((4, 4)).transpose().flatten()
    return xl


def fft256_a2_1(xl):
    for i in range(16):
        xl[i:256:16] = fft(xl[i:256:16])

    global before_x
    before_x = xl.copy()

    wl = array([[exp(-2j*pi*i*j/256) for i in range(16)] for j in range(16)])
    xl = xl * wl.flatten()

    global ww
    ww = wl.copy()
    global after_x
    after_x = xl.copy()

    for i in range(16):
        xl[i * 16:(i + 1) * 16] = fft(xl[i * 16:(i + 1) * 16])

    xl = xl.reshape((16, 16)).transpose().flatten()
    return xl


def fft256_half(xl):
    for i in range(8):
        xl[i:128:8] = fft(xl[i:128:8])

    wl = array([[exp(-2j*pi*(2*i+1)*j/256) for i in range(8)] for j in range(16)])
    xl = xl * wl.flatten()

    for i in range(16):
        tmp = xl[i*8:(i+1)*8]
        tmp[0:8:2] = fft(tmp[0:8:2])
        tmp[1:8:2] = fft(tmp[1:8:2])
        wl = array([[exp(-2j*pi*(2*i+1)*j/16) for i in range(2)] for j in range(4)])
        tmp = tmp * wl.flatten()
        for j in range(4):
            t1 = tmp[2*j]
            t2 = tmp[2*j+1]
            tmp[2*j] = t1 + t2
            tmp[2*j+1] = -1j*(t1 - t2)
        xl[i*8:(i+1)*8] = tmp

    xl = xl.reshape((8, 16)).transpose().flatten()
    xl = xl.reshape((2, 64)).transpose().flatten()

    # temp solution
    xtmp = zeros(128, dtype=complex)
    xtmp[0:16] = xl[0:16]
    xtmp[16:32] = xl[32:48]
    xtmp[32:48] = xl[64:80]
    xtmp[48:64] = xl[96:112]

    xtmp[64:80] = xl[16:32]
    xtmp[80:96] = xl[48:64]
    xtmp[96:112] = xl[80:96]
    xtmp[112:128] = xl[112:128]

    return xtmp


def fft256_a2_2(xl):
    n = 256
    wl = array([exp(-1j*2*pi*i/n) for i in range(n//2)])
    t0 = fft(xl[0:n:2])
    t1 = fft256_half(xl[1:n:2])
    # y1 = fft(x[1:n:2]) * wl
    return hstack((t0 + t1, t0 - t1))


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
test_func(fft256_a2_2, 256)

# half test
# x = array([x for x in range(256)], dtype=complex)
# w = array([exp(-1j*2*pi*i/256) for i in range(128)])
# h1 = fft(x[0:256:2])
# h2 = fft(x[1:256:2])
# h3 = h2 * w.flatten()
# # print(max(abs(hstack((h1 + h3, h1 - h3))-fft(x))))
#
# after_x = 0
# before_x = 0
# ww = 0
#
# s2 = ifft(h2)
# y0 = fft(s2) * w.flatten()
# y2 = fft256_a2_1(x)
# y1 = fft256_half(s2)
# print(y1)
# print(y0)
# print(max(abs(y1 - y0)))
# print(y1-y0)