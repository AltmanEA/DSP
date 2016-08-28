from numpy import array, zeros, pi, exp, cos, hstack
from numpy.fft import fft, ifft
from classic_fft import split_radix_butt, ncpsrfft, shift_right
from tools import flops4Muls, bit_revers, wf

#
#       Algorithm 1
#   SR + fft64s with carry
#

# scaled (s) fft for 64 point



def fft64s(x, s):
    global mul_count
    for k in range(4):
        z = array(x[k:64:4])
        for i in range(4):
            z[i:16:4] = fft(z[i:16:4])
        w = array([[exp(-2j*pi*i*j/16)/even_koef(j) for i in range(4)] for j in range(4)])
        z = z * w.flatten()
        mul_count += flops4Muls(w)

        y = zeros(16, dtype=complex)
        for i in range(4):
            y[i:16:4] = fft(z[i*4:i*4+4])
        x[k:64:4] = y

    w = array([[exp(-2j*pi*i*j/64)*even_koef(j)/s[j] for i in range(4)] for j in range(16)])
    x = x * w.flatten()
    mul_count += flops4Muls(w)
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
    global mul_count
    n = 256
    s1 = array([exp(1j*2*pi*i/n) for i in range(16)]) / cos(pi/8)
    s2 = array([exp(-1j*2*pi*i/n) for i in range(16)]) / cos(pi/8)
    s1[0:64:8] = 1
    s2[0:64:8] = 1
    w1 = array([exp(-1j*2*pi*i/n)*s1[i%16] for i in range(64)])
    w2 = array([exp(1j*2*pi*i/n)*s2[i%16] for i in range(64)])
    u = ncpsrfft(x[0:n:2])
    mul_count += 1000
    z = fft64s(x[1:n:4], s1)*w1
    mul_count += flops4Muls(w1)
    z1 = fft64s(shift_right(x[3:n:4]), s2)*w2
    mul_count += flops4Muls(w2)
    return split_radix_butt(u, z, z1, n)


#
#       Algorithm 2
#   combo -- SR with radix16
#

def fft16_a2(xl):
    global mul_count
    for i in range(4):
        xl[i:16:4] = fft(xl[i:16:4])

    wl = array([[exp(-2j * pi * i * j / 16) for i in range(4)] for j in range(4)])
    xl = xl * wl.flatten()
    mul_count += flops4Muls(wl)

    for i in range(4):
        xl[i*4:i*4+4] = fft(xl[i*4:i*4+4])

    xl = xl.reshape((4, 4)).transpose().flatten()
    return xl


def fft256_half_a2(xl):
    global mul_count
    for i in range(8):
        xl[i:128:8] = fft16_a2(xl[i:128:8])

    wl = array([[exp(-2j*pi*(2*i+1)*j/256) for i in range(8)] for j in range(16)])
    xl = xl * wl.flatten()
    mul_count += flops4Muls(wl)

    for i in range(16):
        tmp = xl[i*8:(i+1)*8]
        tmp[0:8:2] = fft(tmp[0:8:2])
        tmp[1:8:2] = fft(tmp[1:8:2])
        wl = array([[exp(-2j*pi*(2*i+1)*j/16) for i in range(2)] for j in range(4)])
        tmp = tmp * wl.flatten()
        mul_count += flops4Muls(wl)
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


def fft256_a2(xl):
    global mul_count
    n = 256
    t0 = fft(xl[0:n:2])
    mul_count += 1032#1000
    t1 = fft256_half_a2(xl[1:n:2])
    return hstack((t0 + t1, t0 - t1))


#
#       Algorithm 3
#   combo SR with radix16 and carry
#

def fft16_a3(xl):
    global mul_count
    for i in range(4):
        xl[i:16:4] = fft(xl[i:16:4])

    wl = array([[exp(-2j * pi * i * j / 16)/even_koef(j) for i in range(4)] for j in range(4)])
    xl = xl * wl.flatten()
    mul_count += flops4Muls(wl)

    for i in range(4):
        xl[i*4:i*4+4] = fft(xl[i*4:i*4+4])

    xl = xl.reshape((4, 4)).transpose().flatten()
    return xl


def fft256_half_a3(xl):
    global mul_count
    for i in range(8):
        xl[i:128:8] = fft16_a3(xl[i:128:8])

    curry_up = zeros(16)+1
    curry_up[1:16:2] = cos(pi/8)
    curry_down = zeros(16) + 1
    curry_down[1:16] = cos(pi / 8)
    wl = array([[exp(-2j*pi*(2*i+1)*j/256)*curry_up[j]*curry_down[j] for i in range(8)] for j in range(16)])
    xl = xl * wl.flatten()
    mul_count += flops4Muls(wl)

    for i in range(16):
        tmp = xl[i*8:(i+1)*8 ]
        tmp[0:8:2] = fft(tmp[0:8:2])
        tmp[1:8:2] = fft(tmp[1:8:2])
        wl = array([[exp(-2j*pi*(2*k+1)*j/16) for k in range(2)] for j in range(4)])
        if i != 0:
            wl /= cos(pi/8)
        tmp = tmp * wl.flatten()
        mul_count += flops4Muls(wl)
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


def fft256_a3(xl):
    global mul_count
    n = 256
    t0 = fft(xl[0:n:2])
    mul_count += 1000
    t1 = fft256_half_a3(xl[1:n:2])
    return hstack((t0 + t1, t0 - t1))


#
#       Algorithm 4
#   radix16 and carry
#       in progress
#
def fft256_a4(x):
    global mul_count
    w = array([[exp(-2j*pi*i*j/256) for i in range(16)] for j in range(16)])
    mul_count += flops4Muls(w)

    for i in range(16):
        x[i:256:16] = fft16_a4_1(x[i:256:16])

    x *= w.flatten()

    for i in range(16):
        x[i*16:(i+1)*16] = fft16_a4_2(x[i*16:(i+1)*16])

    return x.reshape((16, 16)).transpose().flatten()


def get_w_fft16_r2():
    w1 = zeros(16, dtype=complex)+1
    w2 = zeros(16, dtype=complex) + 1
    w3 = zeros(16, dtype=complex) + 1
    w1[3:16:4] = -1j
    for i in range(4):
        w2[i+4] = wf(i, 8)
        w2[i+12] = wf(i, 8)
    for i in range(8):
        w3[i+8] = wf(i, 16)
    return w1, w2, w3


def fft16_a4_1(x):
    global mul_count
    tmp = bit_revers(x)
    w1, w2, w3 = get_w_fft16_r2()
    mul_count += flops4Muls(w1)
    mul_count += flops4Muls(w2)
    mul_count += flops4Muls(w3)

    for i in range(8):
        tmp[i*2:i*2+2] = fft(tmp[i*2:i*2+2])
    tmp *= w1

    for i in range(4):
        for j in range(2):
            tmp[i*4+j:i*4+j+3:2] = fft(tmp[i*4+j:i*4+j+3:2])
    tmp *= w2

    for i in range(2):
        for j in range(4):
            tmp[i*8+j:i*8+j+5:4] = fft(tmp[i*8+j:i*8+j+5:4])
    tmp *= w3

    for j in range(8):
        tmp[j:j+9:8] = fft(tmp[j:j+9:8])

    return tmp


def fft16_a4_2(x):
    global mul_count
    tmp = x
    w3, w2, w1 = get_w_fft16_r2()
    mul_count += flops4Muls(w1)
    mul_count += flops4Muls(w2)
    mul_count += flops4Muls(w3)

    for j in range(8):
        tmp[j:j+9:8] = fft(tmp[j:j+9:8])
    tmp *= w1

    for i in range(2):
        for j in range(4):
            tmp[i*8+j:i*8+j+5:4] = fft(tmp[i*8+j:i*8+j+5:4])
    tmp *= w2

    for i in range(4):
        for j in range(2):
            tmp[i*4+j:i*4+j+3:2] = fft(tmp[i*4+j:i*4+j+3:2])
    tmp *= w3

    for i in range(8):
        tmp[i*2:i*2+2] = fft(tmp[i*2:i*2+2])

    return bit_revers(tmp)

#
#       TEST
#

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
mul_count = 0
test_func(fft16_a4_1, 16)
print ("for muls flops = ", mul_count)
print("flops = ", mul_count+4096)

