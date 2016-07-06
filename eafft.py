# new fft algorithm
from numpy import array, zeros, pi, exp, cos, hstack
from numpy.fft import fft
from classic_fft import fft4, shift_right, split_radix_butt, ncpsrfft


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


# new fft 256 point
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


# scaled (s) fft for 128 point
def fft1281s(x, s):
    for i in range(32):
        x[i:128:32] = fft(x[i:128:32])

    cor = [exp(-2j*pi*i/32) * exp(-2j*pi*i/32).real for i in range(8)]
    cor[0] = 1
    cor[4] = 1

    w = array([[exp(-2j*pi*i*j/128)*cor[i%8]/s[i] for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    w = array([[exp(-2j*pi*i*j/32)/cor[i] for i in range(8)] for j in range(4)])

    for i in range(4):
        tmp = x[i*16:(i+1)*16]

        for j in range(8):
            tmp[j:16:8] = fft(tmp[j:16:8])
        tmp = tmp*w.flatten()
        for j in range(4):
            tmp[j*8:(j+1)*8] = fft(tmp[j*8:(j+1)*8])

        tmp = tmp.reshape((4, 8)).transpose().flatten()
        x[i*16:(i+1)*16] = tmp

    x = x.reshape((4, 32)).transpose().flatten()
    return x


# scaled (s) fft for 128 point
def fft128s(x, s):
    for i in range(32):
        x[i:128:32] = fft(x[i:128:32])

    w = array([[exp(-2j*pi*i*j/128) for i in range(32)] for j in range(4)])
    x = x*w.flatten()

    w = array([[exp(-2j*pi*i*j/32) for i in range(8)] for j in range(4)])

    for i in range(4):
        tmp = x[i*16:(i+1)*16]

        for j in range(8):
            tmp[j:16:8] = fft(tmp[j:16:8])
        tmp = tmp*w.flatten()
        for j in range(4):
            tmp[j*8:(j+1)*8] = fft(tmp[j*8:(j+1)*8])

        tmp = tmp.reshape((4, 8)).transpose().flatten()
        x[i*16:(i+1)*16] = tmp

    x = x.reshape((4, 32)).transpose().flatten()
    return x


# new fft 512 point
def fft512(x):
    n = 512
    s1 = array([exp(1j*2*pi*i/n) for i in range(32)])/cos(pi/8)
    s2 = array([exp(-1j*2*pi*i/n) for i in range(32)])/cos(pi/8)
    s1[0:32:8] = 1
    s2[0:32:8] = 1
    w1 = array([exp(-1j*2*pi*i/n)*s1[i%32] for i in range(128)])
    w2 = array([exp(1j*2*pi*i/n)*s2[i%32] for i in range(128)])
    u = fft256(x[0:n:2])
    z = fft128s(x[1:n:4], s1) * w1
    z1 = fft128s(shift_right(x[3:n:4]), s2) * w2
    return split_radix_butt(u, z, z1, n)


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
# test_func(lambda x: fft64s(x, zeros(x.size//4)+1), 64)
# test_func(fft256, 256)
test_func(lambda x: fft128s(x, zeros(x.size//4)+1), 128)
# test_func(fft512, 512)