from numpy import array, zeros
from numpy.fft import fft


def extended_euclid(number_one, number_two):
    a = number_one
    b = number_two
    x2 = 1
    x1 = 0
    y2 = 0
    y1 = 1
    while b > 0:
        q = a // b
        r = a - q * b
        tmp1 = x2 - q * x1
        tmp2 = y2 - q * y1
        x2 = x1
        x1 = tmp1
        y2 = y1
        y1 = tmp2
        a = b
        b = r
    return a, x2, y2


def good_thomas(x, P, Q):
    N = P * Q
    _, p1, q1 = extended_euclid(P, Q)
    if p1 < 0:
        p1 += Q
    if q1 < 0:
        q1 += P
    i = array([i for i in range(N)])
    s = q1*i % P
    t = p1*i % Q
    r = i % P
    k = i % Q

    x1 = zeros([P, Q], dtype=complex)
    for i in range(N):
        x1[s[i], t[i]] = x[i]

    for i in range(P):
        x1[i, :] = fft(x1[i, :])
    for i in range(Q):
        x1[:, i] = fft(x1[:, i])

    for i in range(N):
        x[i] = x1[r[i], k[i]]

    return x


# def agarwal_cooley(x, y):


# test
arr_p = [3, 5, 7, 125]
arr_q = [5, 7, 11, 8]
for i in range(4):
    x = array([2*i+1j*i for i in range(arr_p[i]*arr_q[i])])
    pfft = fft(x)
    x = good_thomas(x, arr_p[i], arr_q[i])
    print("accuracy for (", arr_p[i], arr_q[i], ") --", max(abs(pfft-x)))