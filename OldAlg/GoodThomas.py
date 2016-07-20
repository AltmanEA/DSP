from numpy import array, zeros
from numpy.fft import fft
from OldAlg.tools import extended_euclid


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

# test
arr_p = [3, 5, 7, 125]
arr_q = [5, 7, 11, 8]
for i in range(arr_p.__sizeof__()):
    x = array([2*i+1j*i for i in range(arr_p[i]*arr_q[i])])
    pfft = fft(x)
    x = good_thomas(x, arr_p[i], arr_q[i])
    print("error for ", arr_p[i], arr_q[i], "--", max(abs(pfft-x)))