from numpy import array


def isEquals(x, y):
    if abs(x-y) < 1e-15:
        return True
    return False


def flops4Mul(x):
    if isEquals(x.real, 0):
        if isEquals(abs(x.imag), 1):
            return 0
        else:
            return 2
    if isEquals(x.imag, 0):
        if isEquals(abs(x.real), 1):
            return 0
        else:
            return 2

    if isEquals(abs(x.real), 1):
        if isEquals(abs(x.imag), 1):
            return 2
        else:
            return 4
    if isEquals(abs(x.imag), 1):
        return 4

    if isEquals(abs(x.real), abs(x.imag)):
        return 4
    return 6


def flops4Muls(x):
    tmp = x.reshape(x.size)
    s = 0
    for n in tmp:
        s += flops4Mul(n)
    return s


# test
# x = array([1, 1-1j, -1+2j, 2+1j, 2+2j, 2-3j, 2j, -2])
# ans = 24
# if flops4Muls(x) != ans:
#     print("error, ", ans, "!=", flops4Muls(x))