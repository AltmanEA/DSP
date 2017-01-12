from numpy import zeros, log2, array
from numpy.fft import fft

from tools import bit_reverse_index, wf


def model_fft(data, w, permutations):
    stages, points = w.shape
    assert w.shape == permutations.shape
    assert points == data.size

    result = zeros((stages+1, points), dtype=complex)
    result[0, :] = data

    return result



def generate_p_matrix(n):
    stages = int(log2(n))
    p_matrix = zeros((stages+1, n))
    for i in range(n):
        for j in range(stages):
            p_matrix[j, i] = i
        p_matrix[stages, i] = bit_reverse_index(i, stages)
    return p_matrix


def generate_radix2(n):
    stages = int(log2(n))
    w_matrix = zeros((stages+1, n), dtype=complex)+1
    block = 1
    len = n
    for s in range(1, stages):
        ind = 1
        for i in range(len//2):
            for j in range(block):
                w_matrix[s, ind] = wf(i, len)
                ind += 2
        len //= 2
        block *= 2
    return w_matrix


n = 16
print("point -", n)
w_matrix = generate_radix2(n)
p_matrix = generate_p_matrix(n)
x = array([i for i in range(n)], dtype=complex)
pfft = fft(x)
y = model_fft(x, w_matrix, p_matrix)