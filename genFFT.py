from numpy import array, exp, pi, log2, zeros, vstack, hstack, conj
from numpy.fft import fft
from tools import flops4Muls, wf, bit_reverse_index


# region Main function
def genFFT(data, w_matrix, p_matrix):
    stages, n = w_matrix.shape
    half = n // 2
    tmp = zeros(data.shape, dtype=complex)
    for s in range(stages-1):
        data *= w_matrix[s]
        for i in range(n):
            tmp[i] = data[p_matrix[s, i]]
        for i in range(half):
            data[2*i] = tmp[i] + tmp[i+half]
            data[2*i+1] = tmp[i] - tmp[i+half]
    data *= w_matrix[stages-1]
    for i in range(n):
        tmp[i] = data[p_matrix[stages-1, i]]
    return tmp
# endregion

# region Utils
def get_flops(w_matrix):
    stages, n = w_matrix.shape
    butt_flops = 2*n*log2(n)
    return butt_flops + flops4Muls(w_matrix)


def revers_matrix_row(matrix):
    rows, n = w_matrix.shape
    result = zeros(matrix.shape, matrix.dtype)
    for i in range(rows):
        result[i, :] = matrix[rows-i-1, :]
    return result
# endregion


# region Generators
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

def genetare_step_radix2(w_matrix1, p_matrix1, w_matrix2, p_matrix2):
    stages, n = w_matrix1.shape
    w_matrix = zeros((stages+1, 2*n), dtype=complex)
    p_matrix = zeros((stages+1, 2*n))
    for i in range(2*n):
        p_matrix[0, i] = i
        w_matrix[0, i] = 1
        if i % 2 == 0:
            for j in range(1, stages):
                p_matrix[j, i] = 2*p_matrix1[j-1, i//2]
                w_matrix[j, i] = w_matrix1[j-1, i//2]
            p_matrix[stages, i] = p_matrix1[stages-1, i//2]
            w_matrix[stages, i] = w_matrix1[stages-1, i//2]
        else:
            p_matrix[1, i] = 2*p_matrix2[0, i//2] + 1
            w_matrix[1, i] = w_matrix2[0, i//2] * wf(i//2, 2*n)
            for j in range(2, stages):
                p_matrix[j, i] = 2*p_matrix2[j-1, i//2] + 1
                w_matrix[j, i] = w_matrix2[j-1, i//2]
            p_matrix[stages, i] = p_matrix2[stages-1, i//2] + n
            w_matrix[stages, i] = w_matrix2[stages-1, i//2]
    return w_matrix, p_matrix
# endregion


# region Test
n = 16
print("point -", n)
w_matrix = generate_radix2(n)
p_matrix = generate_p_matrix(n)

# w_matrix = generate_radix2(n//2)
# p_matrix = generate_p_matrix(n//2)
# w_matrix, p_matrix = genetare_step_radix2(w_matrix, p_matrix, w_matrix, p_matrix)


x = array([i for i in range(n)], dtype=complex)
pfft = fft(x)
y = genFFT(x, w_matrix, p_matrix)
print("accuracy - ", max(abs(y - pfft)))
print("flops -", int(get_flops(w_matrix)))
# endregion