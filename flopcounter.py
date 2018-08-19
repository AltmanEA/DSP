import numpy as np

# np.set_printoptions(precision=2)
#
# # Split radix number of complex multiplication
# n = 5
# ops = {
#     0: lambda k: k * (2 ** k),
#     1: lambda k: (2 ** k),
#     2: lambda k: ((-1) ** k),
#     3: lambda k: k,
#     4: lambda k: 1
# }
# A = [[ops[i](j) for i in range(n)] for j in range(n)]
# B = [0, 0, 0, 2, 8]
# C = np.linalg.solve(A, B) * 9
# print("\nMultiplication in split radix")
# print(np.array(A))
# print(C)
#
# # Split radix number of complex addition
# # c_1*k*2^k + c_2*2^k + c_3*(-1)^k
# n = 3
# ops = {
#     0: lambda k: k * (2 ** k),
#     1: lambda k: (2 ** k),
#     2: lambda k: ((-1) ** k),
# }
# A = [[ops[i](j) for i in range(n)] for j in range(n)]
# B = [0, 2, 8]
# C = np.linalg.solve(A, B)
# print("\nAddition in split radix")
# print(np.array(A))
# print(C)
#
# # Split radix number of flop
# n = 6
# ops = {
#     0: lambda k: k * (2 ** k),
#     1: lambda k: (2 ** k),
#     2: lambda k: ((-1) ** k) * k,
#     3: lambda k: k,
#     4: lambda k: ((-1) ** k),
#     5: lambda k: 1
# }
# A = [[ops[i](j) for i in range(n)] for j in range(1, n+1)]
# B = [4, 16, 56, 168, 456, 1160]
# C = np.linalg.solve(A, B)
# print("\nFlops in split radix")
# print(np.array(A))
# print(C)
#
# # Modified split radix number of flop
# B = [4, 16, 56, 168, 456, 1152]
# C = np.linalg.solve(A, B)
# print("\nFlops in modified split radix")
# print(np.array(A))
# print(C)
# print(C*27)


# Mixed radix
extras = {
    2: 0,
    4: 0,
    8: 8,
    16: 40,
    32: 136,
    64: 392,
}

m = 7
stages = [8]

n = 2 ** m
n1 = n
n2 = 1
s = 2 * m * n
for stage in stages:
    n1 = n1//stage
    if n1 > 2:
        if stage > 2:
            s = s + n2*(6 * ((stage - 1) * (n1 - 1) - 5) + 16) + n*extras[stage]//stage
        else:
            s = s + n2*(6 * (n1 - 4) + 8)
    else:
        if stage > 2:
            s = s + n2*(6 * (stage - 4) + 8)+n*extras[stage]//stage
    n2 = n//n1
s = s + n2*(extras[n1])
print(s)