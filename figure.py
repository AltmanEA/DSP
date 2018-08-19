import matplotlib.pyplot as plt
from numpy.core.umath import pi
from numpy.ma import cos


def fun_dct1(x, i, n):
    return cos(pi*x*i/n)


def fun_dct2(x, i, n):
    return cos(pi*x*(i+1/2)/n)


n = 64
x = [i for i in range(n)]
def fun(x, i):
    return fun_dct2(x, i, n)


plt.figure(figsize=(8, 16))
m=8
for i in range(m):
    y = [-fun(x, i)/2.5+i for x in x]
    plt.plot(x, y)
    plt.axhline(i, linewidth=0.3, linestyle='dashed')
plt.axis([0, n, m, -1])
plt.show()