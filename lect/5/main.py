import math
import cmath
import numpy as np
import sympy as sym
import random
from tabulate import tabulate


def F():
    p = sym.symbols('p')
    return (p ** (1 / 4) - 2) / (p ** (1 / 2) - 1)


def der_F(n):
    f = F()
    p = sym.symbols('p')
    return sym.diff(f, p, n)


def Widder(t, n):
    p = sym.symbols('p')
    return ((-1) ** n) * (n / t) ** (n + 1) * der_F(int(n)).subs(p, n / t) / math.factorial(int(n))


def dk(n, k):
    d = [0]
    random.seed(5)
    for i in range(1, k + 1):
        temp = (random.randint(0, 100) % 10) / n
        while temp in d:
            temp = (random.randint(0, 100) % 10) / n
        d.append(temp)
    return sorted(d)


def ck(n, k):
    d = dk(n, k)
    c = [0]
    for j in range(1, k + 1):
        ck = 1
        for i in range(1, n + 1):
            if i == j:
                continue
            ck *= d[j] / (d[j] - d[i])
        c.append(ck)
    return c, d


def accelerated_Widder(t, n, k):
    c, d = ck(n, k)
    res = 0
    for j in range(1, k + 1):
        temp = n * d[j]
        res += c[j] * Widder(t, temp)
    return res


def G(z, t):
    p = sym.symbols('p')
    return 1 / t * F().subs(p, (1 - z) / t)


def em(x, m):
    return cmath.exp(1j * 2 * math.pi * x / m)


def numerical_Widder(n, m, t):
    res = 0
    r = 0.1
    for j in range(1, m + 1):
        temp = r * em(j, m)
        res += complex((temp ** (-n)) * G(temp, t / n))
    a = res.real
    b = res.imag
    if ((a < 0) and (b < 0)) or ((a < 0) and (b > 0)):
        res = -1 * math.sqrt((a ** 2 + b ** 2)) / m
    else:
        res = math.sqrt((a ** 2 + b ** 2)) / m
    return res


def main():
    res = np.zeros((9, 4))
    headers = ['n', 'Wn(f, 0.5)', 'Wn(n, f, 0.5)', 'Wnm(n, f, 0.5) ']
    for i in range(1, 10):
        res[i - 1, 0] = i
        res[i - 1, 1] = Widder(0.5, i)
    for i in range(1, 10):
        res[i - 1, 2] = accelerated_Widder(0.5, i, i)
    m = 500
    for i in range(1, 10):
        res[i - 1, 3] = numerical_Widder(i, m, 0.5)
    res_table = tabulate(res, headers=headers, tablefmt='github', numalign="right", floatfmt=(".0f", ".10f", ".10f", ".10f"))
    print(res_table)


if __name__ == '__main__':
    main()
