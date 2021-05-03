import numpy as np
import math
from tabulate import tabulate


def u(x):
    return math.log(1 + 2 * x) / (2 * x)


def K(x, s):
    return 1 / (1 + 2 * x * s)


def find_nodes(N, a, b):
    sk = []
    h = (b - a) / N
    alpha = a + h / 2
    for k in range(1, N + 1):
        sk.append(alpha + (k - 1) * h)
    return sk


def find_c(N, a, b):
    h = (b - a) / N
    c = np.zeros((N, N))
    nodes = find_nodes(N, a, b)
    for i in range(N):
        for j in range(N):
            c[i, j] = K(nodes[i], nodes[j]) * h
    return c


def find_u(N, a, b):
    uk = np.zeros((N, 1))
    nodes = find_nodes(N, a, b)
    for i in range(N):
        uk[i] = u(nodes[i])
    return uk


def find_solution(N, alpha, a, b):
    C = find_c(N, a, b)
    uk = find_u(N, a, b)
    return np.linalg.solve(C.transpose() @ C + alpha * np.eye(N), C.transpose() @ uk)


def main():
    N = 10
    max_degree_alpha = 15
    a = 0
    b = 1
    res = np.zeros((N, max_degree_alpha - 4 + 1))
    headers = [r'n\a']
    for j in range(5, max_degree_alpha + 1):
        headers.append(10 ** (-j))
    for i in range(1, N + 1):
        u0 = np.ones((10 * i, 1))
        res[i - 1, 0] = 10 * i
        for j in range(5, max_degree_alpha + 1):
            solution = find_solution(10 * i, 10 ** (-j), a, b)
            res[i - 1, j - 4] = np.linalg.norm(u0 - solution)
    res_table = tabulate(res, headers=headers, tablefmt='github', numalign="right")
    print(res_table)
    index = np.unravel_index(np.argmin(res, axis=None), res.shape)
    print('Оптим. n =', res[index[0], 0])
    print('Оптим. alpha =', headers[index[1]])


if __name__ == '__main__':
    main()

