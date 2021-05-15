import math
import numpy as np
from tabulate import tabulate
from scipy.special import sh_legendre


def u(x):
    # return math.log(1 + 2 * x) / (2 * x)  # 1
    return 1 / (2 * x) - math.log(1 + 2 * x) / (4 * x * x)  # 2


def K(x, s):
    return 1 / (1 + 2 * x * s)


def find_nodes(N, a, b):
    sk = []
    h = (b - a) / N
    alpha = a + h / 2
    for k in range(1, N + 1):
        sk.append(alpha + (k - 1) * h)
    return sk


def find_Aw_k(w, a, b, M):
    h = (b - a) / M
    nodes = find_nodes(M, a, b)
    return lambda x: h * sum(K(x, t) * w(t) for t in nodes)


def find_b(N, a, b, M):
    h = (b - a) / M
    matrix = np.zeros((N, N))
    nodes = find_nodes(M, a, b)
    w_k = [sh_legendre(i) for i in range(N)]
    for j in range(N):
        for k in range(N):
            matrix[j, k] = h * sum(find_Aw_k(w_k[k], a, b, M)(x) * w_k[j](x) for x in nodes)
    return matrix


def find_u(N, a, b, M):
    h = (b - a) / M
    uk = np.zeros((N, 1))
    nodes = find_nodes(M, a, b)
    w_k = [sh_legendre(i) for i in range(N)]
    for i in range(N):
        uk[i] = h * sum(u(x) * w_k[i](x) for x in nodes)
    return uk


def find_solution(N, alpha, a, b, M):
    B = find_b(N, a, b, M)
    uk = find_u(N, a, b, M)
    nodes = find_nodes(M, a, b)
    w_k = [sh_legendre(i) for i in range(N)]
    coef_solution = np.linalg.solve(B.transpose() @ B + alpha * np.eye(N), B.transpose() @ uk)
    solution = 0
    return lambda x: sum(sh_legendre(i)(x) * coef_solution[i] for i in range(N))


def scalar_mult(fun1,fun2):
    nodes = find_nodes(20, 0, 1)
    return (1/20)*sum(fun1(x)*fun2(x) for x in nodes)


def norm(fun):
    return scalar_mult(fun,fun)**(0.5)


def main():
    N = 9
    M = 20
    max_degree_alpha = 15
    a = 0
    b = 1
    res = np.zeros((N, max_degree_alpha - 4 + 1))
    headers = [r'n\a']
    nodes = find_nodes(20, 0, 1)
    for j in range(5, max_degree_alpha + 1):
        headers.append(10 ** (-j))
    for i in range(1, N + 1):
        u0 = np.ones((i, 1))
        res[i - 1, 0] = i + 1
        for j in range(5, max_degree_alpha + 1):
            solution = find_solution(i, 10 ** (-j), a, b, M)
            # res[i - 1, j - 4] = norm(lambda x: solution(x) - 1)  # 1
            res[i - 1, j - 4] = norm(lambda x: solution(x) - x)  # 2
    res_table = tabulate(res, headers=headers, tablefmt='github', numalign="right")
    print(res_table)
    index = np.unravel_index(np.argmin(res, axis=None), res.shape)
    print('Оптим. n =', res[index[0], 0])
    print('Оптим. alpha =', headers[index[1]])
    print('Погрешность =', res[index])


if __name__ == '__main__':
    main()



