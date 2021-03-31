import numpy as np
import math


def max_mod(A):
    max_elem = 0
    i_max = 0
    j_max = 0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if abs(max_elem) < abs(A[i, j]):
                max_elem = A[i, j]
                i_max = i
                j_max = j
    return i_max, j_max


def max_gersh(A, iter, r, i, j):
    max_elem = 0
    max_i = 0
    max_j = 0
    if iter == 0:
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if i != j:
                    r[i] += abs(A[i, j]) ** 2
    else:
        for k in (i, j):
            r[k] = 0
            for j in range(A.shape[0]):
                if k != j:
                    r[k] += abs(A[k, j]) ** 2
    for i in range(A.shape[0]):
        if max_elem < r[i]:
            max_elem = r[i]
            max_i = i
    max_elem = 0
    for j in range(A.shape[0]):
        if max_elem < abs(A[max_i, j]) and max_i != j:
            max_elem = A[max_i, j]
            max_j = j
    return max_i, max_j


def jacobi_method(A, eps):
    iter = 0
    r = np.array(np.zeros(A.shape[0]))
    max_i = 0
    max_j = 0
    while True:
        H = np.eye(A.shape[0], dtype=float)
        # max_i, max_j = max_gersh(A, iter, r, max_i, max_j)  # оптимальный элемент с помощью кругов Гершгорина
        max_i, max_j = max_mod(A)  # оптимальный элемент - наибольший наддиагональный по модулю
        if abs(A[max_i, max_j]) < eps:
            return np.diag(A), iter
        iter += 1
        phi = 1 / 2 * (math.atan((2 * A[max_i, max_j]) / (A[max_i, max_i] - A[max_j, max_j])))
        H[max_i, max_i] = math.cos(phi)
        H[max_j, max_j] = math.cos(phi)
        H[max_i, max_j] = - math.sin(phi)
        H[max_j, max_i] = math.sin(phi)
        A = H.T @ A @ H


n = 10
A = np.array(np.zeros((n, n)), dtype=float)
for i in range(n):
    for j in range(n):
        A[i][j] = 1 / (i + 1 + j + 1 - 1)
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    print("Матрица Гильбертва", n, "порядка")
    # print(*A, sep='\n')
    print("Погрешность:", eps)
    lambda_comp, iter = jacobi_method(A, eps)
    print("Количество итераций:", iter)

