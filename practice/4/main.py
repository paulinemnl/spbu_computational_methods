import numpy as np


def find_b(A, x):  # находим b
    b = np.dot(A, x)
    return b


def find_coef(A, b):  # находим alpha и beta
    alpha = np.array(np.zeros((A.shape[0], A.shape[0])))
    beta = np.array(np.zeros(b.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j:
                alpha[i][j] = - A[i][j] / A[i][i]
                beta[i] = b[i] / A[i][i]
            else:
                alpha[i][i] = 0
    return alpha, beta


def iteration_method(alpha, beta, x, eps, iter=1):  # метод простой итерации
    err = eps + 1
    while err > eps and iter < 500:
        err = np.linalg.norm(np.dot(alpha, x) + beta - x)
        x = np.dot(alpha, x) + beta
        iter += 1
    x = np.dot(alpha, x) + beta
    return x, iter


def seidel_method(A, b, eps):  # метод Зейделя
    iter = 0
    x = np.array(np.zeros((b.shape[0])))
    err = eps + 1
    while err > eps:
        x_new = x.copy()
        for i in range(A.shape[0]):
            x1 = sum(A[i][j] * x_new[j] for j in range(i))
            x2 = sum(A[i][j] * x[j] for j in range(i + 1, A.shape[0]))
            x_new[i] = (b[i] - x1 - x2)/A[i][i]
        err = np.linalg.norm(x_new - x)
        iter += 1
        x = x_new
    return x, iter


n = 3
A = np.array(np.zeros((n, n)), dtype=float)
x = np.random.uniform(0, 100, size=A.shape[0])
for i in range(n):
    for j in range(n):
        A[i][j] = 1 / (i + 1 + j + 1 - 1)
b = find_b(A, x)
alpha, beta = find_coef(A, b)
for eps in (1e-4, 1e-7, 1e-10, 1e-13):
    print("Матрица:")
    print(*A, sep='\n')
    print("Погрешность:", eps)
    print("Методе простой итерации:")
    print("   Количество итераций (максимум = 500):", iteration_method(alpha, beta, beta, eps)[1])
    print("   ||x - x_a||:", np.linalg.norm(x - iteration_method(alpha, beta, beta, eps)[0]))
    print("Метод Зейделя:")
    print("   Количество итераций:", seidel_method(A, b, eps)[1])
    print("   ||x - x_a||:", np.linalg.norm(x - seidel_method(A, b, eps)[0]))



