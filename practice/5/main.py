import numpy as np


def power_method(A, eps):
    x_0 = np.array(np.ones(A.shape[0]))
    lamb = 0
    iter = 0
    while True:
        iter += 1
        x_1 = np.dot(A, x_0)
        if abs(x_1[0] / x_0[0] - lamb) < eps:
            lamb = x_1[0] / x_0[0]
            break
        lamb = x_1[0] / x_0[0]
        x_0 = x_1
    return lamb, iter


def scal_method(A, eps):
    x_0 = np.array(np.ones(A.shape[0]))
    y_0 = np.array(np.ones(A.shape[0]))
    lamb = 0
    iter = 0
    while True:
        iter += 1
        x_1 = np.dot(A, x_0)
        y_1 = np.dot(A.T, y_0)
        if abs(np.dot(x_1, y_1) / np.dot(x_0, y_1) - lamb) < eps:
            lamb = np.dot(x_1, y_1) / np.dot(x_0, y_1)
            break
        lamb = np.dot(x_1, y_1) / np.dot(x_0, y_1)
        x_0 = x_1
        y_0 = y_1
    return lamb, iter


n = 4
A = np.array(np.zeros((n, n)), dtype=float)
for i in range(n):
    for j in range(n):
        A[i][j] = 1 / (i + 1 + j + 1 - 1)
lambda_acc = max(abs(np.linalg.eig(A)[0]))
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    print("Матрица:")
    print(*A, sep='\n')
    print("Погрешность:", eps)
    print("Степенной метод:")
    print("    Количество итераций:", power_method(A, eps)[1])
    print("    |lambda_acc - lambda|:", abs(lambda_acc - abs(power_method(A, eps)[0])))
    print("Метод скалярных произведений:")
    print("    Количество итераций:", scal_method(A, eps)[1])
    print("    |lambda_acc - lambda|:", abs(lambda_acc - abs(scal_method(A, eps)[0])))


