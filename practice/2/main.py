import numpy as np

from math import sqrt
from tabulate import tabulate


def find_b(L, x):  # находим b
    b = np.dot(L, x)
    return b


def solve(L, b):  # решаем регуляризационную систему
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.transpose(), y)
    return x


def sqrt_method(A):  # метод квадратого корня
    L = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(i):
            temp = 0
            for k in range(j):
                temp += L[i][k] * L[j][k]
            L[i][j] = (A[i][j] - temp) / L[j][j]
        temp = A[i][i]
        for k in range(i):
            temp -= L[i][k] * L[i][k]
        L[i][i] = sqrt(temp)
    return L


def variation_matrix(A, alpha, b, x):
    df.append([alpha, np.linalg.cond(A), np.linalg.cond(A + alpha * np.eye(A.shape[0])),
                 np.linalg.norm(x - solve(sqrt_method(A + alpha * np.eye(A.shape[0])), b))])
    return df


def print_all(A, df, alpha, diff_matr):
    print("Матрица:")
    print(*A, sep='\n')
    print()
    print(tabulate(df, headers=['alpha', 'cond(A)', 'cond(A + alpha * E)', '||x - x_a||'],
                   tablefmt='github', numalign="right"))
    print()
    print("Наилучшее значение alpha =", alpha)
    print()
    print("||x - x_a|| для различных матриц:")
    print(tabulate(diff_matr,
                   headers=['Ax = b', 'A + alpha * x = b', 'A + 10 * alpha * x = b', 'A + 0.1 * alpha * x = b'],
                   tablefmt='github', numalign="right"))
    print()


data = []
with open("data.txt") as f:
    for line in f:
        if line == '\n':
            A = np.array(data, dtype=float)
            x = np.ones(A.shape[0])
            b = find_b(A, x)
            df = []
            for i in (-2, -3, -5, -7, -9, -12):
                variation_matrix(A, 10 ** i, b, x)
            x = np.random.uniform(0, 100, size=A.shape[0])
            b = find_b(A, x)
            min_diff = df[0][3]
            for i in range(len(df)):
                if min_diff > df[i][3]:
                    min_diff = df[i][3]
                    alpha = df[i][0]
            diff_matr = [[np.linalg.norm(x - solve(sqrt_method(A), b)),
                          np.linalg.norm(x - solve(sqrt_method(A + alpha * np.eye(A.shape[0])), b)),
                          np.linalg.norm(x - solve(sqrt_method(A + 10 * alpha * np.eye(A.shape[0])), b)),
                          np.linalg.norm(x - solve(sqrt_method(A + 0.1 * alpha * np.eye(A.shape[0])), b))]]
            print_all(A, df, alpha, diff_matr)
            data.clear()
            continue
        else:
            data.append([float(x) for x in line.split()])
