import numpy as np
import cmath
import scipy

from math import sqrt
from tabulate import tabulate
from scipy import linalg


def find_b(L, x):  # находим b
    b = np.dot(L, x)
    return b


def solve(L, b):  # решаем регуляризационную систему
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.transpose(), y)
    return x


def solveLU(L, U, b):  # решаем регуляризационную систему
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x


def sqrt_method(A):  # метод квадратого корня
    L = np.zeros((A.shape[0], A.shape[0]), dtype=complex)
    for i in range(A.shape[0]):
        for j in range(i):
            temp = 0
            for k in range(j):
                temp += L[i][k] * L[j][k]
            L[i][j] = (A[i][j] - temp) / L[j][j]
        temp = A[i][i]
        for k in range(i):
            temp -= L[i][k] * L[i][k]
        L[i][i] = cmath.sqrt(temp)
    return L


# def variation_matrix(A, alpha, b, x):
#     df.append([alpha, np.linalg.cond(A), np.linalg.cond(A + alpha * np.eye(A.shape[0])),
#                # np.linalg.norm(x - solve(sqrt_method(np.dot(A, A.transpose().conj()) + alpha * np.eye(A.shape[0])),
#                #                          np.dot(A.transpose().conj(), b)))])
#                np.linalg.norm(x - solve(sqrt_method(A + alpha * np.eye(A.shape[0])), b))])
#     return df

def variation_matrix(A, alpha, b, x):
    L, U = scipy.linalg.lu(A + alpha * np.eye(A.shape[0]), permute_l=True)
    df.append([alpha, np.linalg.cond(A), np.linalg.cond(A + alpha * np.eye(A.shape[0])),
               # np.linalg.norm(x - solve(sqrt_method(np.dot(A, A.transpose().conj()) + alpha * np.eye(A.shape[0])),
               #                          np.dot(A.transpose().conj(), b)))])
               np.linalg.norm(x - solveLU(L, U, b))])
    return df


def print_all(A, df, alpha, diff_matr):
    L = sqrt_method(A)
    # if np.array_equal(np.round(L, 5), np.round(np.linalg.cholesky(A), 5)):
    #     print("да")
    # else:
    #     print("нет")
    # k = 0
    # for i in range(L.shape[0]):
    #     for j in range(L.shape[0]):
    #         if L[i][j] == np.linalg.cholesky(A)[i][j]:
    #             k += 1
    # if k == L.shape[0] ** 2:
    #     print("дадада")
    # if L[2][2] == np.linalg.cholesky(A)[2][2]:
    #     print("Крайние равны")
    # else:
    #     print("Крайние не равны")
    # print(L[2][2])
    # print(np.linalg.cholesky(A)[2][2])
    # print(L)
    # print()
    # print(np.linalg.cholesky(A))

    print("Матрица:")
    # print(*A, sep='\n')
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


# data = []
# with open("data.txt") as f:
#     for line in f:
#         if line == '\n':
#             A = np.array(data, dtype=float)
#             x = np.ones(A.shape[0])
#             b = find_b(A, x)
#             df = []
#             for i in (-2, -3, -5, -7, -9, -12):
#                 variation_matrix(A, 10 ** i, b, x)
#             x = np.random.uniform(0, 100, size=A.shape[0])
#             b = find_b(A, x)
#             min_diff = df[0][3]
#             for i in range(len(df)):
#                 if min_diff > df[i][3]:
#                     min_diff = df[i][3]
#                     alpha = df[i][0]
#             diff_matr = [[np.linalg.norm(x - solve(sqrt_method(A), b)),
#                           np.linalg.norm(x - solve(sqrt_method(A + alpha * np.eye(A.shape[0])), b)),
#                           np.linalg.norm(x - solve(sqrt_method(A + 10 * alpha * np.eye(A.shape[0])), b)),
#                           np.linalg.norm(x - solve(sqrt_method(A + 0.1 * alpha * np.eye(A.shape[0])), b))]]
#             print_all(A, df, alpha, diff_matr)
#             data.clear()
#             continue
#         else:
#             data.append([float(x) for x in line.split()])
n = 15
A = np.array(np.zeros((n, n)), dtype=complex)
for i in range(n):
    for j in range(n):
        A[i][j] = 1 / (i + 1 + j + 1 - 1)
# print("Матрица:")
# print(*A, sep='\n')
# print()
x = np.ones(A.shape[0])
b = find_b(A, x)
df = []
for i in (-2, -3, -5, -7, -9, -10, -11, -12):
    variation_matrix(A, 10 ** i, b, x)
x = np.random.uniform(1, 1, size=A.shape[0])
print(x)
b = find_b(A, x)
min_diff = df[0][3]
for i in range(len(df)):
    if min_diff > df[i][3]:
        min_diff = df[i][3]
        alpha = df[i][0]
# diff_matr = [[np.linalg.norm(x - solve(sqrt_method(A), b)),
#               np.linalg.norm(x - solve(sqrt_method(A + alpha * np.eye(A.shape[0])), b)),
#               np.linalg.norm(x - solve(sqrt_method(A + 10 * alpha * np.eye(A.shape[0])), b)),
#               np.linalg.norm(x - solve(sqrt_method(A + 0.1 * alpha * np.eye(A.shape[0])), b))]]

L, U = scipy.linalg.lu(A, permute_l=True)
La, Ua = scipy.linalg.lu(A + alpha * np.eye(A.shape[0]), permute_l=True)
L10, U10 = scipy.linalg.lu(A + 10 * alpha * np.eye(A.shape[0]), permute_l=True)
L01, U01 = scipy.linalg.lu(A + 0.1 * alpha * np.eye(A.shape[0]), permute_l=True)
print(solveLU(L, U, b))
diff_matr = [[np.linalg.norm(x - solveLU(L, U, b)),
              np.linalg.norm(x - solveLU(La, Ua, b)),
              np.linalg.norm(x - solveLU(L10, U10, b)),
              np.linalg.norm(x - solveLU(L01, U01, b))]]




# print(A + 10 * np.eye(A.shape[0]))
# print(np.linalg.eigvalsh(A))
# linalg.cholesky(A)
# linalg.cholesky(A, lower=True)
print_all(A, df, alpha, diff_matr)
L1, U = scipy.linalg.lu(A, permute_l=True)
# print(np.dot(L1, U))
# print(L1)
# print(U)