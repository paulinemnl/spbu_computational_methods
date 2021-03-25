import numpy as np


def find_b(A, x):  # находим b
    b = np.dot(A, x)
    return b


def solve_rotation(A, b):  # метод вращений
    q = np.column_stack([A, b])
    for i in range(q.shape[0] - 1):
        for j in range(i + 1, q.shape[0]):
            c = q[i, i] / (q[i, i] ** 2 + q[j, i] ** 2) ** (1/2)
            s = q[j, i] / (q[i, i] ** 2 + q[j, i] ** 2) ** (1/2)
            tmp = q[i, :] * c + q[j, :] * s
            q[j, :] = q[i, :] * -s + q[j, :] * c
            q[i, :] = tmp
    x = np.linalg.solve(q[:, :-1], q[:, -1])
    return x


for n in (3, 4, 5, 10):
    A = np.array(np.zeros((n, n)), dtype=float)
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + 1 + j + 1 - 1)
    print("Матрица Гильберта", n, "порядка:")
    x = np.random.uniform(0, 100, size=A.shape[0])
    b = find_b(A, x)
    x_rot = solve_rotation(A, b)
    print("    ||x - x_rot|| =", np.linalg.norm(x - x_rot))

