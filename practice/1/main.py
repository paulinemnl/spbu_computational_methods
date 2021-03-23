import numpy as np


def find_b(A, x_acc=np.array([None])):  # находим b
    if x_acc[0] is None:
        x_acc = np.random.uniform(-100, 100, size=A.shape[0])  # задаем случайные значения для x_acc, если он не задан
    b = np.dot(A, x_acc)
    print(b)
    return b, x_acc


def cond(A):  # находим числа обусловленности
    v0 = 1
    mult = np.zeros(A.shape[0])
    cond_s = np.dot(np.linalg.norm(A), np.linalg.norm(np.linalg.inv(A)))  # Спектральный критерий
    for i in range(A.shape[0]):
        v0 *= np.linalg.norm(A[i])
    cond_v = v0 / abs(np.linalg.det(A))  # Объемный критерий
    C = np.linalg.inv(A)
    for i in range(A.shape[0]):
        mult[i] = np.linalg.norm(A[i]) * np.linalg.norm(C[:, i])
    cond_a = max(mult)  # Угловой критерий
    return cond_s, cond_v, cond_a


def varied_matrix(A, b):  # варьируем матрицу и b
    k = 0
    x_inacc = np.zeros((3, A.shape[0]))
    for i in (-2, -5, -8):
        A_inacc = A - 10 ** i
        b_inacc = b - 10 ** i
        for j in range(A.shape[0]):
            x_inacc[k] = np.linalg.solve(A, b_inacc)
        k += 1
    return x_inacc


def print_matrix(A, x_acc, x_inacc):
    print("Матрица:")
    print(*A, sep='\n')
    print()
    print("Точное решение:")
    for elem in x_acc:
        print('|{:12.8f}'.format(elem), '|')
    print()
    print("Числа обусловленности:")
    print("    Спектральный критерий обусловленности:", cond_s)
    print("    Объемный критерий (критерий Ортеги):", cond_v)
    print("    Угловой критерий:", cond_a)
    print()
    k = 0
    for i in (-2, -5, -8):
        print("eps = 10^({}):".format(i))
        print("    Приближенное решение:")
        for elem in x_inacc[k]:
            print("   ", '|{:16.8f}'.format(elem), '|')
        print("    Невязка:")
        for elem in abs(x_acc - x_inacc[k]):
            print("   ", '|{:16.12f}'.format(elem), '|')
        k += 1
    print()


data = []
with open("data.txt") as f:
    for line in f:
        if line == '\n':
            A = np.array(data, dtype=float)
            b, x_acc = find_b(A)
            cond_s, cond_v, cond_a = cond(A)
            x_inacc = varied_matrix(A, b)
            print_matrix(A, x_acc, x_inacc)
            print("{:-^100}".format(""))
            data.clear()
            continue
        else:
            data.append([float(x) for x in line.split()])
