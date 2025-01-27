import numpy as np
import scipy.stats


def matrix_multiplication(matrix_a, matrix_b):
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Incorrect input")

    result = [[0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    if a_1 == a_2:
        return None

    a_diff, b_diff, c_diff = a1 - a2, b1 - b2, c1 - c2

    if a_diff == 0 and b_diff == 0:
        return []

    if a_diff == 0:
        solution = -c_diff / b_diff
        y_value = a1 * solution ** 2 + b1 * solution + c1
        return [(solution, y_value)]

    discriminant = b_diff ** 2 - 4 * a_diff * c_diff

    if discriminant < 0:
        return None
    elif discriminant == 0:
        solution = -b_diff / (2 * a_diff)
        y_value = a1 * solution ** 2 + b1 * solution + c1
        return [(solution, y_value)]
    else:
        sqrt_disc = discriminant ** 0.5
        x1 = (-b_diff + sqrt_disc) / (2 * a_diff)
        x2 = (-b_diff - sqrt_disc) / (2 * a_diff)
        return [(x1, a1 * x1 ** 2 + b1 * x1 + c1),
                (x2, a1 * x2 ** 2 + b1 * x2 + c1)]


def skew(x):
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n
    std_dev = variance ** 0.5

    skewness = sum((xi - mean) ** 3 for xi in x) / n
    skewness /= std_dev ** 3

    return round(skewness, 2)


def kurtosis(x):
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n
    std_dev = variance ** 0.5
    kurt = sum((xi - mean) ** 4 for xi in x) / n
    kurt /= std_dev ** 4
    kurt -= 3
    return round(kurt, 2)