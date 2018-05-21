import numpy as np
from theano_funcs import theano_dot


def test_func(args0):
    assert isinstance(args0, np.ndarray), "Got {}".format(type(args0))
    eq1_coeff = np.array([2, 1, 3])
    eq2_coeff = np.array([8, -5, 3])
    return pow(np.matmul(args0, eq1_coeff) - 7, 2) + pow(np.matmul(args0, eq2_coeff) - 11, 2)


def solve_test1(x, y, z):
    return pow((2*x) + y + (3*z) - 7, 2) + pow((8*x) - (5*x) + (3*z) - 11, 2)\


def test_func2(args0):
    assert isinstance(args0, np.ndarray), "Got {}".format(type(args0))
    eq0_coeff = np.array([1, 0, 0, 0, 0, 0, 0])
    eq1_coeff = np.array([3, 1, 0, 0, 0, 0, 0])
    eq2_coeff = np.array([4, 3, 1, 0, 1, 0, 0])
    eq3_coeff = np.array([3, 4, 3, 1, 1, 1, 0])
    eq4_coeff = np.array([0, 3, 4, 3, 1, 1, 1])
    eq5_coeff = np.array([0, 0, 3, 4, 0, 1, 1])
    eq6_coeff = np.array([0, 0, 0, 3, 0, 0, 1])
    return pow(theano_dot(args0, eq0_coeff) - 1, 2) + \
           pow(theano_dot(args0, eq1_coeff) - 6, 2) + \
           pow(theano_dot(args0, eq2_coeff) - 15, 2) + \
           pow(theano_dot(args0, eq3_coeff) - 20, 2) + \
           pow(theano_dot(args0, eq4_coeff) - 15, 2) + \
           pow(theano_dot(args0, eq5_coeff) - 6, 2) + \
           pow(theano_dot(args0, eq6_coeff) - 1, 2)


def majic_func(args0):
    eval = []
    for particle in args0:
        h, j, l, m, n, o, q, r, s, t, v, w, C, D, M, N = particle
        eval_particle = [
            -N - 8 * (D - M + j + o + t) * (-M + h + m + r + w) * (
                    2 * C + 2 * D - 2 * M + j - m + n + o - q + s + t + v + w) * (
                    2 * C + 2 * D - 3 * M + h + 2 * j + l - m + n + 2 * o + r + 2 * t + 2 * v + 2 * w) * (
                    2 * C + 4 * D - 3 * M + h + 2 * j - l + m + n + 2 * o - 2 * q + r + 2 * s + 2 * t + 2 * w),
            -N - 8 * h * j * (-M - h - 2 * j + l + 3 * m + n + 2 * q + r + 2 * s) * (
                    C + 2 * D - M + j - m + o - q + t + v + w) * (
                    2 * C + 4 * D - 5 * M + h + 2 * j + l + m + n + 2 * o + r + 2 * s + 2 * t + 2 * v + 2 * w),
            -N - 32 * l * m * n * o * (-M + l + m + n + o),
            -N - 32 * q * r * s * t * (-M + q + r + s + t),
            -32 * C * D * v * w * (C + D - M + v + w) - N,
            -N - 8 * (C + D - M + v + w) * (-M + l + m + n + o) * (-M + q + r + s + t) * (
                    -M - h - 2 * j + l + 3 * m + n + 2 * q + r + 2 * s) * (
                    2 * C + 2 * D - 3 * M + h + 2 * j + l - m + n + 2 * o + r + 2 * t + 2 * v + 2 * w),
            -N - 8 * l * q * v * (
                    2 * C + 4 * D - 5 * M + h + 2 * j + l + m + n + 2 * o + r + 2 * s + 2 * t + 2 * v + 2 * w) * (
                    2 * C + 4 * D - 3 * M + h + 2 * j - l + m + n + 2 * o - 2 * q + r + 2 * s + 2 * t + 2 * w),
            -N - 32 * h * m * r * w * (-M + h + m + r + w),
            -32 * C * n * s * (C + 2 * D - M + j - m + o - q + t + v + w) * (
                    2 * C + 2 * D - 2 * M + j - m + n + o - q + s + t + v + w) - N,
            -32 * D * j * o * t * (D - M + j + o + t) - N,
            -N + 32 * m * q * (C + D - M + v + w) * (D - M + j + o + t) * (C + 2 * D - M + j - m + o - q + t + v + w),
            -8 * D * m * s * (2 * C + 2 * D - 3 * M + h + 2 * j + l - m + n + 2 * o + r + 2 * t + 2 * v + 2 * w) * (
                    2 * C + 4 * D - 5 * M + h + 2 * j + l + m + n + 2 * o + r + 2 * s + 2 * t + 2 * v + 2 * w) - N
        ]
        eval += [sum([pow(x, 2) for x in eval_particle])]
    return np.array(eval)


if __name__ == "__main__":
    a = np.asarray([[1, 2, 3]])
    print(test_func(a))
    print(solve_test1(6, 16,  6))
    print(majic_func(np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])))

