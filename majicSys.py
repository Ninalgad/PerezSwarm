from sympy import Symbol, Matrix, init_printing, symbols, linsolve, simplify
from math import floor
from sympy.polys.polytools import is_zero_dimensional
from sympy import nonlinsolve as nls

init_printing(use_unicode=True)
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = \
    symbols('a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z', real=True)
N = Symbol("N", real=True) # Magic product
M = Symbol("M", real=True) # Magic sum
C = Symbol("C", real=True) # x
D = Symbol("D", real=True) # y

system = [-3 * M / 2 + h / 2 + j + l / 2 - m / 2 + n / 2 + o + r / 2 + t + v + w + x + y,
          -3 * M / 2 + h / 2 + j - l / 2 + m / 2 + n / 2 + o - q + r / 2 + s + t + w + x + 2 * y, M - h - m - r - w,
          2 * M - j + m - n - o + q - s - t - v - w - 2 * x - 2 * y, M - j - o - t - y,
          -M / 2 - h / 2 - j + l / 2 + 3 * m / 2 + n / 2 + q + r / 2 + s,
          5 * M / 2 - h / 2 - j - l / 2 - m / 2 - n / 2 - o - r / 2 - s - t - v - w - x - 2 * y, h,
          -M + j - m + o - q + t + v + w + x + 2 * y, j, M - l - m - n - o, l, m, n, o, M - q - r - s - t, q, r, s, t,
          M - v - w - x - y, v, w, x, y]
mul_sys = [
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
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        return False
    else:
        return x % m

def newton_sqrt(num):
    _x = num
    next_x = (_x + (num // _x)) // 2
    while abs(next_x - _x) >= 1:
        _x = next_x
        next_x = (_x + (num // _x)) // 2
    return _x


def newton_isqrt(num):
    return pow(newton_sqrt(num), 2) == num


def get_ts(particle):
    sigma, tau = 0, 0
    for num in particle:
        sigma += num
        tau += num ** 2
    return sigma, tau


def test_N_2(particle):
    sigma, tau = get_ts(particle)
    discriminant = pow(sigma, 2) - (5 * tau)
    return all([discriminant >= 0,
                newton_isqrt(discriminant)])


def test_N_1(particle, maj_prod):
    sigma, tau = get_ts(particle)
    mod_test = modinv(2 * maj_prod * sigma, 5)
    if mod_test:
        return ((((tau - (2 * maj_prod * sigma)) % 5) == 0) and
                (((tau * mod_test) % 5) == 1))
    return False


def solve_for_M():
    for equa in mul_sys:
        print(solve(equa, M))


if __name__ == "__main__":
    from sympy.solvers import solve
    import numpy as np
    from sympy import simplify
    solve_for_M()
    proposed = [1110, 1154, 2475, 202, 1644,
                       3042, 3740, 2750, 2686, 2083,
                       2248, 3111, 2954, 1378, 2989]
    mp = 100000
    while not (test_N_1(proposed, mp) and test_N_2(proposed)):
        proposed = np.random.randint(0, 4000, size=15)
        mp = np.random.randint(2000, 4000)
    print("done")
    sigma, tau = get_ts(proposed)
    discriminant = pow(sigma, 2) - (5 * tau)
    print("{}: {}\n\t{}\n\t{}".format(mp,
                                      proposed,
                                      (newton_sqrt(discriminant) - sigma) % 5,
                                      (- newton_sqrt(discriminant) - sigma) % 5))
    print(sum([pow(mp - prop, 2) for prop in proposed]))
    """
    big = 0
    for equa in mul_sys:
        big += pow(equa, 2)
    print(big)
    print(simplify(big))

    #

    """