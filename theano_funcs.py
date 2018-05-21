from theano.tensor import dmatrix, dot, vector
from theano import function

a = dmatrix("a")
b = vector("b")

_mul_out = dot(a, b)
_th_mul = function([a, b], _mul_out)


def theano_dot(x, y):
    return _th_mul(x, y)
