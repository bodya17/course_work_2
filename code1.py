import numpy as np
from sympy import simplify, lambdify, Symbol, latex, Matrix

x = Symbol('x')

def genFuncs(deg):
    x = Symbol('x')
    funcs = []
    for i, el in enumerate(range(deg+1)):
        funcs.append(x**i)
    return funcs[::-1]

def makePol(coefs):
    deg = len(coefs) - 1
    x = Symbol('x')
    expr = coefs[0]
    for i, el in enumerate(range(deg)):
        expr = expr * x + coefs[i+1]
        print(expr)
    return expr.expand()

def max_error(f, a, b): # f is sympy expression
    _f = np.vectorize(lambdify(x, f))
    x_vals = np.linspace(a, b, (b - a) * 10000) # x values to check for
    y_vals = _f(x_vals)

    neg_err = min(y_vals)
    pos_err = max(y_vals)

    if abs(neg_err) > pos_err: e_max = neg_err
    else: e_max = pos_err
    return e_max

def x_of_max_error(f, a, b):
    _f = np.vectorize(lambdify(x, f))
    x_vals = np.linspace(a, b, (b - a) * 10000) # x values to check for maximum
    y_vals = _f(x_vals)

    absolute_y_vals = list(map(lambda x: abs(x), y_vals))
    e_max = max(absolute_y_vals)

    i = list(absolute_y_vals).index(e_max) # index of max error
    return x_vals[i]

def lssq(x_vals, y_vals, deg):
    f = genFuncs(deg)
    A = []
    B = Matrix(y_vals)
    for i in x_vals:
        A.append(list(map(lambda el: el.subs(x, i), f)))
    A = Matrix(A)
    return ((A.T*A).inv() * (A.T*B)).T


def main(fun, degree, start, end, points_ctn):
     f = np.vectorize(lambdify(x, simplify(fun)))

     x_vals = np.linspace(start, end, points_ctn)
     y_vals = f(x_vals)

     approximation = makePol( list(lssq(x_vals, y_vals, degree)) )
     f_approx = np.vectorize(lambdify(x, simplify(approximation)))

     x_approx = np.linspace(start, end, 100)

     x_err = x_of_max_error(approximation - simplify(fun), start, end)

     return {
        'formula': latex(approximation),
        'x_vals': list(x_vals),
        'y_vals': list(y_vals),
        'f_x_approx': list(f(x_approx)),
        'x_approx': list(x_approx),
        'approximation': list(f_approx(x_approx)),
        'max_error': max_error(approximation - simplify(fun), start, end),
        'x_of_max_error': x_err,
        'max_error_line': {
            'x': [x_err for i in range(100)],
            'y': list(np.linspace(f_approx(x_err), f(x_err), 100))
        }
    }