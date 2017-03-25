import plotly
from plotly.graph_objs import Scatter, Layout
import numpy as np
import sympy

plotly.offline.init_notebook_mode()
x = sympy.Symbol('x')

sympy.init_printing()

start = 1 # start
end = 4 # end
degree = 2 # degree of polynomial

error_on_iteration = 0 # error on each iteration
precision = 1e-2 # precision

alternance = [start + (end-start) * k / float(degree + 1) for k in range(degree+2)]

alternance # first alternance

# f = np.e**x # function to approximate
f = sympy.log(x)

def make_eq(coefs, point):
    _f = sympy.lambdify(x, f)
    eq = _f(point)
    for i, c in enumerate(coefs):
        eq -= c*point**i
    return eq

def pol(t):
    global error_on_iteration
    e = sympy.Symbol('e')
    vars_str = ' '.join(['a' + str(i) for i in range(degree+1)])
    variables = sympy.symbols(vars_str)
    eqs = []

    for i in range(degree+2):
        eqs.append(make_eq(variables, t[i]) + e)
        e *= -1
    if (degree + 2) % 2 == 1:
        e *= -1

    solution = sympy.solve(eqs, variables + (e,))

    error_on_iteration = solution[e]
    polynom = x - x
    for i, v in enumerate(variables):
        polynom += solution[v] * x**i

    return polynom

def max_error():
    polyn = pol(alternance)
    err_fun = np.vectorize(sympy.lambdify(x, f - polyn))
    x_vals = np.linspace(start, end, (end - start) * 1000) # x values to check for maximum
    y_vals = err_fun(x_vals)
    
    neg_err = min(y_vals)
    pos_err = max(y_vals)
    
    if abs(neg_err) > pos_err:
        e_max = neg_err
    else:
        e_max = pos_err
    return e_max

def x_of_max_error():
    polyn = pol(alternance)
    err_fun = np.vectorize(sympy.lambdify(x, f - polyn))
    x_vals = np.linspace(start, end, (end - start) * 10000) # x values to check for maximum
    y_vals = err_fun(x_vals)
    
    absolute_y_vals = list(map(lambda x: abs(x), y_vals))
    e_max = max(absolute_y_vals)

    i = list(absolute_y_vals).index(e_max) # index of max error

    return x_vals[i]


def error():
    return np.vectorize(sympy.lambdify(x, f - pol(alternance)))

def plot_error_function(plot_max_err=False, title="Error"):
    x = sympy.Symbol('x')
    _f = np.vectorize(sympy.lambdify(x, f))
    p = np.vectorize(sympy.lambdify(x, pol(alternance)))
    x_vals = np.linspace(start, end, (end - start) * 1000)
    

    if plot_max_err == False:
        data = [Scatter(x=x_vals, y = _f(x_vals) - p(x_vals))]

    else:
        y_err = max_error()
        x_err = x_of_max_error()
        data = [Scatter(x=x_vals, y = _f(x_vals) - p(x_vals),  name="Error"),
                Scatter(x=[x_err for i in range(100)], y=np.linspace(0, y_err, 100), name="Max error")]
        
    plotly.offline.iplot({
        "data": data,
        "layout": Layout(title=title)
    })

plot_error_function(plot_max_err=True)


def plot_approximation(plot_max_error=False):
    
    x = sympy.Symbol('x')
    _f = np.vectorize(sympy.lambdify(x, f))
    p = np.vectorize(sympy.lambdify(x, pol(alternance)))
    x_vals = np.linspace(start, end, (end - start) * 1000)
    data = [Scatter(x=x_vals, y=_f(x_vals), name='f(x)'), Scatter(x = x_vals, y = p(x_vals), name='P(x)')]
    
    if plot_max_error == True:
        y_err = max_error()
        x_err = x_of_max_error()
        data.append(Scatter(x=[x_err for i in range(100)], y=np.linspace(_f(x_err), p(x_err), 100), name='Error'))
        
    plotly.offline.iplot({
        "data": data,
        "layout": Layout(title="Function and approximation")
    })

plot_approximation(True)

def sign(x):
    if x > 0: return '+'
    elif x < 0: return '-'
    else: return 0

sign = np.vectorize(sign)

def change_alternance(): # change alternance
    global alternance
    x_err = x_of_max_error()
    temp = alternance[:]
    temp.append(x_err)
    temp.sort()
    index_of_x_err = temp.index(x_err)
    if index_of_x_err != 0 and index_of_x_err != (len(temp)-1):
        if sign(error()(temp[index_of_x_err])) == sign(error()(temp[index_of_x_err-1])):
            del temp[index_of_x_err-1]
    
        else: del temp[index_of_x_err+1]
        
        alternance = temp[:]
    else: print('Index {}'.format(index_of_x_err))


alternance = [start + (end-start) * k / float(degree + 1) for k in range(degree+2)]
iterations = 1
while abs(abs(max_error()) - abs(error_on_iteration)) / abs(error_on_iteration) > precision:
    print('Alternance before: {}'.format(alternance))
    print('Signs of alternance: {}'.format(sign(error()(alternance))))
    print('Max error: {:.5f}'.format(max_error()))
    print('Error on iteration: {:.5f}'.format(error_on_iteration))
    print('Error in each point of alternance: {}', error()(alternance))
    print('X in which max error: {:.5f}'.format(x_of_max_error()))
    change_alternance()
    
    print('Alternance before: {}'.format(alternance))
    print('Error in each point of alternance: {}', error()(alternance))
    plot_error_function(True)

    print('\n\n')
    iterations += 1

print('Max error: {}'.format(max_error()))
print('Error on iteration: {}'.format(error_on_iteration))
print('Difference of errors: {}'.format(abs(abs(max_error()) - abs(error_on_iteration)) / abs(error_on_iteration)))
print('Iterations: {}'.format(iterations))