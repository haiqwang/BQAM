import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def cubic_4_fitting(x, y):
    def func(b, x):
        return b[0]*np.power(x, 3) + b[1]*np.power(x, 2) + b[2]*np.asarray(x) + b[3]

    def objective(b):
        return np.sum(np.power(func(b, x) - y, 2))

    def const_1st_derivative(b):
        # return 1
        return 3*b[0]*np.power(x, 2) + 2*b[1]*np.asarray(x) + b[2]

    cons = (dict(type='ineq', fun=const_1st_derivative))
    init = np.array([0., 0., 0., 0.])
    res = minimize(objective, x0=init, method='SLSQP', constraints=cons)
    print(res)

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    if res.success:
        curve = func(res.x, x_axis)
        fitted = func(res.x, x)
    else:
        print('fail, cubic curving fitting')
        curve = x_axis
        fitted = x

    return x_axis, curve, fitted


if __name__ == '__main__':

    x = np.linspace(0, 100, 100)
    x = (x - 50.0) / 10
    y_logistic = scipy.special.expit(x)
    noise = np.random.normal(0, 0.8, 100)
    y = y_logistic + noise

    x_axis, curve, fitted = cubic_4_fitting(x, y)
    plt.plot(x, y_logistic, 'r-.', linewidth=3, label='logistic')
    plt.plot(x, y, 'go', linewidth=3, label='logistic+AWGN')
    plt.plot(x_axis, curve, 'k--', linewidth=3, label='cubic fitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper left")
    plt.show()
