import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def logistic_5_fitting(x, y):
    def func(b, x):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b[1] * (x - b[2])))
        y_hat = b[0] * logistic_part + b[3] * np.asarray(x) + b[4]
        return y_hat

    def objective(b):
        return np.sum(np.power(func(b, x) - y, 2))

    def const_1st_derivative(b):
        exp_part = np.exp(b[1] * (x - b[2]))
        return b[3] + np.divide(b[0] * b[1] * exp_part, np.power(1 + exp_part, 2))

    cons = (dict(type='ineq', fun=const_1st_derivative))
    # init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    init = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    res = minimize(objective, x0=init, method='SLSQP', constraints=cons)

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    if res.success:
        curve = func(res.x, x_axis)
        fitted = func(res.x, x)
    else:
        print('fail, logistic (5) curving fitting')
        curve = x_axis
        fitted = x

    return x_axis, curve, fitted


if __name__ == '__main__':

    x = np.linspace(0, 100, 100)
    x = (x - 50.0) / 10
    y_logistic = scipy.special.expit(x)
    noise = np.random.normal(0, 0.8, 100)
    y = y_logistic + noise

    x_axis, curve, fitted = logistic_5_fitting(x, y)
    plt.plot(x, y_logistic, 'r-.', linewidth=3, label='logistic')
    plt.plot(x, y, 'go', linewidth=3, label='logistic+AWGN')
    plt.plot(x_axis, curve, 'k--', linewidth=3, label='logistic (5) fitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper left")
    plt.show()
