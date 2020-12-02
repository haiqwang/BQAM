import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted


if __name__ == '__main__':

    x = np.linspace(0, 100, 100)
    x = (x - 50.0) / 10
    y_logistic = scipy.special.expit(x)
    noise = np.random.normal(0, 0.8, 100)
    y = y_logistic + noise

    x_axis, curve, fitted = logistic_5_fitting_no_constraint(x, y)
    plt.plot(x, y_logistic, 'r-.', linewidth=3, label='logistic')
    plt.plot(x, y, 'go', linewidth=3, label='logistic+AWGN')
    plt.plot(x_axis, curve, 'k--', linewidth=3, label='logistic (5) no cons')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper left")
    plt.show()
