import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def cubic_4_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3):
        return b0*np.power(x, 3) + b1*np.power(x, 2) + b2*np.asarray(x) + b3

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([0., 0., 0., 0.])
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

    x_axis, curve, fitted = cubic_4_fitting_no_constraint(x, y)
    plt.plot(x, y_logistic, 'r-.', linewidth=3, label='logistic')
    plt.plot(x, y, 'go', linewidth=3, label='logistic+AWGN')
    plt.plot(x_axis, curve, 'k--', linewidth=3, label='cubic (4) no cons')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper left")
    plt.show()
