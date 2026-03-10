import numpy as np
from scipy.stats import linregress

def rmse(diff):
    return np.sqrt(np.mean(diff**2))


def add_quad(data):
    return np.sqrt((data**2).sum())


def comparison_stats(xdata, ydata):
    m, b, r, p, err = linregress(xdata.flatten(), ydata.flatten())
    bias = (ydata - xdata).mean()
    return m, b, r, bias


def rma_modified(xdata, ydata):
    m, _, _, _ = np.linalg.lstsq(xdata.reshape((-1, 1)),
                                 ydata.reshape((-1, 1)), rcond=None)
    slope = np.sign(m[0][0])*ydata.std()/xdata.std()
    return slope


def rma(xdata, ydata):
    print(xdata.shape, ydata.shape)
    _, _, r, _ = comparison_stats(xdata, ydata)
    slope = np.sign(r)*ydata.std()/xdata.std()
    return slope


def rel_err(data, truth):
    return (data - truth)/truth