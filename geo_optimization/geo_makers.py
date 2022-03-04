import numpy as np
from scipy import interpolate


def resample(vals, num_p, kind='linear'):
    if isinstance(vals, (int, float)):
        vals = [vals]*num_p
    l_space = np.linspace(0, 1, len(vals))
    vals_interp = interpolate.interp1d(l_space, vals, kind=kind)
    l_space = np.linspace(0, 1, num_p)
    return vals_interp(l_space)


def maybe_resample(val, num_p):
    if isinstance(val, (list, tuple, np.ndarray)):
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        if len(val) == num_p:
            return resample(val, num_p)
        return val
    else:
        return val


def circle(radius=1.0, factor_pi=1.0, num_p=100, x1=0.0, x2=0.0):
    radius = maybe_resample(radius, num_p)
    theta = np.linspace(0, factor_pi * np.pi, num_p)
    x1 = x1 + radius * np.cos(theta)
    x2 = x2 + radius * np.sin(theta)
    return x1, x2


def spiral(radius=1.0, factor_pi=1.0, height=1.0, num_p=100, x1=0.0, x2=0.0, x3=0.0):
    x1, x2 = circle(radius=radius, factor_pi=factor_pi, num_p=num_p, x1=x1, x2=x2)
    x3 = x3 + np.linspace(0, height, num_p)
    return x1, x2, x3


def pretzel_knot_init(length: float = 4.0, puffer=1.0, num_p=100):
    lp = length / 2 - puffer
    lf = length / 2
    x1 = np.array([-lf, -lp, 0, lp, lp, 0, -lp, -lp, 0, lp, lf])
    x2 = np.array([0, 0, 1, 0, -1, 0, 1, 0, -1, 0, 0])
    x3 = np.array([0, 0, 1, 1, -1, -1, -1, 1, 1, 0, 0])
    return resample(x1, num_p), resample(x2, num_p), resample(x3, num_p)
