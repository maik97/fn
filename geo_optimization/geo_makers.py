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


def pretzel_knot_init(radius, length: float = 4.0, puffer=1.0, num_p=100):
    d = 2 * radius
    lp = length / 2 - puffer
    lf = length / 2
    #x1 = np.array([-lf, -lp, 0, lp, lp, 0, -lp, -lp, 0, lp, lf])
    x1 = np.array([-lf, -lp, 0, lp, lp+radius, 0, -lp-radius, -lp, 0, lp, lf])
    x2 = np.array([0, 0, d, d, -d, -d, -d, d, d, 0, 0])
    x3 = np.array([0, 0, d, 0, -d, 0, d, 0, -d, 0, 0])
    print(x1, x2, x3)
    return resample(x1, num_p), resample(x2, num_p), resample(x3, num_p)


def neuron_connection_init(circles: list, height: float = 5.0, puffer=1.0, num_p: int = 10):
    c_1, c_2 = circles

    x1 = [c_1.x,
          c_1.x + (c_2.x - c_1.x) * 0.1 * c_1.r,
          c_2.x - (c_2.x - c_1.x) * 0.1 * c_2.r,
          c_2.x]

    x2 = [c_1.y,
          c_1.y + (c_2.y - c_1.y) * 0.1 * c_1.r,
          c_2.y - (c_2.y - c_1.y) * 0.1 * c_2.r,
          c_2.y]

    x3 = [- 0.5 * height,
          - 0.5 * height + puffer,
          0.5 * height - puffer,
          0.5 * height]

    r_linear = interpolate.interp1d(
        np.array([-height, -(height / 2), (height / 2), height]),
        np.array([c_1.r, c_1.r, c_2.r, c_2.r]),
        kind='linear'
    )
    r_vals = np.linspace(-height, height, num_p)

    return resample(x1, num_p), resample(x2, num_p), resample(x3, num_p), r_linear(r_vals)

