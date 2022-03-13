import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter


def smooth_interpolation(x, y, z, num=25, window_length=11):
    x_spline = interpolate.interp1d(y, x, kind='cubic')
    z_spline = interpolate.interp1d(y, z, kind='cubic')

    y_new = np.linspace(y[0], y[-1], num=num)
    x_new = np.clip(x_spline(y_new), np.min(x), np.max(x))
    z_new = np.clip(z_spline(y_new), np.min(z), np.max(z))

    x_new = savgol_filter(x_new, window_length, 3)
    z_new = savgol_filter(z_new, window_length, 3)

    return x_new, y_new, z_new


class InterpolatingValues:

    def __init__(self, x, y, kind='cubic'):

        self.min = np.min(y)
        self.max = np.max(y)

        self.x_min = np.min(x)
        self.x_max = np.max(x)

        self.update_spline(x, y, kind)

    def update_spline(self, x, y, kind='cubic'):
        self.spline = interpolate.interp1d(x, y, kind=kind)

    def smoothing(self, steps=1, num=25, window_length=11, polyorder=3):
        x = np.linspace(self.x_min, self.x_max, num=num)
        y = self(x)
        for _ in range(steps):
            y = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        self.update_spline(x, y)

    def __call__(self, x):
        y = self.spline(x)
        return np.clip(y, self.min, self.max)


class AxisValues:

    def __init__(self, x):
        self.min = np.min(x)
        self.max = np.max(x)

    def __call__(self, num):
        return np.linspace(self.min, self.max, num=num)