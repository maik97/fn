import numpy as np

from scipy.signal import savgol_filter

import torch as th
from torch import nn


def noise_reduction_loss(vecs, window_length=11, polyorder=3):
    smooth = vecs.clone().detach().transpose(-2, -1).numpy()
    smooth_x1 = savgol_filter(smooth[0], window_length=window_length, polyorder=polyorder)
    smooth_x2 = savgol_filter(smooth[1], window_length=window_length, polyorder=polyorder)
    smooth = th.tensor([smooth_x1, smooth_x2]).transpose(-2, -1)
    return th.abs(vecs - smooth)


def arg_dist(num_p):
    a_d = []
    for i in range(num_p):
        for j in range(num_p):
            i_j = 1 - (abs(i - j) / num_p)
            a_d.append(i_j)
    return th.tensor(a_d).reshape(num_p, num_p)


class WeightedEuclideanDistance:

    def __init__(self, num_p, w_scale=1.0, w_bias=0.0):
        self.W = arg_dist(num_p) * w_scale + w_bias

    def __call__(self, x1, x2):
        dist = th.cdist(x1, x2)
        return th.square(dist * self.W)


class WeightedDirection:

    def __init__(self, num_p, w_scale=1.0, w_bias=0.0):
        self.W = arg_dist(num_p - 1) * w_scale + w_bias

    def __call__(self, vec_path):
        directions = vec_path[:-1] - vec_path[1:]
        nVs = directions / th.norm(directions, p=2, dim=-1, keepdim=True)
        reshaped_nVs = nVs.reshape(1, len(nVs), 1, 3)
        diff = reshaped_nVs - nVs
        diff = th.sum(th.abs(diff), dim=-1)
        return th.square(diff * self.W)


class PufferZone:

    def __init__(self, vecs, puffer_0, puffer_1, y_vals):
        vecs = vecs.clone().detach().numpy()
        y_vals = y_vals.clone().detach().numpy()

        self.ids_0 = np.argwhere(y_vals < puffer_0)
        self.ids_1 = np.argwhere(y_vals > puffer_1)

        self.compare_0 = th.squeeze(th.tensor(vecs[self.ids_0]))
        self.compare_1 = th.squeeze(th.tensor(vecs[self.ids_1]))

        self.ids_0 = len(self.ids_0)
        self.ids_1 = len(self.ids_1)

        self.weight_0 = th.tensor(np.linspace(1000, 1, self.ids_0))
        self.weight_1 = th.tensor(np.linspace(1, 1000, self.ids_1))

    def __call__(self, vecs):
        loss_0 = th.square(th.abs(vecs[:self.ids_0] - self.compare_0)).sum(-1)
        loss_1 = th.square(th.abs(vecs[-self.ids_1:] - self.compare_1)).sum(-1)
        return th.sum(self.weight_0*loss_0) + th.sum(self.weight_1*loss_1)
