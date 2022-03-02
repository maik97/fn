from scipy.signal import savgol_filter

import torch as th
from torch import nn


def noise_reduction_loss(vecs, window_length=11, polyorder=3):
    smooth = vecs.clone().detach().transpose(-2, -1).numpy()
    smooth_x1 = savgol_filter(smooth[0], window_length=window_length, polyorder=polyorder)
    smooth_x2 = savgol_filter(smooth[1], window_length=window_length, polyorder=polyorder)
    smooth = th.tensor([smooth_x1, smooth_x2]).transpose(-2, -1)
    return (vecs - smooth)


def arg_dist(vec_len):
    a_d = []
    for i in range(vec_len):
        for j in range(vec_len):
            i_j = 1 - (abs(i - j) / vec_len)
            a_d.append(i_j)
    return th.tensor(a_d).reshape(vec_len, vec_len)


class WeightedEuclideanDistance:

    def __init__(self, vecs):
        self.W = arg_dist(len(vecs)).sqrt()
        self.pdist = nn.PairwiseDistance(p=2)

    def __call__(self, x1, x2):
        x2 = x2.reshape(1, len(x2), 1, -1)
        dist = self.pdist(x1, x2)
        return dist * self.W


class WeightedDirection:

    def __init__(self, vecs):
        self.W = arg_dist(len(vecs) - 1).sqrt()

    def __call__(self, vec_path):
        directions = vec_path[:-1] - vec_path[1:]
        nVs = directions / th.norm(directions, p=2, dim=-1, keepdim=True)
        reshaped_nVs = nVs.reshape(1, len(nVs), 1, 3)
        diff = reshaped_nVs - nVs
        diff = th.sum(th.abs(diff), dim=-1)
        return diff * self.W
