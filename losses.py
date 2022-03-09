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


def arg_dist(vec_len):
    a_d = []
    for i in range(vec_len):
        for j in range(vec_len):
            i_j = 1 - (abs(i - j) / vec_len)
            a_d.append(i_j)
    return th.tensor(a_d).reshape(vec_len, vec_len)


class WeightedEuclideanDistance:

    def __init__(self, vec_len):
        self.W = arg_dist(vec_len)*2
        #self.pdist = nn.PairwiseDistance(p=2)

    def __call__(self, x1, x2):
        #x2 = x2.reshape(1, len(x2), 1, -1)
        dist = th.cdist(x1, x2)
        #dist = self.pdist(x1, x2)
        return th.square(dist * self.W)


class WeightedDirection:

    def __init__(self, vecs):

        self.W = arg_dist(len(vecs) - 1)*2

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

        print(puffer_0)
        print(puffer_1)
        print(y_vals)
        print(self.ids_0)
        print(self.ids_1)

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


class MaskedSelfRepel:

    def __init__(self, dist_matrix, min_dist_matrix):

        self.mask_ranges = self.make_mask_ranges(dist_matrix, min_dist_matrix)


    def make_mask_ranges(self, dist_matrix, min_dist_matrix):
        dist_matrix = dist_matrix.clone()

        mask_ranges = []
        for i in range(len(dist_matrix)):
            m_range_0 = 0
            m_range_1 = len(dist_matrix)

            for j in range(i, m_range_0):
                if dist_matrix[i , j] > min_dist_matrix[i , j]:
                    m_range_0 = j
                    break

            for j in range(i, m_range_1):
                if dist_matrix[i , j] > min_dist_matrix[i , j]:
                    m_range_1 = j
                    break

            mask_ranges.append([m_range_0, m_range_1])
        return np.array(mask_ranges)




