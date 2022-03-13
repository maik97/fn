import numpy as np

from scipy.signal import savgol_filter

import torch as th
from torch import nn


def noise_reduction_loss(points_2d, window_length=11, polyorder=3):
    smooth = points_2d.clone().detach().transpose(-2, -1).numpy()
    smooth_x1 = savgol_filter(smooth[0], window_length=window_length, polyorder=polyorder)
    smooth_x2 = savgol_filter(smooth[1], window_length=window_length, polyorder=polyorder)
    smooth = th.tensor([smooth_x1, smooth_x2]).transpose(-2, -1)
    return th.abs(points_2d - smooth)


def arg_dist(num_p):
    a_d = []
    for i in range(num_p):
        for j in range(num_p):
            i_j = 1 - (abs(i - j) / num_p)
            a_d.append(i_j)
    return th.tensor(a_d).reshape(num_p, num_p)


class WeightedEuclideanDistance:

    def __init__(self, num_p, w_scale=2.0, w_bias=0.0):
        self.W = arg_dist(num_p) * w_scale + w_bias

    def __call__(self, x1, x2):
        dist = th.cdist(x1, x2)
        #dist = dist.clone()
        #dist = dist / th.max(dist)
        return th.square(dist * self.W)


class WeightedDirection:

    def __init__(self, num_p, w_scale=2.0, w_bias=0.0):
        self.W = arg_dist(num_p - 1) * w_scale + w_bias

    def __call__(self, vec_path):
        directions = vec_path[:-1] - vec_path[1:]
        nVs = directions / th.norm(directions, p=2, dim=-1, keepdim=True)
        reshaped_nVs = nVs.reshape(1, len(nVs), 1, 3)
        diff = reshaped_nVs - nVs
        diff = th.sum(th.abs(diff), dim=-1)
        return th.square(diff * self.W)


def group_connected_indices(indices_arr):
    grouped = []
    cur_i = 0
    for i in range(len(indices_arr) - 1):
        if indices_arr[i] + 1 != indices_arr[i + 1]:
            grouped.append(indices_arr[cur_i:i])
            cur_i = i + 1
    grouped.append(indices_arr[cur_i:])
    return grouped


class PufferZone:

    def __init__(self, points_2d, length, puffer, height_points):
        lp = length / 2 - puffer
        points_2d = points_2d.clone().detach().numpy()
        height_points = height_points.clone().detach().numpy()

        self.ids_0 = group_connected_indices(np.argwhere(height_points < -lp))[0]
        self.ids_1 = group_connected_indices(np.argwhere(height_points > lp))[-1]

        self.compare_0 = th.squeeze(th.tensor(points_2d[self.ids_0]))
        self.compare_1 = th.squeeze(th.tensor(points_2d[self.ids_1]))

        self.ids_0 = len(self.ids_0)
        self.ids_1 = len(self.ids_1)

        self.weight_0 = th.tensor(np.linspace(1000, 1, self.ids_0))
        self.weight_1 = th.tensor(np.linspace(1, 1000, self.ids_1))

    def __call__(self, points_2d):
        loss_0 = th.square(th.abs(points_2d[:self.ids_0] - self.compare_0)).sum(-1)
        loss_1 = th.square(th.abs(points_2d[-self.ids_1:] - self.compare_1)).sum(-1)
        return th.sum(self.weight_0*loss_0) + th.sum(self.weight_1*loss_1)

    def force_constraints(self, points_2d):
        points_2d[:self.ids_0] = self.compare_0
        points_2d[-self.ids_1:] = self.compare_1
        return points_2d

def masked_intersect_alt(dist_matrix, min_dist_matrix):
    dist_matrix = dist_matrix.clone()
    for i in range(len(dist_matrix)):

        for j in reversed(range(0, i)):
            if dist_matrix[i, j] <= min_dist_matrix[i, j]:
                dist_matrix[i, j] = min_dist_matrix[i, j] + 1
            else:
                break

        for j in range(i, len(dist_matrix)):
            if dist_matrix[i, j] <= min_dist_matrix[i, j]:
                dist_matrix[i, j] = min_dist_matrix[i, j] + 1
            else:
                break

        if i != len(dist_matrix) - 1:
            dist_matrix[i, i+1] = min_dist_matrix[i, i+1] + 1
        if i != 0:
            dist_matrix[i, i-1] = min_dist_matrix[i, i-1] + 1

    return th.square(th.relu(min_dist_matrix - dist_matrix)*2)#

def masked_intersect(dist_matrix, min_dist_matrix, n_neighbours=5):
    dist_matrix = dist_matrix.clone()
    for i in range(len(dist_matrix)):

        for j in reversed(range(0, i)):
            if dist_matrix[i, j] <= min_dist_matrix[i, j]:
                dist_matrix[i, j] = min_dist_matrix[i, j] + 1
            else:
                break

        for j in range(i, len(dist_matrix)):
            if dist_matrix[i, j] <= min_dist_matrix[i, j]:
                dist_matrix[i, j] = min_dist_matrix[i, j] + 1
            else:
                break

        for j in range(n_neighbours):
            if i+j < len(dist_matrix) - 1:
                dist_matrix[i, i+j+1] = min_dist_matrix[i, i+j+1] + 1
            if i-j > 0:
                dist_matrix[i, i-j-1] = min_dist_matrix[i, i-j-1] + 1

    return th.square(th.relu(min_dist_matrix - dist_matrix)*2)#


class MaskedSelfRepel:

    def __init__(self, dist_matrix, min_dist_matrix):

        print('dist_matrix_0')
        print(dist_matrix)
        print('min_dist_matrix_0')
        print(min_dist_matrix)
        #print(masked_dist_matrix(dist_matrix, min_dist_matrix))


        exit()

        self.mask_ranges = self.make_mask_ranges(dist_matrix, min_dist_matrix)

        print(self.mask_ranges)
        print(self.make_mask_matrix(self.mask_ranges))


    def make_mask_ranges(self, dist_matrix, min_dist_matrix):
        dist_matrix = dist_matrix.clone()

        mask_ranges = []
        for i in range(len(dist_matrix)):
            m_range_0 = 0
            m_range_1 = len(dist_matrix)

            for j in reversed(range(m_range_0, i)):
                print(j)
                if dist_matrix[i , j] > min_dist_matrix[i , j]:
                    m_range_0 = j + 1
                    break

            for j in range(i, m_range_1):
                if dist_matrix[i , j] > min_dist_matrix[i , j]:
                    m_range_1 = j - 1
                    break

            mask_ranges.append([m_range_0, m_range_1])
        return np.array(mask_ranges)

    def make_mask_matrix(self, mask_ranges):

        ids_list = []
        for m_range in mask_ranges:
            ids_list.append(np.arange(m_range[0], m_range[1]+1))

        print(ids_list)


