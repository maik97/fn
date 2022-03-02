import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

import torch as th
from torch import nn
from torch.optim import Adam

import matplotlib.pyplot as plt

from sim_stuff import layer_planes, plot_circle_pack

from functional import InterpolatingValues, AxisValues
from losses import WeightedEuclideanDistance, WeightedDirection, noise_reduction_loss


class NeuralConnection(nn.Module):

    def __init__(self, circles: list, height: float = 5.0, num_points: int = 10, puffer=1.0):
        super(NeuralConnection, self).__init__()

        self.num_points = num_points

        c_1, c_2 = circles

        init_y = [- 0.5 * height,
                  - 0.5 * height + puffer,
                  0.5 * height - puffer,
                  0.5 * height]

        init_x = [c_1.x,
                  c_1.x + (c_2.x - c_1.x) * 0.1 * c_1.r,
                  c_2.x - (c_2.x - c_1.x) * 0.1 * c_2.r,
                  c_2.x]

        init_z = [c_1.y,
                  c_1.y + (c_2.y - c_1.y) * 0.1 * c_1.r,
                  c_2.y - (c_2.y - c_1.y) * 0.1 * c_2.r,
                  c_2.y]
        asdgasdf

        self.x = InterpolatingValues(init_y, init_x)
        self.z = InterpolatingValues(init_y, init_z)
        self.y = AxisValues(init_y)
        self.r = interpolate.interp1d(
            np.array([-height, -(height / 2), (height / 2), height]),
            np.array([c_1.r, c_1.r, c_2.r, c_2.r]),
            kind='linear'
        )

        x, y, z, r = self.make_points(self.num_points)

        self.p_bottom = th.tensor([c_1.x,c_1.y])
        self.p_top = th.tensor([c_2.x,c_2.y])
        self.radii = th.tensor(r)

        self.points = nn.Parameter(
            th.tensor([x, z]).transpose(-2, -1)
        )

        self.weighted_cdist = WeightedEuclideanDistance(self.points)
        self.weighted_dir = WeightedDirection(self.points)
        self.pdist = nn.PairwiseDistance(p=2)
        self.y_vals = th.tensor(y)

    def make_points(self, num):
        y = self.y(num)
        x = self.x(y)
        z = self.z(y)
        r = self.r(y)
        return x, y, z, r

    def forward(self, other_connections, scale=1.0):
        fixed_loss = th.sum(th.abs(self.p_bottom - self.points[0]))
        fixed_loss = fixed_loss + th.sum(th.abs(self.p_top - self.points[-1]))

        intersect_loss = th.tensor(0.0)
        for o_c in other_connections:
            min_dist = (self.radii + o_c.radii) * scale
            eucl_dist = self.pdist(self.points, o_c.points)
            intersect_loss = intersect_loss + th.sum(th.relu(min_dist - eucl_dist))

        ps = th.transpose(self.points,-2, -1)
        ps = th.cat([ps, th.unsqueeze(self.y_vals, 0)]).transpose(-2, -1)

        dist_loss = th.sum(self.weighted_cdist(ps, ps))
        dir_loss = th.sum(self.weighted_dir(ps))

        noise_loss = noise_reduction_loss(self.points)
        return fixed_loss, intersect_loss, dist_loss, dir_loss, noise_loss

    def plot_to_fig(self, ax):
        points = self.points.clone().detach().numpy()
        p_t = np.transpose(points)
        ax.plot3D(p_t[0], p_t[1], self.y(len(points)))

    def smooth_interpolate(self):
        p_t = np.transpose(self.points)
        x, z = p_t[0], p_t[1]
        for i in range(5):
            x = savgol_filter(x, 11, 3)
            z = savgol_filter(z, 11, 3)
        self.points = np.transpose([x, z])


class ConnectionSim:

    def __init__(self, num_points):
        self.l_plane = layer_planes(weighted_radius=1.0, equal_radius=1.0)
        self.l_test = nn.ModuleList()
        for l in self.l_plane:
            self.l_test.append(NeuralConnection(l, num_points=num_points))

        self.optimizer = Adam(self.l_test.parameters(), lr=0.001)

    def plot_planes(self):
        pl_0 = []
        pl_1 = []
        for l in self.l_plane:
            pl_0.append(l[0])
            pl_1.append(l[1])
        plot_circle_pack(pl_0)
        plot_circle_pack(pl_1)

    def step(self, epochs=2_000, scale=0.9):

        for e in range(epochs):

            self.optimizer.zero_grad()
            fixed_losses, intersect_losses, dist_losses, dir_losses, noise_losses  = [], [], [], [], []
            for i in range(len(self.l_test)):

                cur_l = self.l_test[i]
                others = [self.l_test[j] for j in range(len(self.l_test)) if j != i]
                fixed_loss, intersect_loss, dist_loss, dir_loss, noise_loss = cur_l(others, scale)

                fixed_losses.append(fixed_loss)
                intersect_losses.append(intersect_loss)
                dist_losses.append(dist_loss)
                dir_losses.append(dir_loss)
                noise_losses.append(noise_loss)

            fixed_loss = th.sum(th.stack(fixed_losses)) * 1000
            intersect_loss = th.sum(th.stack(intersect_losses))
            dist_loss = th.sum(th.stack(dist_losses)) / 10000
            dir_loss = th.sum(th.stack(dir_losses)) / 1000
            noise_loss = th.sum(th.stack(noise_losses)) / 2

            loss = fixed_loss + intersect_loss + dist_loss + dir_loss + noise_loss

            loss.backward()
            self.optimizer.step()

            print('Epoch:', e,
                  '- Loss:', np.round(loss.detach().numpy(), 3),
                  '- fixed:', np.round(fixed_loss.detach().numpy(), 3),
                  '- intersect:', np.round(intersect_loss.detach().numpy(), 3),
                  '- dist:', np.round(dist_loss.detach().numpy(), 3),
                  '- dir:', np.round(dir_loss.detach().numpy(), 3),
                  '- noise:', np.round(noise_loss.detach().numpy(), 3),
                  )

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for l in self.l_test:
            l.plot_to_fig(ax)
        plt.show()

    def smooth(self):
        for l in self.l_test:
            l.smooth_interpolate()


connect_sim = ConnectionSim(50)
#connect_sim.plot()
connect_sim.step()
connect_sim.plot()




