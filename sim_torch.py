import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

import torch as th
from torch import nn
from torch.optim import Adam

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg

import pylab

from sim_stuff import layer_planes, plot_circle_pack

from functional import InterpolatingValues, AxisValues
from losses import WeightedEuclideanDistance, WeightedDirection, noise_reduction_loss, PufferZone

from render_sim import Visualizer


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

        self.weighted_cdist = WeightedEuclideanDistance(self.num_points)
        self.weighted_dir = WeightedDirection(self.points)
        self.pdist = nn.PairwiseDistance(p=2)
        self.y_vals = th.tensor(y)
        self.puffer_fix = PufferZone(
            self.points,
            - 0.5 * height + puffer,
            0.5 * height - puffer,
            self.y_vals
        )

    def make_points(self, num):
        y = self.y(num)
        x = self.x(y)
        z = self.z(y)
        r = self.r(y)
        return x, y, z, r

    def forward(self, other_connections, scale=1.0):
        fixed_loss = self.puffer_fix(self.points)

        '''intersect_loss = th.tensor(0.0)
        for o_c in other_connections:
            min_dist = (self.radii + o_c.radii) * scale
            eucl_dist = self.pdist(self.points, o_c.points)
            intersect_loss = intersect_loss + th.sum(th.relu(min_dist - eucl_dist))'''

        ps = th.transpose(self.points,-2, -1)
        ps = th.cat([ps, th.unsqueeze(self.y_vals, 0)]).transpose(-2, -1)

        intersect_loss = th.tensor(0.0)
        for o_c in other_connections:
            min_dist = (self.radii + o_c.radii) * scale
            ops = th.transpose(o_c.points,-2, -1)
            ops = th.cat([ops, th.unsqueeze(self.y_vals, 0)]).transpose(-2, -1)
            ops = ops.reshape(1, len(ops), 1, -1)
            eucl_dist = self.pdist(ps, ops)
            intersect_loss = intersect_loss + th.sum(th.relu(min_dist - eucl_dist))

        dist_loss = th.sum(self.weighted_cdist(ps, ps))
        dir_loss = th.sum(self.weighted_dir(ps))

        noise_loss = noise_reduction_loss(self.points)
        return fixed_loss, intersect_loss, dist_loss, dir_loss, noise_loss

    def plot_to_fig(self, ax):
        points = self.points.clone().detach().numpy()
        p_t = np.transpose(points)
        ax.plot3D(p_t[0], p_t[1], self.y(len(points)))

    def smooth_interpolate(self):
        with th.no_grad():
            p = self.points.clone().detach().numpy()
            p_t = np.transpose(p)
            x, z = p_t[0], p_t[1]
            for i in range(1):
                x = savgol_filter(x, 11, 3)
                z = savgol_filter(z, 11, 3)
            p = np.transpose([x, z])
            p = th.tensor(p)
            self.points = nn.Parameter(p)


class ConnectionSim:

    def __init__(self, num_points, lr):
        self.l_plane = layer_planes(weighted_radius=1.0, equal_radius=1.0)
        self.l_test = nn.ModuleList()
        for l in self.l_plane:
            self.l_test.append(NeuralConnection(l, num_points=num_points))

        self.optimizer = Adam(self.l_test.parameters(), lr=lr)
        self.visualizer = Visualizer()

    def plot_planes(self):
        pl_0 = []
        pl_1 = []
        for l in self.l_plane:
            pl_0.append(l[0])
            pl_1.append(l[1])
        plot_circle_pack(pl_0)
        plot_circle_pack(pl_1)

    def render(self):
        fig = self.plot()
        self.visualizer.render_step(fig)
        plt.close('all')

    def step(self, epochs, scale, visualize=True):

        if visualize:
            self.render()

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

            fixed_loss = th.sum(th.stack(fixed_losses))
            intersect_loss = th.sum(th.stack(intersect_losses)) * 10
            dist_loss = th.sum(th.stack(dist_losses)) / 100000
            dir_loss = th.sum(th.stack(dir_losses)) / 10000
            noise_loss = th.sum(th.stack(noise_losses)) * 20

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
            if visualize:
                self.render()

        self.plot()
        plt.show()

    def plot(self):
        fig = pylab.figure(figsize=[10, 10],  # Inches
                           dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca(projection='3d')
        for l in self.l_test:
            l.plot_to_fig(ax)
        return fig

    def smooth(self):
        for l in self.l_test:
            l.smooth_interpolate()


connect_sim = ConnectionSim(100, 0.1)
#connect_sim.plot()
connect_sim.step(epochs=200_000, scale=0.7)
#connect_sim.plot()




