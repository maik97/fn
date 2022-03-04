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

from functional import InterpolatingValues, AxisValues
from losses import WeightedEuclideanDistance, WeightedDirection, noise_reduction_loss, PufferZone

from render_sim import Visualizer


class NeuronTurboModel(nn.Module):

    def __init__(self, num_points, radius: float = 0.5, height: float = 5.0, puffer=1.0):
        super(NeuronTurboModel, self).__init__()

        self.radius = radius

        x = np.array([-2,-1, 0, 1, 1, 0,-1,-1, 0, 1, 2])
        z = np.array([ 0, 0, 1, 0,-1, 0, 1, 0,-1, 0, 0])
        y = np.array([ 0, 0, 1, 1,-1,-1,-1, 1, 1, 0, 0])

        print(x, y, z)

        l_space = np.linspace(0, 1, len(x))
        x_interp = interpolate.interp1d(l_space, x)
        y_interp = interpolate.interp1d(l_space, y)
        z_interp = interpolate.interp1d(l_space, z)

        l_space = np.linspace(0, 1, num_points)
        x = x_interp(l_space)
        y = y_interp(l_space)
        z = z_interp(l_space)

        self.x = th.tensor(x)
        self.y = nn.Parameter(th.tensor(y))
        self.z = nn.Parameter(th.tensor(z))

        self.points_2d = th.cat([self.y, self.z]).reshape(2,-1).transpose(-2, -1)
        self.points_3d = th.cat([self.x, self.y, self.z]).reshape(3,-1).transpose(-2, -1)

        self.puffer_fix = PufferZone(self.points_2d, -1.0, 1.0, self.x)
        self.dist_loss = WeightedEuclideanDistance(num_points)
        self.dir_loss = WeightedDirection(self.points_3d)

    def mask_distances(self, dist_matrix, min_dist):
        dist_matrix = dist_matrix.clone()
        for i in range(len(dist_matrix)):

            for j in range(i, 0):
                if dist_matrix[i , j] <= min_dist:
                    dist_matrix[i, j] = min_dist + 1
                else:
                    break

            for j in range(i, len(dist_matrix)):
                if dist_matrix[i , j] <= min_dist:
                    dist_matrix[i, j] = min_dist + 1
                else:
                    break
        return dist_matrix

    def forward(self, scale=1.0):
        fixed_loss = self.puffer_fix(th.cat([self.y, self.z]).reshape(2,-1).transpose(-2, -1))

        p3d = th.cat([self.x, self.y, self.z]).reshape(3,-1).transpose(-2, -1)
        dist = th.cdist(p3d, p3d)
        dist_mask = self.mask_distances(dist, self.radius * 2 * scale)
        intersect = th.relu(self.radius * 2 * scale - dist_mask)
        p2d = th.cat([self.y, self.z]).reshape(2,-1).transpose(-2, -1)
        noise = noise_reduction_loss(p2d)
        direct = self.dir_loss(p3d)

        return th.sum(fixed_loss) + th.sum(self.dist_loss(p3d, p3d)) / 10_000 + th.sum(intersect) + th.sum(noise)*10# + th.sum(noise)

    def plot_to_fig(self, ax):
        #points = self.points_3d.clone().detach().numpy()
        #p_t = np.transpose(points)
        ax.plot3D(self.x.clone().detach().numpy(),
                  self.y.clone().detach().numpy(),
                  self.z.clone().detach().numpy())


class NeuronTurboSim:

    def __init__(self, num_points, lr):
        self.n_turbo = NeuronTurboModel(num_points)
        self.optimizer = Adam(self.n_turbo.parameters(), lr=lr)
        self.visualizer = Visualizer()

    def render(self):
        fig = self.plot()
        self.visualizer.render_step(fig)
        plt.close('all')

    def step(self, epochs, scale, visualize=True):

        if visualize:
            self.render()

        for e in range(epochs):

            self.optimizer.zero_grad()

            loss = self.n_turbo(scale)
            loss.backward()
            self.optimizer.step()

            print('Epoch:', e,
                  '- Loss:', np.round(loss.detach().numpy(), 3),
                  )
            if visualize:
                self.render()

        print(self.n_turbo.x)
        print(self.n_turbo.y)
        print(self.n_turbo.z)
        self.plot()
        plt.show()

    def plot(self):
        fig = pylab.figure(figsize=[10, 10],  # Inches
                           dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca(projection='3d')
        self.n_turbo.plot_to_fig(ax)
        return fig

    def smooth(self):
        self.n_turbo.smooth_interpolate()


connect_sim = NeuronTurboSim(100, 0.01)
#connect_sim.plot()
connect_sim.step(epochs=1_000, scale=1.0)
#connect_sim.plot()



'''num_p = int(100 * (puffer) / ((0.5 * height - 2 * puffer) * 2))

        init_y_0 = np.linspace(- 0.5 * height,
                  - 0.5 * height + puffer, num_p)
                  #- 0.5 * height + 2 * puffer]

        init_y_1 = np.linspace(0.5 * height - puffer,
                    0.5 * height, num_p)


        init_x_0 = np.linspace(0.0, 0.0, num_p)

        init_x_1 = np.linspace(0.0, 0.0, num_p)

        init_z_0 = np.linspace(0.0, 0.0, num_p)

        init_z_1 = np.linspace(0.0, 0.0, num_p)


        # Creating equally spaced 100 data in range 0 to 2*pi
        theta = np.linspace(0, 3 * np.pi, 100)

        # Setting radius
        radius = 5

        # Generating x and y data
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        y = np.linspace(- 0.5 * height + 2 * puffer,
                        0.5 * height - 2 * puffer,
                        100)'''

'''
#self.x = InterpolatingValues(init_y, init_x)
#self.z = InterpolatingValues(init_y, init_z)
#self.y = AxisValues(init_y)
self.radius = radius

x = np.concatenate([init_x_0, x, init_x_1])
y = np.concatenate([init_y_0, y, init_y_1])
z = np.concatenate([init_z_0, z, init_z_1])'''

'''for i in range(1):
    x = savgol_filter(x, window_length=11, polyorder=3)
    z = savgol_filter(z, window_length=11, polyorder=3)

#y_ax = AxisValues(y)
#x_spline = InterpolatingValues(y, x)
#z_spline = InterpolatingValues(y, z)

#y = y_ax(100)
#x = x_spline(y)
#z = z_spline(y)'''

'''cm_x, cm_y, cm_z = curve(2, 2, -height/2 + puffer, height/2 - puffer)
cm_x_1, cm_y_1, cm_z_1 = curve(
    radius=1,
    factor_pi=1.0,
    y_0=cm_y[-1],
    y_1=cm_y[-1],
    x=cm_x[-1]-1,
    z=cm_z[-1],
)
cm_y_2, cm_x_2, cm_z_2  = curve(
    radius=-1.0,
    factor_pi=0.5,
    y_0=cm_z[-1],
    y_1=cm_z[-1],
    x=cm_x[-1]+0.5,
    z=cm_y[-1]-1.5,
)

x = np.concatenate([np.flip(cm_x_2), np.flip(cm_x_1), cm_x, cm_x_1, cm_x_2])
y = np.concatenate([-np.flip(cm_y_2), -np.flip(cm_y_1), cm_y, cm_y_1, cm_y_2])
z = np.concatenate([-np.flip(cm_z_2), -np.flip(cm_z_1), cm_z, cm_z_1, cm_z_2])

#x,y,z = cm_x_1, cm_y_1, cm_z_1

print(cm_x_1, cm_y_1, cm_z_1)
print(x)
print(y)
print(z)'''

'''
        fig = plt.figure(figsize=[10, 10],  # Inches
                         dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                         )
        ax = fig.gca(projection='3d')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])


        ax.plot3D(x, y, z)

        plt.show()
                plt.close()

        plt.plot(x, y)
        plt.show()
        plt.close()
        plt.plot(z, y)
        plt.show()
        plt.close()
        plt.plot(x, z)
        plt.show()

        print(x,y,z)'''
