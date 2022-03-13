import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab

import torch as th
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from geo_optimization.geos import Geometry
from geo_optimization.geo_makers import resample, neuron_connection_init
from geo_optimization.geo_losses import PufferZone, WeightedEuclideanDistance, WeightedDirection, masked_intersect, noise_reduction_loss_1d
from geo_optimization.geo_render import GeoVisualizer
from geo_optimization.value_scheduling import make_factor_scheduler

from sim_stuff import layer_planes, plot_circle_pack


class LayerConnections(Geometry):

    def __init__(self, num_p, circles_list: list, height: float = 4.0, puffer=1.0):
        super(LayerConnections, self).__init__()
        self.visualizer = GeoVisualizer()

        self.length, self.puffer, self.num_p = height, puffer, num_p

        x1_list, x2_list, x3_list, radii_list = [], [], [], []
        for circles in circles_list:
            x1, x2, x3, r = neuron_connection_init(circles=circles, height=height, puffer=puffer, num_p=num_p)
            x1_list.append(th.tensor(x1))
            x2_list.append(th.tensor(x2))
            x3_list.append(th.tensor(x3))
            radii_list.append(th.tensor(r))

        self.x1 = nn.Parameter(th.stack(x1_list))
        self.x2 = nn.Parameter(th.stack(x2_list))
        self.x3 = th.stack(x3_list)
        self.radii = th.stack(radii_list)

        detached_x1, detached_x2, detached_x3 = self.points_detached()
        detached_x1, detached_x2, detached_x3 = detached_x1.flatten(), detached_x2.flatten(), detached_x3.flatten()
        points_2d = th.cat([detached_x1, detached_x2]).reshape(2, -1).transpose(-2, -1)
        self.puffer_zone = PufferZone(points_2d, height, puffer, detached_x3, only_connected=False)

        self.weighted_distances = WeightedEuclideanDistance(num_p)

        flat_radii = self.radii.flatten()
        mask_matrix = np.ones((len(flat_radii), len(flat_radii)))
        start = 0
        for i in range(len(self.radii)):
            for j in range(num_p):
                mask_matrix[start + j][start: start+num_p] = np.zeros((num_p))
            start += num_p

        self.min_dist_matrix = (flat_radii + flat_radii.reshape(1, -1, 1)).squeeze()
        self.min_dist_matrix = self.min_dist_matrix * mask_matrix

    def forward(self, scale=1.0):
        flat_x1 = self.x1.flatten()
        flat_x2 = self.x2.flatten()
        flat_x3 = self.x3.flatten()

        points_3d = th.cat([self.x1, self.x2, self.x3], dim=-1).reshape(-1, 3, self.num_p).transpose(-2, -1).squeeze()
        flat_points_2d = th.cat([flat_x1, flat_x2]).reshape(2, -1).transpose(-2, -1).squeeze()
        flat_points_3d = th.cat([flat_x1, flat_x2, flat_x3]).reshape(3, -1).transpose(-2, -1).squeeze()
        dist_matrix = th.cdist(flat_points_3d, flat_points_3d)

        fixed_loss = self.puffer_zone(flat_points_2d).sum()

        dist_loss = self.weighted_distances(points_3d, points_3d).sum()
        noise_loss = noise_reduction_loss_1d(self.x1).sum() + noise_reduction_loss_1d(self.x2).sum()
        intersect_loss = th.relu(self.min_dist_matrix * scale - dist_matrix).square().sum()

        return fixed_loss, dist_loss, noise_loss, intersect_loss

    def optimize(
            self,
            optimizer=None,
            epochs=1_000,
            scale=1.0,
            visualize=True,
            fixed_scale=1.0,
            intersect_scale=0.0,
            dist_scale=1.0/100,
            noise_scale=1.0,
            scheduler=None,
    ):
        fixed_scale = make_factor_scheduler(fixed_scale)
        intersect_scale = make_factor_scheduler(intersect_scale)
        dist_scale = make_factor_scheduler(dist_scale)
        noise_scale = make_factor_scheduler(noise_scale)

        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=0.01)

        if visualize:
            self.render()

        for e in range(epochs):

            optimizer.zero_grad()

            fixed_loss, dist_loss, noise_loss, intersect_loss = self(scale)

            fixed_loss = fixed_scale(fixed_loss)
            intersect_loss = intersect_scale(intersect_loss)
            dist_loss = dist_scale(dist_loss)
            noise_loss = noise_scale(noise_loss)

            loss = (fixed_loss + intersect_loss + dist_loss + noise_loss)

            loss.backward()
            optimizer.step()

            run_log = ['Epoch:', e,
                  '- loss:', np.round(loss.detach().numpy(), 3),
                  '- fixed:', np.round(fixed_loss.detach().numpy(), 3),
                  '- intersect:', np.round(intersect_loss.detach().numpy(), 3),
                  '- dist:', np.round(dist_loss.detach().numpy(), 3),
                  '- noise:', np.round(noise_loss.detach().numpy(), 3)]

            if visualize:
                self.render()

            if scheduler is not None:
                scheduler.step()
                run_log.append('- lr:',)
                run_log.append(scheduler.get_last_lr())

            print(*run_log)

    def plot(self):
        fig = pylab.figure(figsize=[10, 10],  # Inches
                           dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca(projection='3d')
        x1, x2, x3 = self.points_numpy()
        for e1, e2, e3 in zip(x1, x2, x3):
            ax.plot3D(e1, e2, e3)
            #ax.scatter(e1, e2, e3)
        return fig


    def render(self):
        fig = self.plot()
        self.visualizer.render_step(fig)
        plt.close('all')

def main():

    scale = 0.9
    lr = 0.25

    circles_list = layer_planes(weighted_radius=1.0, equal_radius=1.0)
    connections = LayerConnections(50, circles_list=circles_list)


    optimizer = Adam(connections.parameters(), lr=lr)
    scheduler = StepLR(optimizer=optimizer, step_size=200, gamma=0.5)
    connections.optimize(
        optimizer=optimizer,
        scale=scale,
        epochs=1200,
        scheduler=scheduler,
        intersect_scale=[0.1, 200, 2.0],
    )

    points_3d = connections.get_points_3d()
    dist_matrix = th.cdist(points_3d, points_3d)
    intersect = masked_intersect(dist_matrix, connections.min_dist_matrix * scale)
    intersect = intersect.detach().numpy()
    print(np.mean(intersect))
    print(np.sum(intersect))
    print(np.min(intersect))
    print(np.max(intersect))
    print(np.std(intersect))

    print(th.max(connections.x2), th.min(connections.x2), th.max(connections.x3), th.min(connections.x3))


if __name__ == '__main__':
    main()