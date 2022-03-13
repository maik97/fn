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
from geo_optimization.geo_makers import resample, pretzel_knot_init
from geo_optimization.geo_losses import PufferZone, WeightedEuclideanDistance, WeightedDirection, masked_intersect, noise_reduction_loss
from geo_optimization.geo_render import GeoVisualizer


class PretzelKnot(Geometry):

    def __init__(self, num_p, radius: (float, list) = 0.5, length: float = 4.0, puffer=1.0):
        super(PretzelKnot, self).__init__()

        self.radius, self.length, self.puffer, self.num_p = radius, length, puffer, num_p

        x1, x2, x3 = pretzel_knot_init(radius=radius, length=length, puffer=puffer, num_p=num_p)
        self.x1 = th.tensor(x1)
        self.x2 = nn.Parameter(th.tensor(x2))
        self.x3 = nn.Parameter(th.tensor(x3))
        self.radius = th.tensor(resample(self.radius, num_p=self.num_p))

        detached_x1, detached_x2, detached_x3 = self.points_detached()
        points_2d = th.cat([detached_x2, detached_x3]).reshape(2, -1).transpose(-2, -1)
        self.puffer_zone = PufferZone(points_2d, self.length, self.puffer, detached_x1)

        self.min_dist_matrix = (self.radius + self.radius.reshape(1, -1, 1)).squeeze()

        self.weighted_distances = WeightedEuclideanDistance(num_p)
        self.weighted_direction = WeightedDirection(num_p)

        self.visualizer = GeoVisualizer()

    def resample(self, num_p=None):
        if num_p is not None:
            self.num_p = num_p

        points_2d = self.get_points_2d().clone().detach()
        points_2d = self.puffer_zone.force_constraints(points_2d)
        points_2d = points_2d.transpose(-2, -1).numpy()
        x2, x3 = points_2d[0], points_2d[1]
        x1 = self.x1.clone().detach().numpy()

        x1 = th.tensor(resample(x1, num_p=self.num_p))
        x2 = th.tensor(resample(x2, num_p=self.num_p))
        x3 = th.tensor(resample(x3, num_p=self.num_p))

        self.x1 = x1
        state_dict = self.state_dict()
        state_dict['x2'].copy_(x2)
        state_dict['x3'].copy_(x3)

        self.radius = th.tensor(resample(self.radius, num_p=self.num_p))
        self.min_dist_matrix = (self.radius + self.radius.reshape(1, -1, 1)).squeeze()

        detached_x1, detached_x2, detached_x3 = self.points_detached()
        points_2d = th.cat([detached_x2, detached_x3]).reshape(2, -1).transpose(-2, -1)
        self.puffer_zone = PufferZone(points_2d, self.length, self.puffer, detached_x1)

    def get_points_2d(self):
        return th.cat([self.x2, self.x3]).reshape(2,-1).transpose(-2, -1).squeeze()

    def get_points_3d(self):
        return th.cat([self.x1, self.x2, self.x3]).reshape(3,-1).transpose(-2, -1).squeeze()

    def forward(self, scale=1.0):
        points_2d = self.get_points_2d()
        points_3d = self.get_points_3d()
        dist_matrix = th.cdist(points_3d, points_3d)

        intersect_loss = th.sum(masked_intersect(dist_matrix, self.min_dist_matrix * scale))
        dist_loss = th.sum(self.weighted_distances(points_3d, points_3d))
        direct_loss = th.sum(self.weighted_direction(points_3d))
        noise_loss = th.sum(noise_reduction_loss(points_2d))
        fixed_loss = th.sum(self.puffer_zone(points_2d))
        volume_loss = th.max(self.x2) + th.abs(th.min(self.x2)) + th.max(self.x3) + th.abs(th.min(self.x3))


        return fixed_loss, intersect_loss, dist_loss, direct_loss, noise_loss, volume_loss*0

    def optimize(
            self,
            optimizer=None,
            epochs=1_000,
            scale=1.0,
            visualize=True,
            fixed_scale=1.0,
            intersect_scale=1.0,
            dist_scale=1.0/10_000,
            noise_scale=10.0,
            scheduler=None,
    ):

        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=0.01)

        if visualize:
            self.render()

        for e in range(epochs):

            optimizer.zero_grad()

            fixed_loss, intersect_loss, dist_loss, direct_loss, noise_loss, volume_loss = self(scale)

            fixed_loss = fixed_loss * fixed_loss_scale
            intersect_loss = intersect_loss * intersect_loss_scale
            dist_loss = dist_loss * dist_loss_scale
            direct_loss = direct_loss * dist_loss_scale
            noise_loss = noise_loss * noise_loss_scale

            loss = (fixed_loss + intersect_loss + dist_loss + noise_loss + direct_loss + volume_loss)

            loss.backward()
            optimizer.step()

            run_log = ['Epoch:', e,
                  '- loss:', np.round(loss.detach().numpy(), 3),
                  '- fixed:', np.round(fixed_loss.detach().numpy(), 3),
                  '- intersect:', np.round(intersect_loss.detach().numpy(), 3),
                  '- dist:', np.round(dist_loss.detach().numpy(), 3),
                  '- direct:', np.round(direct_loss.detach().numpy(), 3),
                  '- noise:', np.round(noise_loss.detach().numpy(), 3),
                  '- volume:', np.round(volume_loss.detach().numpy(), 3),]

            if visualize:
                self.render()

            if scheduler is not None:
                scheduler.step()
                run_log.append('- lr:',)
                run_log.append(scheduler.get_last_lr())

            if e % 200 == 0:
                print(e)
                self.resample()
                intersect_loss_scale = intersect_loss_scale*2

            print(*run_log)

    def plot(self):
        fig = pylab.figure(figsize=[10, 10],  # Inches
                           dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca(projection='3d')
        self.plot_to_ax(ax)
        return fig

    def render(self):
        fig = self.plot()
        self.visualizer.render_step(fig)
        plt.close('all')

def main():
    radius = 0.9
    scale = 1.0
    lr = 0.5

    pretzel_knot = PretzelKnot(100, radius=radius)
    optimizer = Adam(pretzel_knot.parameters(), lr=lr)
    scheduler = StepLR(optimizer=optimizer, step_size=200, gamma=0.5)
    pretzel_knot.optimize(optimizer=optimizer, scale=scale, epochs=1200, scheduler=scheduler)

    points_3d = pretzel_knot.get_points_3d()
    dist_matrix = th.cdist(points_3d, points_3d)
    intersect = masked_intersect(dist_matrix, pretzel_knot.min_dist_matrix * scale)
    intersect = intersect.detach().numpy()
    print(np.mean(intersect))
    print(np.sum(intersect))
    print(np.min(intersect))
    print(np.max(intersect))
    print(np.std(intersect))

    print(th.max(pretzel_knot.x2), th.min(pretzel_knot.x2), th.max(pretzel_knot.x3), th.min(pretzel_knot.x3))


if __name__ == '__main__':
    main()