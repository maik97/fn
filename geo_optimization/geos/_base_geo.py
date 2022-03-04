import torch as th
from torch import nn


class Geometry(nn.Module):

    def __init__(self):
        super(Geometry, self).__init__()
        pass

    def points(self):
        return (self.x1.clone().detach().numpy(),
                self.x2.clone().detach().numpy(),
                self.x3.clone().detach().numpy())

    def plot_to_ax(self, ax):
        x1, x2, x3 = self.points()
        ax.plot3D(x1, x2, x3, c='lightgrey')
        ax.scatter(x1, x2, x3)
