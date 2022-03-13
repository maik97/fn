import torch as th
from torch import nn


class Geometry(nn.Module):

    def __init__(self):
        super(Geometry, self).__init__()
        pass

    def points_detached(self):
        return (self.x1.clone().detach(),
                self.x2.clone().detach(),
                self.x3.clone().detach())

    def points_numpy(self):
        return (self.x1.clone().detach().numpy(),
                self.x2.clone().detach().numpy(),
                self.x3.clone().detach().numpy())

    def plot_to_ax(self, ax):
        x1, x2, x3 = self.points_numpy()

        if x1.ndim == 1:
            ax.plot3D(x1, x2, x3, c='lightgrey')
            ax.scatter(x1, x2, x3)

        else:
            for e1, e2, e3 in zip(x1, x2, x3):
                ax.plot3D(e1, e2, e3, c='lightgrey')
                ax.scatter(e1, e2, e3)