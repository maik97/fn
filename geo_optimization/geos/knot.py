import torch as th
from torch import nn

from geo_optimization.geos import Geometry
from geo_optimization.geo_makers import resample, pretzel_knot_init

class PretzelKnot(Geometry):

    def __init__(self, num_p, radius: float = 0.5, lenght: float = 5.0, puffer=1.0):
        super(PretzelKnot, self).__init__()

        x1, x2, x3 = pretzel_knot_init(length=lenght, puffer=puffer, num_p=num_p)
        self.x1 = th.tensor(x1)
        self.x2 = nn.Parameter(th.tensor(x2))
        self.x3 = nn.Parameter(th.tensor(x3))

        self.radius = resample(radius, num_p=num_p)
