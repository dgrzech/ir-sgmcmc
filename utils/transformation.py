from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import nn

from utils import init_identity_grid_2D, init_identity_grid_3D, transform_coordinates, transform_coordinates_inv


class TransformationModule(nn.Module, ABC):
    """
    abstract class for a transformation model, e.g. B-splines or a stationary velocity field
    """

    def __init__(self):
        super(TransformationModule, self).__init__()

    @abstractmethod
    def forward(self, v):
        pass


class SVF_2D(TransformationModule):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dims, no_steps=12):
        super(SVF_2D, self).__init__()
        identity_grid = init_identity_grid_2D(dims)

        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)
        self.no_steps = no_steps

    def forward(self, v):
        """
        integrate a 2D stationary velocity field through scaling and squaring
        """

        displacement = transform_coordinates(v) / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            transformation = self.identity_grid + displacement.permute([0, 2, 3, 1])
            displacement = displacement + F.grid_sample(displacement, transformation,
                                                        padding_mode='border', align_corners=True)

        transformation = self.identity_grid.permute([0, 3, 1, 2]) + displacement
        return transformation, transform_coordinates_inv(displacement)


class SVF_3D(TransformationModule):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dims, no_steps=12):
        super(SVF_3D, self).__init__()
        identity_grid = init_identity_grid_3D(dims)

        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)
        self.no_steps = no_steps

    def forward(self, v):
        """
        integrate a 3D stationary velocity field through scaling and squaring
        """

        displacement = transform_coordinates(v) / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            transformation = self.identity_grid + displacement.permute([0, 2, 3, 4, 1])
            displacement = displacement + F.grid_sample(displacement, transformation,
                                                        padding_mode='border', align_corners=True)

        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        return transformation, transform_coordinates_inv(displacement)
