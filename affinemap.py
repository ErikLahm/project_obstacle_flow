from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class AffineMap:
    """
    This Class contains the whole structure of the affine map which maps from the reference element
    to the physical element.

    Attributes
    ----------
    vertex_coords: npt.NDArray[np.float64]
        Array containing the 3 vertices of the physical triangle.

    Properties
    ----------
    jacobian: npt.NDArray[np.float64]
        2by2 matrix/ array representing the constant jacobian over the triangle
    inverse_jacobian: npt.NDArray[np.float64]
        2by2 matrix representing the inverse of the jacobian
    determinant_jacobian: float
        Float value representing the determinant of the Jacobian for the specific
        triangle.

    Methods
    -------
    a_map
        Actual affine map, that takes coordinates from the reference space and returns
        the corresponding coordinates in the physical space.
    """

    vertex_coords: npt.NDArray[np.float64]

    # @property
    def jacobian(self, x_ref: float, y_ref: float) -> npt.NDArray[np.float64]:
        """
        2by2 matrix/ array representing the constant jacobian over the triangle
        """
        jacobian = np.array(  # type: ignore
            [
                [
                    self.vertex_coords[1][0] - self.vertex_coords[0][0],
                    self.vertex_coords[2][0] - self.vertex_coords[0][0],
                ],
                [
                    self.vertex_coords[1][1] - self.vertex_coords[0][1],
                    self.vertex_coords[2][1] - self.vertex_coords[0][1],
                ],
            ]
        )
        return jacobian  # type: ignore

    @property
    def inverse_jacobian(self) -> npt.NDArray[np.float64]:
        """
        2by2 matrix representing the inverse of the jacobian
        """
        return np.linalg.inv(self.jacobian(x_ref=0, y_ref=0))

    @property
    def determinant_jacobian(self) -> float:
        """
        Float value representing the determinant of the Jacobian for the specific
        triangle.
        """
        return (
            self.jacobian(0, 0)[0][0] * self.jacobian(0, 0)[1][1]
            - self.jacobian(0, 0)[0][1] * self.jacobian(0, 0)[1][0]
        )

    def aff_map(self, x1_hat: float, x2_hat: float) -> Tuple[float, float]:
        """
        Actual affine map, that takes coordinates from the reference space and returns
        the corresponding coordinates in the physical space.

        Parameters
        ----------
        x1_hat: float
            x_1 coordinate in the reference space
        x2_hat: float
            x_2 coordinate in the reference space

        Returns
        -------
        Tuple[float,float]
            Tuple representing the x_1 and x_2 coordinate in the physical space.
        """

        x_1 = (
            self.vertex_coords[0][0]
            + (self.vertex_coords[1][0] - self.vertex_coords[0][0]) * x1_hat
            + (self.vertex_coords[2][0] - self.vertex_coords[0][0]) * x2_hat
        )
        x_2 = (
            self.vertex_coords[0][1]
            + (self.vertex_coords[1][1] - self.vertex_coords[0][1]) * x1_hat
            + (self.vertex_coords[2][1] - self.vertex_coords[0][1]) * x2_hat
        )
        return x_1, x_2
