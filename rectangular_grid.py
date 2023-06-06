# pyright: reportMissingTypeStubs=false
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class RectangularGrid:
    """
    This Class structures a grid of points that can be used to make up a triangulation.

    The points are ordered in a rectangular form.

    Attributes
    ----------
    x_domain: Tuple[float, float]
        Tuple that describes the size of the domain in the x dimension.
    y_domain: Tuple[float, float]
        Tuple that describes the size of the domain in the y dimension.
    x_discretisation: int
        Integer value that describes the discretisation of the x dimension.
    y_discretisation: int
        Integer value that describes the discretisation of the y dimension.

    Properties
    ----------
    x_linespace: NDArray[np.float64]
        Array that devides the x dimension in the specified discrete points.
    y_linespace: NDArray[np.float64]
        Array that devides the y dimension in the specified discrete points.
    sorted_mesh_coords: NDArray[np.float64]
        Array with shape (n_dof,2) that describes x and y coordinates of each point in
        the grid. It is sorted in the way that all Dirichlet boundary points are last.

    Methods
    -------
    get_reordered_idx_neumann_right
        Returns an Array of indices where all indices of Dirichlet boundary points from the
        left side of the domain are in the end.
    """

    x_domain: Tuple[float, float]
    y_domain: Tuple[float, float]
    x_discretisation: int
    y_discretisation: int

    @property
    def x_linspace(self) -> npt.NDArray[np.float64]:
        return np.linspace(self.x_domain[0], self.x_domain[1], self.x_discretisation)

    @property
    def y_linspace(self) -> npt.NDArray[np.float64]:
        return np.linspace(self.y_domain[0], self.y_domain[1], self.y_discretisation)

    @property
    def sorted_mesh_coords(self) -> npt.NDArray[np.float64]:
        """
        This property generates a mesh of grid points based on the given domanin and discretisation.
        The mesh array is reshaped to (n_dof,2) where the first column are x coordinates and second
        column the y coordinates of each point of the grid. Then this Array is sorted such that all
        Dirichlet boundary points are moved to the end of the array.
        """

        x_vertex_coords, y_vertex_coords = np.meshgrid(self.x_linspace, self.y_linspace)
        pre_ordered_y = np.roll(y_vertex_coords, -self.x_discretisation)
        x_vertex_coords = x_vertex_coords.flatten()
        pre_ordered_y = pre_ordered_y.flatten()
        pre_ordered_coords = np.vstack((x_vertex_coords, pre_ordered_y))
        reorder_idx = self.get_reordered_idx_neumann_right()
        split_pre_order = np.hsplit(
            pre_ordered_coords,
            np.array([self.x_discretisation * (self.y_discretisation - 2)]),
        )
        unordered_coords = split_pre_order[0]
        ordered_coords_back = split_pre_order[1].T
        ordered_coords_front: npt.NDArray[
            np.float64
        ] = unordered_coords.T[  # type:ignore
            reorder_idx
        ]
        sorted_vertices_coords = np.vstack(
            (ordered_coords_front, ordered_coords_back)  # type:ignore
        )
        return sorted_vertices_coords

    def get_reordered_idx_dirichlet(self) -> npt.NDArray[np.int32]:
        """
        The method is moving also the boundary terms from the left and right side to the end of the array
        of vertices. Hence, this method prepares the vertices in such a way that Dirichlet boundaries are
        assumed all around the domain.
        """

        left_bdn_lst: list[int] = [
            self.x_discretisation * i for i in range(self.y_discretisation - 2)
        ]
        right_bdn_lst: list[int] = [
            (self.x_discretisation - 1) + i * self.x_discretisation
            for i in range(self.y_discretisation - 2)
        ]
        left_bdn_idx = np.array(left_bdn_lst)
        right_bdn_idx = np.array(right_bdn_lst)
        lr_bdn_id = np.hstack((left_bdn_idx, right_bdn_idx))
        complete_idx = np.arange(0, self.x_discretisation * (self.y_discretisation - 2))  # type: ignore
        inner_idx = np.setdiff1d(complete_idx, lr_bdn_id)
        reorder_idx = np.hstack((inner_idx, lr_bdn_id))
        return reorder_idx

    def get_reordered_idx_neumann_right(self) -> npt.NDArray[np.int32]:
        """
        The method gives an index array which can be used to reorder the coordinate array such that also the
        boundary values from the left hand side are put to the back of the array. That way all Dirichlet
        boundary terms are moved to the end of the array. Only the Neumann boundary terms stay in their
        initial position.
        """

        left_bdn_lst: list[int] = [
            self.x_discretisation * i for i in range(self.y_discretisation - 2)
        ]
        left_bdn_idx = np.array(left_bdn_lst)
        complete_idx = np.arange(0, self.x_discretisation * (self.y_discretisation - 2))  # type: ignore
        inner_idx = np.setdiff1d(complete_idx, left_bdn_idx)
        reorder_idx = np.hstack((inner_idx, left_bdn_idx))
        return reorder_idx

    def get_reverted_idx_neumann_right(self) -> npt.NDArray[np.int32]:
        """
        Function to revert the reordering of the array. This is necessary to later reorder the solution arrays
        back to the original order for plotting for example the pressure scalar field.
        """
        complete_idx = np.arange(0, self.x_discretisation * (self.y_discretisation - 2))  # type: ignore
        start_bdn_idx = (self.x_discretisation - 1) * (self.y_discretisation - 2)
        left_bdn_lst = [start_bdn_idx + i for i in range(self.y_discretisation - 2)]
        left_bdn_arr = np.array(left_bdn_lst)
        for j, bdn_idx in enumerate(left_bdn_lst):
            complete_idx = np.insert(complete_idx, j * self.y_discretisation, bdn_idx)
        complete_idx = np.delete(complete_idx, (left_bdn_arr + len(left_bdn_lst)))
        return complete_idx
