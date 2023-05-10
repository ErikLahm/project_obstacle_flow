# pyright: reportMissingTypeStubs=false
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from rectangular_grid import RectangularGrid


@dataclass
class Triangulation:
    """
    This Class describes a triangulation of the domain given by a rectangular grid.

    Attributes
    ----------
    rect_mesh: RectangularGrid
        Rectangular Mesh that describes all vertices for the triangulation.
    delauny_tri: Delaunay = field(init=False)
        The actual triangulation that describes the vertices and triangles.

    Properties
    ----------
    number_of_triangles
        The number of trianlges in the Triangulation.
    number_of_vertices
        The number of vertices that make up the whole triangulation.
    number_of_bdn
        The number of boundary points. Boundary points literally means all
        boundary points regardless of boundary type.
    number_diri_points
        The number of points that belong to the Dirichlet boundary.
    num_dof_neumann_right
        Number of degrees of freedom of the system. These are made up of all inner
        vertices and the vertices on the boundary corresponding to Neumann boundary
        conditions.
    number_of_inner
        Number of inner points. Literally the number of all points that do not belong
        to any boundary.
    """

    rect_mesh: RectangularGrid
    delauny_tri: Delaunay = field(init=False)

    def __post_init__(self):
        self.delauny_tri = Delaunay(self.rect_mesh.sorted_mesh_coords)

    @property
    def number_of_triangles(self) -> int:
        return len(self.delauny_tri.simplices)

    @property
    def number_of_vertices(self) -> int:
        return len(self.rect_mesh.sorted_mesh_coords)  # type: ignore

    @property
    def number_of_bdn(self) -> int:
        """
        The number of boundary points. Boundary points literally means all
        boundary points regardless of boundary type.
        """
        return int(
            2 * self.rect_mesh.x_discretisation
            + 2 * (self.rect_mesh.y_discretisation - 2)
        )

    @property
    def num_diri_points(self) -> int:
        """
        The number of points that belong to the Dirichlet boundary.
        """
        return self.number_of_bdn - self.num_dof_neumann_right + self.number_of_inner

    @property
    def num_dof_neumann_right(self) -> int:
        """
        Number of degrees of freedom of the system. These are made up of all inner
        vertices and the vertices on the boundary corresponding to Neumann boundary
        conditions.
        """
        return self.number_of_inner + (self.rect_mesh.y_discretisation - 2)

    @property
    def number_of_inner(self) -> int:
        """
        Number of inner points. Literally the number of all points that do not belong
        to any boundary.
        """
        return (
            self.rect_mesh.x_discretisation * self.rect_mesh.y_discretisation
            - self.number_of_bdn
        )

    def plot_triangulation(
        self,
        all_vertex_coords: npt.NDArray[np.float64],
        all_discrete_points: npt.NDArray[np.float64],
    ):
        """
        Method to visualise the grid. Especially helpful for low discretisation to understand the triangulation.

        Plots the vertices and edges of each triangle. Further it annotates each point in the grid with the
        corresponding index in the vertex array.

        Parameters
        ----------
        all_vertex_coords: npt.NDArray[np.float64]
            Array that contains all vertex coordinates that make up the triangulation.
        all_discrete_points: npt.NDArray[np.float64]
            Array that contains all discrete points making up the finite elements (including the barycentric
            points for the bubble function)
        """

        _, ax = plt.subplots()  # type: ignore
        ax.triplot(  # type: ignore
            all_vertex_coords[:, 0],  # type: ignore
            all_vertex_coords[:, 1],  # type: ignore
            self.delauny_tri.simplices,  # type: ignore
            color="grey",
        )
        ax.scatter(all_discrete_points[:, 0], all_discrete_points[:, 1], c="red")  # type: ignore
        idx_points = np.arange(len(all_discrete_points))  # type: ignore
        for i, idx_p in enumerate(idx_points):
            ax.annotate(idx_p, (all_discrete_points[i, 0], all_discrete_points[i, 1]))  # type: ignore
        # plt.show()  # type: ignore
