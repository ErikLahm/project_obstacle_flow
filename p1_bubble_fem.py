# pyright: reportMissingTypeStubs=false
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt

from triangulation import Triangulation


@dataclass
class P1BubbleFE:
    """
    This class describes the structure of a P1 finite element enriched with bubble functions.

    This class holds information such as all coordinates of all points including the barycentric
    points of each triangle.

    Attributes
    ----------
    triangulation: Triangulation
        Describing the base triangulation on which we initialise the P1 Element with bubble functions.
    dof_coords: npt.NDArray[np.float64] = field(init=False)
        Array containing all degrees of freedom in the case of boundary conditions applied through a
        lifting. (Dirichlet boundary points are NO dofs)
    discret_points_complete: npt.NDArray[np.float64] = field(init=False)
        Array containing all discrete points that describe the FEM. It can be used for the the approach
        of boundary values applied through penalty terms, since here all discrete points are considered
        dofs.

    Properties
    ----------
    ltg_u1: npt.NDArray[np.float64]
        local to global map of first velocity component in the case of boundary lifitng.
    ltg_u1_penatly: npt.NDArray[np.float64]
        local to global map of first velocity component in the case of boundary penalty terms
    ...
    """

    triangulation: Triangulation
    dof_coords: npt.NDArray[np.float64] = field(init=False)
    discret_points_complete: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        self.dof_coords, self.discret_points_complete = self.get_all_discrete_points()

    @property
    def get_index_of_center_points(self) -> npt.NDArray[np.float64]:
        """
        Property represents an index array which represents the numbering of the center points.

        The numbering of center points starts right after the last vertex number. This is to say that
        first all triangle corners (vertices) are numbered that correspond to degrees of freedom (not
        Dirichlet boundary points) and after that all center points are numbered. The order of the
        numbering corresponds to the order of triangles in the triangulation.

        Returns
        -------
        NDArray[np.float64]
            Array with indices of the center points of triangles.
        """

        idx_center = np.arange(  # type: ignore
            self.triangulation.num_dof_neumann_right,  # type: ignore
            self.triangulation.number_of_triangles + self.triangulation.num_dof_neumann_right,  # type: ignore
        )
        center_idx = np.array(idx_center)  # type: ignore
        return center_idx  # type: ignore

    @property
    def number_dof_w_bubble(self) -> int:
        """
        Describes the number of degrees of freedom, including all inner vertices, Neummann boundary
        points and the center points for the bubble functions.

        Returns
        -------
        int
            Integer describing the total number of dof of the system.
        """

        return (
            self.triangulation.num_dof_neumann_right
            + self.triangulation.number_of_triangles
        )

    @property
    def ltg_u1(self) -> npt.NDArray[np.int32 | np.float64]:
        """
        Local to Global map for the first component of the velocity. This LTG describes the case when
        implementing the dirichlet boundary conditions through a lifting.
        """
        ltg = np.copy(self.ltg_w_bubble())
        return ltg

    @property
    def ltg_u2(self) -> npt.NDArray[np.int32 | np.float64]:
        """
        Local to Global map of the second component of the velocity. This LTG describes the case when
        implementing the dirichlet boundary conditions through a lifting.
        """
        ltg = np.copy(self.ltg_u1)
        ltg[ltg > -1] = ltg[ltg > -1] + self.number_dof_w_bubble
        return ltg

    @property
    def ltg_u1_penalty(self) -> npt.NDArray[np.float64 | np.int32]:
        """
        Property containing the LTG map of the first component of velocity in the case of boundary consitions
        applied through penatly terms. (all discrete points are dofs)
        """

        ltg_u1 = np.copy(self.ltg_w_bubble_penalty())
        return ltg_u1

    @property
    def ltg_u2_penalty(self) -> npt.NDArray[np.float64 | np.int32]:
        """
        Property containing the LTG map of the second component of velocity in the case of boundary consitions
        applied through penatly terms. (all discrete points are dofs)
        """
        ltg_u1 = np.copy(self.ltg_u1_penalty)
        ltg_u2 = ltg_u1 + len(self.discret_points_complete)
        return ltg_u2

    @property
    def ltg_p(self) -> npt.NDArray[np.int32 | np.float64]:
        """
        Local to Global map for the pressure degrees of freedom. Note that the pressure dof comprise all
        vertices. That is to say also the boundary terms are unknown for the pressure, regardless of
        being known for the velocity.

        Further since we build up the D matrix separatly we can choose here the pressure LTG map as the
        initial LTG of the triangulation without negative values.
        """

        base_ltg = np.copy(self.triangulation.delauny_tri.simplices)
        return base_ltg

    def get_barycenter_coords(self) -> npt.NDArray[np.float64]:
        """
        Returns an array of shape (n_center_pts, 2) where each tuple describes x and y coordinate
        of a point in the center of each triangle.

        The method loops over all triangles and finds the center points in the same order. Hence,
        each the numbering of triangles and center points is equal. The center points are
        calculated using the barycentric coordinates, which are all 1/3 for a center point.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_center_pts, 2) describing the x, y coordinate of each center point.
        """

        coordinates = np.zeros(shape=(1, 2))
        for triangle in self.triangulation.delauny_tri.simplices:
            triangle_coords: npt.NDArray[np.float64] = self.triangulation.rect_mesh.sorted_mesh_coords[triangle]  # type: ignore
            x_coord: float = (  # type: ignore
                1
                / 3
                * (
                    triangle_coords[0][0]  # type: ignore
                    + triangle_coords[1][0]  # type: ignore
                    + triangle_coords[2][0]  # type: ignore
                )
            )
            y_coord = (  # type: ignore
                1
                / 3
                * (
                    triangle_coords[0][1]  # type: ignore
                    + triangle_coords[1][1]  # type: ignore
                    + triangle_coords[2][1]  # type: ignore
                )
            )
            single_coords = np.array([x_coord, y_coord])  # type: ignore
            coordinates = np.vstack((coordinates, single_coords))
        coordinates = np.delete(coordinates, obj=0, axis=0)
        return coordinates

    def get_all_discrete_points(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Creates tuple of arrays, where the first describes coordinates of all degrees of freedom
        and the second all coordinates of all points in the domain.

        First array is made up of all inner vertices, the boundary vertices corresponding to
        Neumann boundary conditions and the center points for the bubble functions.
        The second array adds all boundary terms to the end of the first array.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            First array describes coords of all dof. Second describes coords of all discrete points
            in the domain (dof and Dirichlet boundary).
        """
        central_dof_coords = self.get_barycenter_coords()
        split_location = self.triangulation.num_dof_neumann_right
        splitted_coords = np.vsplit(  # type: ignore
            self.triangulation.rect_mesh.sorted_mesh_coords, np.array([split_location])  # type: ignore
        )
        dof_vertices = splitted_coords[0]  # type: ignore
        diri_bdn_vertices = splitted_coords[1]  # type: ignore
        dof_coords_complete = np.vstack((dof_vertices, central_dof_coords))  # type: ignore
        dicrete_coords_complete = np.vstack((dof_coords_complete, diri_bdn_vertices))  # type: ignore
        return dof_coords_complete, dicrete_coords_complete

    def ltg_w_neg_bdn(self) -> npt.NDArray[np.int32]:
        """
        Takes the local to global map generated by Delauny and gives all Dirichlet boundary terms
        a negative value.

        The negative value corresponds to the index in the array of all discrete points in the
        domain. (Note: negative values index an array from the back, where the Dirichlet points are)

        Returns
        -------
        NDArray[np.float64]
            The Local to Global Map with negative indices for all Dirichlet boundary points.
        """

        base_ltg = np.copy(self.triangulation.delauny_tri.simplices)
        base_ltg[base_ltg >= self.triangulation.num_dof_neumann_right] = (
            base_ltg[base_ltg >= self.triangulation.num_dof_neumann_right]
            - self.triangulation.number_of_vertices
        )
        return base_ltg

    def ltg_w_new_idx(self) -> npt.NDArray[np.int32]:
        """
        Takes the original LTG generated by Delauny and renumbers the Dirichlet boundary points based on
        their new position in the coordinates array (discrete_points_complete). This LTG map is used for
        the method of applying boundary conditions through penalty terms.
        """
        origin_ltg = np.copy(self.triangulation.delauny_tri.simplices)
        origin_ltg[origin_ltg >= self.triangulation.num_dof_neumann_right] = (
            origin_ltg[origin_ltg >= self.triangulation.num_dof_neumann_right]
            + self.triangulation.number_of_triangles
        )
        return origin_ltg

    def ltg_w_bubble(self) -> npt.NDArray[np.int32 | np.float64]:
        """
        Creates the Local to Global map of the whole system.

        This local to global map describes all triangles with their four dof and references with
        each entry to an index of the array of all points of the systems.

        Returns
        -------
        NDArray[np.float64]
            Local to Global Map of the whole System.
        """
        ltg_neg_diri = self.ltg_w_neg_bdn()
        bubble_idx = np.reshape(
            self.get_index_of_center_points, (self.triangulation.number_of_triangles, 1)
        )
        ltg = np.hstack((ltg_neg_diri, bubble_idx))
        return ltg

    def ltg_w_bubble_penalty(self) -> npt.NDArray[np.int32 | np.float64]:
        """
        Creates the Local to Global map of the whole system in case of boundary conditions applied
        through penatly terms.

        This local to global map describes all triangles with their four dof and references with
        each entry to an index of the array of all points of the systems.

        Returns
        -------
        NDArray[np.float64]
            Local to Global Map of the whole System.
        """
        ltg_new_idx = self.ltg_w_new_idx()
        bubble_idx = np.reshape(
            self.get_index_of_center_points, (self.triangulation.number_of_triangles, 1)
        )
        ltg = np.hstack((ltg_new_idx, bubble_idx))
        return ltg
