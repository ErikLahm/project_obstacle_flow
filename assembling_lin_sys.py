from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from affinemap import AffineMap
from p1_bubble_fem import P1BubbleFE
from reference_element import ReferenceElement

EPSILON = 1e-14


@dataclass
class AssemblerPenalty:
    """
    This class assembles the whole linear system for the case of boundary values induced
    through penalty terms.

    Attributes
    ----------
    reference_element: ReferenceElement
        Class that contains all information about the reference element (shape functions,
        gradients etc.)
    fe_physical: P1BubbleFE
        the physical finite element structure.
    nu: float
        the parameter nu of the system.
    c_const: float
        the constant parameter c of the system.

    """

    reference_element: ReferenceElement
    fe_physical: P1BubbleFE
    nu: float
    c_const: float

    def assemble_n_penalty(self) -> npt.NDArray[np.float64]:
        """
        Assembles the matrix representing the laplacian term of the navier stokes equation.
        """
        ltg_list = [self.fe_physical.ltg_u1_penalty, self.fe_physical.ltg_u2_penalty]
        n_matrix = np.zeros(
            shape=(
                2 * len(self.fe_physical.discret_points_complete),
                2 * len(self.fe_physical.discret_points_complete),
            )
        )
        for i in range(2):
            for l in range(self.fe_physical.triangulation.number_of_triangles):
                triangle_coords = self.fe_physical.discret_points_complete[
                    self.fe_physical.ltg_u1_penalty[l]
                ]
                affine_map_l = AffineMap(vertex_coords=triangle_coords)
                for k in range(self.reference_element.n_dof_bubble):
                    row = ltg_list[i][l][k]
                    for j in range(self.reference_element.n_dof_bubble):
                        col = ltg_list[i][l][j]
                        q = self.integral_a(triangle_map=affine_map_l, k=k, j=j)
                        n_matrix[row][col] = n_matrix[row][col] + q
            for diag in range(
                self.fe_physical.number_dof_w_bubble,
                len(self.fe_physical.discret_points_complete),
            ):
                n_matrix[diag, diag] += 1 / EPSILON
            for diag in range(
                len(self.fe_physical.discret_points_complete)
                + self.fe_physical.number_dof_w_bubble,
                2 * len(self.fe_physical.discret_points_complete),
            ):
                n_matrix[diag, diag] += 1 / EPSILON
        assert np.linalg.matrix_rank(n_matrix) == min(n_matrix.shape), (
            f"Laplacian matrix is not full rank: rank(N)={np.linalg.matrix_rank(n_matrix)} "
            f"but should be rank(N)=min(m,m)={min(n_matrix.shape)}."
        )
        return n_matrix

    def integral_a(self, triangle_map: AffineMap, k: int, j: int) -> float:
        """
        Evaluates the laplacian integral (dot product of two gradients of shape functions) on
        basis of a quadrature rule on a reference triangle of order 6.
        """
        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        points = [
            (0.659027622374092, 0.231933368553031),
            (0.659027622374092, 0.109039009072877),
            (0.231933368553031, 0.659027622374092),
            (0.231933368553031, 0.109039009072877),
            (0.109039009072877, 0.659027622374092),
            (0.109039009072877, 0.231933368553031),
        ]
        integral_sum = 0
        for i, _ in enumerate(weights):
            partial_sum = weights[i] * np.dot(
                self.reference_element.gradient_list[k](points[i][0], points[i][1]),
                np.dot(
                    np.dot(
                        triangle_map.inverse_jacobian, triangle_map.inverse_jacobian.T
                    ),
                    self.reference_element.gradient_list[j](
                        points[i][0], points[i][1]
                    ).T,
                ),
            )
            integral_sum += partial_sum
        integral = (
            1 / 2 * self.nu * integral_sum * abs(triangle_map.determinant_jacobian)
        )
        return integral

    def assemble_d_penalty(self) -> npt.NDArray[np.float64]:
        ltg_list = [self.fe_physical.ltg_u1_penalty, self.fe_physical.ltg_u2_penalty]
        d_matrix = np.zeros(
            shape=(
                self.fe_physical.triangulation.number_of_vertices,
                2 * len(self.fe_physical.discret_points_complete),
            )
        )
        for i in range(2):
            break_p = "break"
            for l in range(self.fe_physical.triangulation.number_of_triangles):
                triangle_coords = self.fe_physical.discret_points_complete[
                    self.fe_physical.ltg_u1_penalty[l]
                ]
                affine_map_l = AffineMap(vertex_coords=triangle_coords)
                for k in range(self.reference_element.n_dof_lin):
                    row = self.fe_physical.ltg_p[l][k]
                    if row < 0:
                        continue
                    # row = row - self.fe_physical.number_dof_with_bubble * 2
                    for j in range(self.reference_element.n_dof_bubble):
                        col = ltg_list[i][l][j]
                        if col < 0:
                            continue
                        q = self.integral_b(triangle_map=affine_map_l, k=k, j=j, i=i)
                        d_matrix[row][col] += q
        assert np.linalg.matrix_rank(d_matrix) == min(d_matrix.shape), (
            f"Gradient matrix is not full rank: rank(G)={np.linalg.matrix_rank(d_matrix)} "
            f"but should be rank(G)=min(m,n)={min(d_matrix.shape)}."
        )
        return d_matrix

    def integral_b(self, triangle_map: AffineMap, k: int, j: int, i: int) -> float:
        """
        Evaluates the divergence integral on basis of a quadrature rule on a reference
        triangle of order 6.
        """
        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        points = [
            (0.659027622374092, 0.231933368553031),
            (0.659027622374092, 0.109039009072877),
            (0.231933368553031, 0.659027622374092),
            (0.231933368553031, 0.109039009072877),
            (0.109039009072877, 0.659027622374092),
            (0.109039009072877, 0.231933368553031),
        ]
        integral_sum = 0
        for s, _ in enumerate(weights):
            partial_sum = (
                weights[s]
                * self.reference_element.linear_shape_func_list[k](
                    points[s][0], points[s][1]
                )
                * (
                    np.dot(
                        triangle_map.inverse_jacobian[:, i],
                        self.reference_element.gradient_list[j](
                            points[s][0], points[s][1]
                        ),
                    )
                )
            )
            integral_sum += partial_sum
        integral = -1 / 2 * integral_sum * abs(triangle_map.determinant_jacobian)
        return integral

    def assemble_s_penalty(
        self, characteristic_func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        """
        Assembles the full matrix of the linear system consiting of N, D, D^T and the
        penalty matrix.
        """
        n_matr = self.assemble_n_penalty()
        penalty = self.assemble_penalty_matrix(characteristic_func)
        n_matr += penalty
        d_matr = self.assemble_d_penalty()
        d_tr_matr = np.transpose(d_matr)
        zero_block = np.zeros(
            shape=(
                self.fe_physical.triangulation.number_of_vertices,
                self.fe_physical.triangulation.number_of_vertices,
            )
        )
        left_side = np.vstack((n_matr, d_matr))
        right_side = np.vstack((d_tr_matr, zero_block))
        s_matrix = np.hstack((left_side, right_side))
        return s_matrix, n_matr, d_matr

    def rhs_penalty(
        self, g_boundary_func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        """
        Assembles the right hand side of the system by imposing the boundary conditions on
        each right hand side element that corresponds to a boundary term.
        """
        rhs_u1 = np.zeros(shape=(len(self.fe_physical.discret_points_complete), 1))
        for i, _ in enumerate(rhs_u1):
            rhs_u1[i] += (
                g_boundary_func(
                    self.fe_physical.discret_points_complete[i][0],
                    self.fe_physical.discret_points_complete[i][1],
                )
                * 1
                / EPSILON
            )
        rhs_u2_p = np.zeros(
            shape=(
                len(self.fe_physical.discret_points_complete)
                + self.fe_physical.triangulation.number_of_vertices,
                1,
            )
        )
        rhs = np.vstack((rhs_u1, rhs_u2_p))
        return rhs

    def assemble_penalty_matrix(
        self, characteristic_func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        """
        Assembles the matrix generated by the penalty term which then gets added to the
        laplacian sub-matrix (N matrix).
        """
        ltg_list = [self.fe_physical.ltg_u1_penalty, self.fe_physical.ltg_u2_penalty]
        penalty_matrix = np.zeros(
            shape=(
                2 * len(self.fe_physical.discret_points_complete),
                2 * len(self.fe_physical.discret_points_complete),
            )
        )
        for i in range(2):
            for l in range(self.fe_physical.triangulation.number_of_triangles):
                triangle_coords = self.fe_physical.discret_points_complete[
                    self.fe_physical.ltg_u1_penalty[l]
                ]
                affine_map_l = AffineMap(vertex_coords=triangle_coords)
                for k in range(self.reference_element.n_dof_bubble):
                    row = ltg_list[i][l][k]
                    for j in range(self.reference_element.n_dof_bubble):
                        col = ltg_list[i][l][j]
                        q = self.integral_penalty(
                            triangle_map=affine_map_l,
                            k=k,
                            j=j,
                            characteristic_func=characteristic_func,
                        )
                        penalty_matrix[row][col] = penalty_matrix[row][col] + q
        return penalty_matrix

    def integral_penalty(
        self,
        triangle_map: AffineMap,
        k: int,
        j: int,
        characteristic_func: Callable[[float, float], float],
    ) -> float:
        """
        Evaluates the penalty integral (product of two shape functions with characteristic
        function) on basis of a quadrature rule on a reference triangle of order 6.
        """

        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        points = [
            (0.659027622374092, 0.231933368553031),
            (0.659027622374092, 0.109039009072877),
            (0.231933368553031, 0.659027622374092),
            (0.231933368553031, 0.109039009072877),
            (0.109039009072877, 0.659027622374092),
            (0.109039009072877, 0.231933368553031),
        ]
        integral_sum = 0
        for i, _ in enumerate(weights):
            x_1_coord_physical = triangle_map.aff_map(points[i][0], points[i][1])[0]
            x_2_coord_physical = triangle_map.aff_map(points[i][0], points[i][1])[1]
            partial_sum = (
                weights[i]
                * characteristic_func(x_1_coord_physical, x_2_coord_physical)
                * self.reference_element.bubble_shape_func_list[k](
                    points[i][0], points[i][1]
                )
                * self.reference_element.bubble_shape_func_list[j](
                    points[i][0], points[i][1]
                )
            )
            integral_sum += partial_sum
        integral = (
            1 / 2 * 1 / EPSILON * integral_sum * abs(triangle_map.determinant_jacobian)
        )
        return integral
