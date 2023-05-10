from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from affinemap import AffineMap
from p1_bubble_fem import P1BubbleFE
from reference_element import ReferenceElement


@dataclass
class AssemblerLifting:
    refernce_element: ReferenceElement
    fe_physical: P1BubbleFE
    nu: float
    c_const: float

    def assemble_n(self) -> npt.NDArray[np.float64]:
        ltg_list = [self.fe_physical.ltg_u1, self.fe_physical.ltg_u2]
        n_matrix = np.zeros(
            shape=(
                2 * self.fe_physical.number_dof_w_bubble,
                2 * self.fe_physical.number_dof_w_bubble,
            )
        )
        for i in range(2):
            for l in range(self.fe_physical.triangulation.number_of_triangles):
                triangle_coords = self.fe_physical.discret_points_complete[
                    self.fe_physical.ltg_u1[l]
                ]
                affine_map_l = AffineMap(vertex_coords=triangle_coords)
                for k in range(self.refernce_element.n_dof_bubble):
                    row = ltg_list[i][l][k]
                    if row < 0:
                        continue
                    for j in range(self.refernce_element.n_dof_bubble):
                        col = ltg_list[i][l][j]
                        if col < 0:
                            continue
                        q = self.integral_a(triangle_map=affine_map_l, k=k, j=j)
                        n_matrix[row][col] = n_matrix[row][col] + q
        return n_matrix

    def integral_a(self, triangle_map: AffineMap, k: int, j: int) -> float:
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
                self.refernce_element.gradient_list[k](points[i][0], points[i][1]),
                np.dot(
                    np.dot(
                        triangle_map.inverse_jacobian, triangle_map.inverse_jacobian.T
                    ),
                    self.refernce_element.gradient_list[j](
                        points[i][0], points[i][1]
                    ).T,
                ),
            )
            integral_sum += partial_sum
        integral = (
            1 / 2 * self.nu * integral_sum * abs(triangle_map.determinant_jacobian)
        )
        return integral

    def assemble_d(self) -> npt.NDArray[np.float64]:
        ltg_list = [self.fe_physical.ltg_u1, self.fe_physical.ltg_u2]
        d_matrix = np.zeros(
            shape=(
                self.fe_physical.triangulation.number_of_vertices,
                2 * self.fe_physical.number_dof_w_bubble,
            )
        )
        for i in range(2):
            for l in range(self.fe_physical.triangulation.number_of_triangles):
                triangle_coords = self.fe_physical.discret_points_complete[
                    self.fe_physical.ltg_u1[l]
                ]
                affine_map_l = AffineMap(vertex_coords=triangle_coords)
                for k in range(self.refernce_element.n_dof_lin):
                    row = self.fe_physical.ltg_p[l][k]
                    if row < 0:
                        continue
                    # row = row - self.fe_physical.number_dof_with_bubble * 2
                    for j in range(self.refernce_element.n_dof_bubble):
                        col = ltg_list[i][l][j]
                        if col < 0:
                            continue
                        q = self.integral_b(triangle_map=affine_map_l, k=k, j=j, i=i)
                        d_matrix[row][col] += q
        return d_matrix

    def integral_b(self, triangle_map: AffineMap, k: int, j: int, i: int) -> float:
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
                * self.refernce_element.linear_shape_func_list[k](
                    points[s][0], points[s][1]
                )
                * (
                    np.dot(
                        triangle_map.inverse_jacobian[:, i],
                        self.refernce_element.gradient_list[j](
                            points[s][0], points[s][1]
                        ),
                    )
                )
            )
            integral_sum += partial_sum
        integral = -1 / 2 * integral_sum * abs(triangle_map.determinant_jacobian)
        return integral

    def assemble_s(self) -> npt.NDArray[np.float64]:
        n_matr = self.assemble_n()
        d_matr = self.assemble_d()
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
        return s_matrix

    def get_g_of_bdn(
        self, g_bdn_func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        g_vec = np.zeros(shape=(self.fe_physical.triangulation.num_diri_points, 1))
        for i in reversed(range(1, self.fe_physical.triangulation.num_diri_points + 1)):
            g_vec[self.fe_physical.triangulation.num_diri_points - i] += g_bdn_func(
                self.fe_physical.discret_points_complete[-i][0],
                self.fe_physical.discret_points_complete[-i][1],
            )
        return g_vec

    def assemble_rhs(
        self, g_boundary_func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        g_vec = self.get_g_of_bdn(g_bdn_func=g_boundary_func)
        b_matrix = np.zeros(
            shape=(
                self.fe_physical.number_dof_w_bubble,
                self.fe_physical.triangulation.num_diri_points,
            )
        )
        for l in range(self.fe_physical.triangulation.number_of_triangles):
            triangle_coords = self.fe_physical.discret_points_complete[
                self.fe_physical.ltg_u1[l]
            ]
            affine_map_l = AffineMap(vertex_coords=triangle_coords)
            for k in range(self.refernce_element.n_dof_bubble):
                row = self.fe_physical.ltg_u1[l][k]
                if row < 0:
                    continue
                for j in range(self.refernce_element.n_dof_bubble):
                    col = self.fe_physical.ltg_u1[l][j]
                    if col > -1:
                        continue
                    col += self.fe_physical.triangulation.num_diri_points
                    q = self.integral_a(triangle_map=affine_map_l, k=k, j=j)
                    b_matrix[row][col] += q
        rhs_u1 = np.dot(b_matrix, g_vec)
        rhs_u2_p = np.zeros(
            shape=(
                self.fe_physical.number_dof_w_bubble
                + self.fe_physical.triangulation.number_of_vertices,
                1,
            )
        )
        rhs = np.vstack((rhs_u1, rhs_u2_p))
        return rhs * (-1)
