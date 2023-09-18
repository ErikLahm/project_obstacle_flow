from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt

from affinemap import AffineMap

EPSILON = 1e-6


def assemble_n(
    ref_ltg: npt.NDArray[np.int64],
    grad_sfs: List[Callable[[float, float], npt.NDArray[np.float64]]],
    domain_coords: npt.NDArray[np.float64],
    domain_ltg: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    assert (
        len(grad_sfs) == ref_ltg.shape[1]
    ), "Number of shape functions given does not correspond to num_loc_dof in ltg map."
    num_vel_dof = ref_ltg.max() + 1
    num_slabs = ref_ltg.shape[0]
    num_loc_dof = ref_ltg.shape[1]
    n_matrix = np.zeros(shape=(num_vel_dof, num_vel_dof))
    for j in range(num_slabs):
        aff_map = AffineMap(domain_coords[domain_ltg[j]])
        for k in range(num_loc_dof):
            row = ref_ltg[j, k]
            for l in range(num_loc_dof):
                col = ref_ltg[j, l]
                integral = integral_a(
                    triangle_map=aff_map, k=k, j=l, gradient_list=grad_sfs
                )
                n_matrix[row][col] += integral
    zero_buffer = np.zeros(shape=(num_vel_dof, num_vel_dof))
    upper_half = np.vstack((n_matrix, zero_buffer))
    rank_upper = np.linalg.matrix_rank(upper_half)
    lower_half = np.vstack((zero_buffer, n_matrix))
    n_matrix = np.hstack((upper_half, lower_half))
    # assert np.linalg.matrix_rank(n_matrix) == min(n_matrix.shape), (
    #     f"Laplacian matrix is not full rank: rank(N)={np.linalg.matrix_rank(n_matrix)} "
    #     f"but should be rank(N)=min(m,m)={min(n_matrix.shape)}."
    # )
    return n_matrix


def integral_a(
    triangle_map: AffineMap,
    k: int,
    j: int,
    gradient_list: List[Callable[[float, float], npt.NDArray[np.float64]]],
) -> float:
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
    nu = 1
    integral_sum = 0
    for i, _ in enumerate(weights):
        partial_sum = weights[i] * np.dot(
            gradient_list[k](points[i][0], points[i][1]),
            np.dot(
                np.dot(triangle_map.inverse_jacobian, triangle_map.inverse_jacobian.T),
                gradient_list[j](points[i][0], points[i][1]).T,
            ),
        )
        integral_sum += partial_sum
    integral = 1 / 2 * nu * integral_sum * abs(triangle_map.determinant_jacobian)
    return integral


def add_inital_penalty(
    num_slabs: int, n_matrix: npt.NDArray[np.float64], velo_sf_shape: Tuple[int, int]
) -> npt.NDArray[np.float64]:
    num_velo_dof = (velo_sf_shape[0] * num_slabs + 1) * (velo_sf_shape[1] - 1)
    for diag in range(velo_sf_shape[1] - 1):
        n_matrix[diag][diag] += 1 / EPSILON
        n_matrix[diag + num_velo_dof][diag + num_velo_dof] += 1 / EPSILON
    return n_matrix


def assemble_g(
    velo_ltg: npt.NDArray[np.int64],
    press_ltg: npt.NDArray[np.int64],
    press_sfs: List[Callable[[float, float], float]],
    grad_vel_sfs: List[Callable[[float, float], npt.NDArray[np.float64]]],
    domain_coords: npt.NDArray[np.float64],
    domain_ltg: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    assert (
        len(press_sfs) == press_ltg.shape[1]
    ), "Number of pressure ref.-sf. does not coincide with number of local dof."
    assert (
        len(grad_vel_sfs) == velo_ltg.shape[1]
    ), "Number of velo. grad. ref.-sf. does not coincide with number of local dof."
    assert (
        velo_ltg.shape[0] == press_ltg.shape[0]
    ), "Different numbers of slabs for velocity and pressure ltg."
    num_vel_dof = velo_ltg.max() + 1
    num_pres_dof = press_ltg.max() + 1
    num_slabs = velo_ltg.shape[0]
    num_loc_velo_dof = velo_ltg.shape[1]
    num_loc_pres_dof = press_ltg.shape[1]
    g_matrix = np.zeros(shape=(num_vel_dof, num_pres_dof))
    g_matrix_halfs: List[npt.NDArray[np.float64]] = []
    for component in range(2):
        for j in range(num_slabs):
            aff_map = AffineMap(domain_coords[domain_ltg[j]])
            for k in range(num_loc_velo_dof):
                row = velo_ltg[j][k]
                for l in range(num_loc_pres_dof):
                    col = press_ltg[j][l]
                    integral = integral_b(
                        triangle_map=aff_map,
                        k=l,
                        j=k,
                        i=component,
                        lin_sf=press_sfs,
                        grad_sf=grad_vel_sfs,
                    )
                    g_matrix[row][col] += integral
        g_matrix_halfs.append(g_matrix)
        g_matrix = np.zeros(shape=(num_vel_dof, num_pres_dof))
    g_matrix = np.vstack((g_matrix_halfs[0], g_matrix_halfs[1]))
    assert np.linalg.matrix_rank(g_matrix) == min(g_matrix.shape), (
        f"Gradient matrix is not full rank: rank(G)={np.linalg.matrix_rank(g_matrix)} "
        f"but should be rank(G)=min(m,n)={min(g_matrix.shape)}."
    )
    return g_matrix


def integral_b(
    triangle_map: AffineMap, k: int, j: int, i: int, lin_sf, grad_sf
) -> float:
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
            * lin_sf[k](points[s][0], points[s][1])
            * (
                np.dot(
                    triangle_map.inverse_jacobian[:, i],
                    grad_sf[j](points[s][0], points[s][1]),
                )
            )
        )
        integral_sum += partial_sum
    integral = -1 / 2 * integral_sum * abs(triangle_map.determinant_jacobian)
    return integral


def assemble_rhs(
    num_slabs: int,
    velo_sf_shape: Tuple[int, int],
    press_sf_shape: Tuple[int, int],
    phys_velo_dof_coords: npt.NDArray[np.float64],
    boundary_func: Callable[[float, float], float],
) -> npt.NDArray[np.float64]:
    num_pres_dof = (press_sf_shape[0] * num_slabs + 1) * (press_sf_shape[1] + 1)
    rhs = np.zeros(shape=(2 * phys_velo_dof_coords.shape[0] + num_pres_dof, 1))
    for i in range(velo_sf_shape[1] - 1):
        rhs[i] = (
            boundary_func(phys_velo_dof_coords[i][0], phys_velo_dof_coords[i][1])
            * 1
            / EPSILON
        )
    return rhs
