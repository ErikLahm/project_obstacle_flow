import numpy as np
from scipy.linalg import svd

from assembling_lin_sys import AssemblerPenalty
from p1_bubble_fem import P1BubbleFE
from rectangular_grid import RectangularGrid
from reference_element import ReferenceElement
from solution_visualiser import Visualiser
from triangulation import Triangulation

NU = 100
PRESSURE_GRAD = -100
C_CONST = 1 / 2 * 1 / NU * (-PRESSURE_GRAD)
RADIUS = 1
LENGTH = 10


def solve_svd(A, b):
    # compute svd of A
    U, s, Vh = svd(A)

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T, b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = np.dot(np.diag(1 / s), c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(Vh.conj().T, w)
    return x


def g_1(x_1: float, x_2: float) -> float:
    if x_1 < 1e-14:
        return C_CONST * (RADIUS - x_2) * (x_2 + RADIUS)
    else:
        return 0


def main():
    x_domain = (0, LENGTH)
    y_domain = (-RADIUS, RADIUS)
    x_discret = 4
    y_discret = 4

    meshgrid = RectangularGrid(
        x_domain=x_domain,
        y_domain=y_domain,
        x_discretisation=x_discret,
        y_discretisation=y_discret,
    )
    triangulation = Triangulation(rect_mesh=meshgrid)
    p1_bubble_fem = P1BubbleFE(triangulation=triangulation)
    print(f"old ltg: {triangulation.delauny_tri.simplices}")
    print(f"new ltg: {p1_bubble_fem.ltg_w_bubble_penalty()}")
    print(p1_bubble_fem.ltg_u1)
    # print(p1_bubble_fem.ltg_u2)
    print(p1_bubble_fem.ltg_p)
    triangulation.plot_triangulation(
        triangulation.rect_mesh.sorted_mesh_coords,
        p1_bubble_fem.discret_points_complete,
    )

    ref_elem = ReferenceElement()
    assembler = AssemblerPenalty(
        reference_element=ref_elem, fe_physical=p1_bubble_fem, nu=1, c_const=1
    )
    n = assembler.assemble_n()
    rank_n = np.linalg.matrix_rank(n)
    d = assembler.assemble_d()
    d_transpose = np.transpose(d)
    n_inverse = np.linalg.inv(n)
    schur = np.dot(d, np.dot(n_inverse, d_transpose))
    rank_schur = np.linalg.matrix_rank(schur)
    s = assembler.assemble_s()
    rank_s = np.linalg.matrix_rank(s)
    rhs = assembler.assemble_rhs(g_boundary_func=g_1)
    rhs_velocity = np.copy(rhs[: 2 * p1_bubble_fem.number_dof_w_bubble])
    rhs_pressure_system = np.dot(d, np.dot(n_inverse, rhs_velocity))
    u_p_sol = np.linalg.solve(s, rhs)
    u_p_sol_svd = solve_svd(s, rhs)
    pressure_svd = solve_svd(schur, rhs_pressure_system)
    visualiser = Visualiser(complete_coeffs=u_p_sol, p1_fem=p1_bubble_fem)
    visualiser.plot_streamline()
    separat = visualiser.separate_coeffs()
    u_1sol = visualiser.get_solution_vel_comp(separat[0])
    u_2sol = visualiser.get_solution_vel_comp(separat[1])
    print(n, d, s, u_p_sol)


main()
