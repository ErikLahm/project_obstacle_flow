import matplotlib.pyplot as plt
import numpy as np

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
LENGTH = 7


def g_1(x_1: float, x_2: float) -> float:
    if x_1 < 1e-14:
        return C_CONST * (RADIUS - x_2) * (x_2 + RADIUS)
    else:
        return 0


def chi(x_1: float, x_2: float) -> float:
    value_1 = (x_1 - 2) ** 2 + x_2**2
    value_2 = (x_1 - 3.5) ** 2 + (x_2 - RADIUS) ** 2
    value_3 = (x_1 - 4.5) ** 2 + (x_2 + RADIUS) ** 2
    if value_1 <= RADIUS / 3 or value_2 <= RADIUS / 3 or value_3 <= RADIUS / 3:
        return 1
    else:
        return 0


def main():
    x_domain = (0, LENGTH)
    y_domain = (-RADIUS, RADIUS)
    x_discret = 50
    y_discret = 50

    meshgrid = RectangularGrid(
        x_domain=x_domain,
        y_domain=y_domain,
        x_discretisation=x_discret,
        y_discretisation=y_discret,
    )
    triangulation = Triangulation(rect_mesh=meshgrid)
    p1_bubble_fem = P1BubbleFE(triangulation=triangulation)
    # triangulation.plot_triangulation(
    #     triangulation.rect_mesh.sorted_mesh_coords,
    #     p1_bubble_fem.discret_points_complete,
    # )
    ref_elem = ReferenceElement()
    assembler = AssemblerPenalty(
        reference_element=ref_elem, fe_physical=p1_bubble_fem, nu=1, c_const=1
    )
    s = assembler.assemble_s_penalty(chi)
    rhs = assembler.rhs_penalty(g_boundary_func=g_1)
    u_p_sol = np.linalg.solve(s, rhs)
    visualizer = Visualiser(u_p_sol, p1_bubble_fem)
    visualizer.plot_penalty()
    # visualizer.plot_pressure()
    plt.show()  # type: ignore


main()
