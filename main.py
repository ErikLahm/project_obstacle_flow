from p1_bubble_fem import P1BubbleFEM
from rectangular_grid import RectangularGrid
from triangulation import Triangulation


def main():
    x_domain = (0, 10)
    y_domain = (-2, 2)
    x_discret = 4
    y_discret = 4

    meshgrid = RectangularGrid(
        x_domain=x_domain,
        y_domain=y_domain,
        x_discretisation=x_discret,
        y_discretisation=y_discret,
    )
    triangulation = Triangulation(rect_mesh=meshgrid)
    p1_bubble_fem = P1BubbleFEM(triangulation=triangulation)
    print(p1_bubble_fem.ltg_w_bubble())
    triangulation.plot_triangulation(
        triangulation.rect_mesh.sorted_mesh_coords,
    )


main()
