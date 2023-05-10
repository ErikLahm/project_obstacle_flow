from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from p1_bubble_fem import P1BubbleFE


@dataclass
class Visualiser:
    complete_coeffs: npt.NDArray[np.float64]
    p1_fem: P1BubbleFE

    def separate_coeffs(self) -> list[npt.NDArray[np.float64]]:
        separat_sols = np.vsplit(
            self.complete_coeffs,
            np.array(
                [
                    self.p1_fem.number_dof_w_bubble,
                    2 * self.p1_fem.number_dof_w_bubble,
                ]
            ),
        )
        return separat_sols

    def separate_coeffs_penalty(self) -> list[npt.NDArray[np.float64]]:
        separat_sols = np.vsplit(
            self.complete_coeffs,
            np.array(
                [
                    len(self.p1_fem.discret_points_complete),
                    2 * len(self.p1_fem.discret_points_complete),
                ]
            ),
        )
        return separat_sols

    def get_solution_vel_comp(
        self, component: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        rows = []
        u_sol = np.zeros(shape=(self.p1_fem.number_dof_w_bubble, 1))
        for l in range(self.p1_fem.triangulation.number_of_triangles):
            for k in range(4):
                row = self.p1_fem.ltg_u1[l][k]
                if row in rows:
                    continue
                if row < 0:
                    continue
                if k < 3 and u_sol[row] < 1e-14:
                    u_sol[row] += component[row]
                if k == 3:
                    u_sol[row] += component[row]
                    adder = 0
                    for j in range(3):
                        shape_f = self.p1_fem.ltg_u1[l][j]
                        if shape_f < 0:
                            continue
                        adder += 1 / 3 * component[shape_f]
                    u_sol[row] += adder
                rows.append(row)  # type: ignore
        return u_sol

    def get_solution_vel_comp_penalty(
        self, component: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        rows = []
        u_sol = np.zeros(shape=(len(self.p1_fem.discret_points_complete), 1))
        for l in range(self.p1_fem.triangulation.number_of_triangles):
            for k in range(4):
                row = self.p1_fem.ltg_u1_penalty[l][k]
                if row in rows:
                    continue
                if row < 0:
                    continue
                if k < 3 and u_sol[row] < 1e-14:
                    u_sol[row] += component[row]
                if k == 3:
                    u_sol[row] += component[row]
                    adder = 0
                    for j in range(3):
                        shape_f = self.p1_fem.ltg_u1_penalty[l][j]
                        if shape_f < 0:
                            continue
                        adder += 1 / 3 * component[shape_f]
                    u_sol[row] += adder
                rows.append(row)  # type: ignore
        print(u_sol)
        return u_sol

    def plot_streamline(self) -> None:
        u_1 = self.get_solution_vel_comp(component=self.separate_coeffs()[0])
        u_2 = self.get_solution_vel_comp(component=self.separate_coeffs()[1])
        _, ax = plt.subplots()  # type: ignore
        ax.quiver(  # type: ignore
            self.p1_fem.discret_points_complete[: len(self.p1_fem.dof_coords), 0],
            self.p1_fem.discret_points_complete[: len(self.p1_fem.dof_coords), 1],
            u_1,
            u_2,
        )
        plt.show()  # type: ignore

    def plot_penalty(self) -> None:
        u_1 = self.get_solution_vel_comp_penalty(
            component=self.separate_coeffs_penalty()[0]
        )
        u_2 = self.get_solution_vel_comp_penalty(
            component=self.separate_coeffs_penalty()[1]
        )
        fig, ax = plt.subplots()  # type: ignore
        pc = ax.quiver(  # type: ignore
            self.p1_fem.discret_points_complete[:, 0],
            self.p1_fem.discret_points_complete[:, 1],
            u_1,
            u_2,
            u_1,
            width=0.001,
            headlength=3,
            headaxislength=3,
            cmap="gnuplot",
        )
        ax.set_xlabel("Length")  # type: ignore
        ax.set_ylabel("Radius")  # type: ignore
        fig.colorbar(pc)  # type: ignore
        circle_1 = plt.Circle((2, 0), 0.57, color="grey", alpha=0.2)  # type: ignore
        circle_2 = plt.Circle((3.5, 1), 0.57, color="grey", alpha=0.2)  # type: ignore
        circle_3 = plt.Circle((4.5, -1), 0.57, color="grey", alpha=0.2)  # type: ignore
        ax.add_patch(circle_1)  # type: ignore
        ax.add_patch(circle_2)  # type: ignore
        ax.add_patch(circle_3)  # type: ignore
        # plt.show()

    def plot_pressure(self) -> None:
        p = self.separate_coeffs_penalty()[2]
        _, ax = plt.subplots()  # type: ignore
        ax.scatter(self.p1_fem.triangulation.rect_mesh.sorted_mesh_coords[:, 0], p)  # type: ignore
