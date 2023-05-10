from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


@dataclass
class ReferenceElement:
    """
    This class describes the structure of the reference element with regard to the
    shape functions and their gradient.
    """

    n_dof_bubble: int = 4
    n_dof_lin: int = 3

    @property
    def gradient_list(self) -> list[Callable[[float, float], npt.NDArray[np.float64]]]:
        return [self.grad_phi_1, self.grad_phi_2, self.grad_phi_3, self.grad_phi_4]

    @property
    def linear_shape_func_list(self) -> list[Callable[[float, float], float]]:
        return [self.phi_1, self.phi_2, self.phi_3]

    @property
    def bubble_shape_func_list(self) -> list[Callable[[float, float], float]]:
        return [self.phi_1, self.phi_2, self.phi_3, self.phi_4]

    def phi_1(self, x_1: float, x_2: float) -> float:
        return 1 - x_1 - x_2

    def phi_2(self, x_1: float, x_2: float) -> float:
        return x_1

    def phi_3(self, x_1: float, x_2: float) -> float:
        return x_2

    def phi_4(self, x_1: float, x_2: float) -> float:
        return 27 * (1 - x_1 - x_2) * (x_1) * (x_2)

    def grad_phi_1(self, x_1: float, x_2: float) -> npt.NDArray[np.float64]:
        return np.array([-1, -1])

    def grad_phi_2(self, x_1: float, x_2: float) -> npt.NDArray[np.float64]:
        return np.array([1, 0])

    def grad_phi_3(self, x_1: float, x_2: float) -> npt.NDArray[np.float64]:
        return np.array([0, 1])

    def grad_phi_4(self, x_1: float, x_2: float) -> npt.NDArray[np.float64]:
        grad = np.array(
            [
                27 * (x_2 - 2 * x_1 * x_2 - x_2**2),
                27 * (x_1 - x_1**2 - 2 * x_1 * x_2),
            ]
        )
        return grad
