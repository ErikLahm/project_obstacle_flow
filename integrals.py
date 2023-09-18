from typing import Callable

import numpy as np
import numpy.typing as npt

WEIGHTS = [
    0.317460317460320 / 4,
    0.317460317460319 / 4,
    0.555555555555555 / 4,
    0.555555555555555 / 4,
    0.555555555555555 / 4,
    0.555555555555555 / 4,
    1.142857142857139 / 4,
]

POINTS = [
    (0.5, 0.9830458915396469),
    (0.5, 0.016954108460353),
    (0.8872983346207419, 0.21132486540518702),
    (0.8872983346207419, 0.788675134594813),
    (0.11270166537925802, 0.788675134594813),
    (0.11270166537925802, 0.21132486540518702),
    (0.5, 0.5),
]


def get_det_jacob(jacobian: npt.NDArray[np.float64]) -> float:
    det: float = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    return det


def integral_a(
    jacobian: Callable[[float, float], npt.NDArray[np.float64]],
    grad_sf_k: Callable[[float, float], npt.NDArray[np.float64]],
    grad_sf_j: Callable[[float, float], npt.NDArray[np.float64]],
) -> float:
    integral_sum = 0
    for i, weight in enumerate(WEIGHTS):
        inv_jacob = np.linalg.inv(jacobian(POINTS[i][0], POINTS[i][1]))
        jac_det = get_det_jacob(jacobian(POINTS[i][0], POINTS[i][1]))
        partial_sum = (
            weight
            * np.dot(
                grad_sf_k(POINTS[i][0], POINTS[i][1]),
                np.dot(
                    np.dot(inv_jacob, inv_jacob.T),
                    grad_sf_j(POINTS[i][0], POINTS[i][1]).T,
                ),
            )
            * abs(jac_det)
        )
        integral_sum += partial_sum
    integral = integral_sum
    return integral


def integral_b(
    jacobian: Callable[[float, float], npt.NDArray[np.float64]],
    sf_k: Callable[[float, float], float],
    grad_sf_j: Callable[[float, float], npt.NDArray[np.float64]],
    component: int,
) -> float:
    integral_sum = 0
    for s, weight in enumerate(WEIGHTS):
        inv_jacob = np.linalg.inv(jacobian(POINTS[s][0], POINTS[s][1]))
        jac_det = get_det_jacob(jacobian(POINTS[s][0], POINTS[s][1]))
        partial_sum = (
            weight
            * sf_k(POINTS[s][0], POINTS[s][1])
            * (
                np.dot(
                    inv_jacob[:, component],
                    grad_sf_j(POINTS[s][0], POINTS[s][1]),
                )
            )
            * abs(jac_det)
        )
        integral_sum += partial_sum
    integral = integral_sum
    return integral
