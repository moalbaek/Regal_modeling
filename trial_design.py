"""Decision-boundary primitives used by the legacy reproducibility audit.

The current v1 final test remains unchanged. This module uses the classical discrete-look
O'Brien-Fleming ``c / sqrt(t)`` boundary so the interim-crossing characterization can be
regenerated from committed code. It is not the protocol's Lan-DeMets O'Brien-Fleming
alpha-spending construction; implementing that spending function remains v2 work.
"""

from functools import lru_cache
from math import exp, pi, sqrt
from statistics import NormalDist

import numpy as np


def _bivariate_normal_cdf(x, y, rho, quadrature_order=96):
    """P(X <= x, Y <= y) for standard normals with correlation ``rho``.

    Conditional-normal integration with Gauss-Legendre quadrature avoids adding a
    SciPy dependency. The lower integration limit of -9 is negligible at the
    precision needed for clinical-trial boundaries.
    """

    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    lower, upper = -9.0, float(x)
    points = 0.5 * (nodes + 1.0) * (upper - lower) + lower
    normal = NormalDist()
    denom = sqrt(1.0 - rho * rho)
    values = np.fromiter(
        (
            exp(-point * point / 2.0)
            / sqrt(2.0 * pi)
            * normal.cdf((y - rho * point) / denom)
            for point in points
        ),
        dtype=float,
        count=len(points),
    )
    return float(0.5 * (upper - lower) * np.dot(weights, values))


@lru_cache(maxsize=None)
def _solve_obrien_fleming_two_look(alpha, interim_information):
    """Solve and cache the expensive scalar boundary calculation."""

    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not 0.0 < interim_information < 1.0:
        raise ValueError("interim_information must be between 0 and 1")

    rho = sqrt(interim_information)

    def crossing_probability(constant):
        interim = constant / sqrt(interim_information)
        no_cross = _bivariate_normal_cdf(interim, constant, rho)
        return 1.0 - no_cross

    lower, upper = 1.0, 4.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if crossing_probability(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    final = 0.5 * (lower + upper)
    return final / sqrt(interim_information), final


def obrien_fleming_two_look(alpha=0.025, interim_information=0.75):
    """Return classical one-sided two-look O'Brien-Fleming efficacy boundaries.

    Boundaries have the form ``c / sqrt(t)`` at information fraction ``t`` and
    ``c`` at the final look. ``c`` is calibrated so the probability of crossing
    either correlated normal boundary under the null equals ``alpha``. This
    classical construction differs slightly from Lan-DeMets O'Brien-Fleming
    alpha spending; see the module docstring.

    The expensive solve is cached, while each call returns a fresh dict so a
    caller cannot mutate cached state.
    """

    alpha = float(alpha)
    interim_information = float(interim_information)
    interim, final = _solve_obrien_fleming_two_look(alpha, interim_information)
    return {
        "interim_z": interim,
        "final_z": final,
        "alpha": alpha,
        "interim_information": interim_information,
    }
