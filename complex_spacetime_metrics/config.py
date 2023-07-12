"""Stores configuration parameters."""

import numpy as np

from complex_spacetime_metrics.parameter_grid import ParameterGrid


# different theta values to sweep over
thetas = {
    "theta_0": 0,
    "theta_pi_4": np.pi / 4,
    "theta_pi_2": np.pi / 2,
    "theta_3_pi_4": 3 * np.pi / 4,
    "theta_pi": np.pi
}


# this seems better
thetas_fine_grain = {
    "theta_0": 0,
    "theta_pi_8": np.pi / 8,
    "theta_pi_4": np.pi / 4
}


thetas_fine_grain_2 = {
    "theta_pi_18": np.pi / 18,
    "theta_pi_9": np.pi / 9
}


# different parameter grids
grids = {
    'coarse': ParameterGrid(delta_a=0.05, delta_r_tilde_plus=0.1),
    'medium': ParameterGrid(delta_a=0.01, delta_r_tilde_plus=0.01),
    'fine': ParameterGrid(delta_a=0.005, delta_r_tilde_plus=0.01)
}
