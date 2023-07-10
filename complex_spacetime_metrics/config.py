"""Stores configuration parameters."""

import numpy as np


# different theta values to sweep over
thetas = {
    "theta_0": 0,
    "theta_pi_12": np.pi / 12,
    "theta_pi_6": np.pi / 6,
    "theta_pi_4": np.pi / 4,
    "theta_pi_2": np.pi / 2,
    "theta_3_pi_4": 3 * np.pi / 4,
    "theta_pi": np.pi
}


# parameter grid
a_vals = np.arange(0, 1.005, 0.005)
r_tilde_plus_vals = np.arange(-1, 1.01, 0.01)
a_vals_grid, r_tilde_plus_vals_grid = np.meshgrid(a_vals, r_tilde_plus_vals)
