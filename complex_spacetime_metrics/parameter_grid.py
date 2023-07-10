"""Contains a class for representing parameter grids."""

import numpy as np


class ParameterGrid:
    """Class for representing 2D parameter grids."""

    # hardcode ranges
    a_range = [0, 1]
    r_tilde_plus_range = [-1, 1]

    def __init__(self, delta_a: float, delta_r_tilde_plus: float):
        """Initialize grid.

        Args:
            delta_a: Spacing for parameter a
            delta_r_tilde_plus: Spacing for parameter r_tilde_plus

        """
        self.delta_a = delta_a
        self.delta_r_tilde_plus = delta_r_tilde_plus

        self.a_vals = np.arange(self.a_range[0], self.a_range[1] + delta_a, delta_a)
        self.r_tilde_plus_vals = np.arange(
            self.r_tilde_plus_range[0], self.r_tilde_plus_range[1] + delta_r_tilde_plus, delta_r_tilde_plus
        )

        # generate grid
        self.a_vals_grid, self.r_tilde_plus_vals_grid = np.meshgrid(self.a_vals, self.r_tilde_plus_vals)

        self.grid_size = self.a_vals_grid.shape
