"""Evaluate admissibility for different parameter values."""

import numpy as np
import pickle
import sympy
from tqdm import tqdm

import complex_spacetime_metrics.metric as mt


# There are different ways an admissibility condition can be evaluated for different parameters:
#   1. Analytical Roots. There is no angular dependence in the condition and the roots can be solved for in advance
#      in terms of the parameters. This is the case for G > 0
#   2. Analytical Roots Angular. There is an angular dependence, but the roots can still be solved for analytically.
#        The condition has to be evaluated for different theta values separately. A discriminant function needs to be
#        provided to check the limiting behavior of the condition. This is the case for A > 0
#   3. Pointwise Analytical Roots. The roots cannot be solved for analytically for general parameter values. For each
#        evaluation, the roots are solved for for given values of the parameters
#   4. r_tilde Sweep. The roots cannot be solved for analytically in advance or for particular parameter values. A
#        sweep of different r_tilde values greater than r_tilde_plus are evaluated. This is much more approximate


class AnalyticalRootsAngular:

    def admissibility(self, a_vals_grid, r_tilde_plus_vals_grid, theta_val, roots, disrim, filename=None):
        """Determine whether the metric is admissible for given values of the parameters.

        Args:
            a_vals_grid: X-Y grid of a parameters values
            r_tilde_plus_grid: X-Y grid of r_tilde_plus parameter values
            theta_val: Value of theta to evaluate admissibility for
            roots: List of roots of the admissibility condition
            discrim: Leading term in r_tilde of the admissibility condition
            filename: Name of file to save admissible map and parameter grid to

        Returns:
            Grid where each element indicates admissibility for given parameter values

        """
        admissible_map = np.zeros(a_vals_grid.shape, dtype=bool)

        # restrict to given theta
        roots_theta = [root.subs({mt.theta: theta_val}) for root in roots]
        discrim_theta = disrim.subs({mt.theta: theta_val})

        with tqdm(total=np.prod(a_vals_grid.shape)) as pbar:
            for i in range(a_vals_grid.shape[0]):
                for j in range(a_vals_grid.shape[1]):
                    admissible_map[i, j] = self._check_if_admissible(
                        a_vals_grid[i, j], r_tilde_plus_vals_grid[i, j], roots_theta, discrim_theta
                    )
                    pbar.update(1)

        if filename:
            file_obj = {
                'a_vals': a_vals_grid,
                'r_tilde_plus_vals': r_tilde_plus_vals_grid,
                'theta_val': theta_val,
                'admissible_map': admissible_map
            }
            with open(filename, 'wb') as f:
                pickle.dump(file_obj, f)

        return admissible_map

    def _check_if_admissible(self, a_val, r_tilde_plus_val, roots, discrim):
        # check the sign of leading term in r_tilde
        D = discrim.subs({mt.a: a_val, mt.r_tilde_plus: r_tilde_plus_val}).evalf()
        if D < 0:
            return False

        for root in roots:
            r = root.subs({mt.a: a_val, mt.r_tilde_plus: r_tilde_plus_val}).evalf()
            if r != sympy.nan:
                if np.abs(r.coeff(sympy.I)) < 1e-10:
                    if r.is_real:
                        real_part = r
                    else:
                        real_part = r.args[0]
                    if real_part > r_tilde_plus_val:
                        return False

        return True
