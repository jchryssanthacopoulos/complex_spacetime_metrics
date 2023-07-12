"""Evaluate admissibility for different parameter values."""

from multiprocessing import Pool
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


class AdmissibilityEvaluator:
    """Base class for evaluating admissibility."""

    def save_to_file(self, filename, a_vals_grid, r_tilde_plus_vals_grid, theta_val, admissible_map):
        file_obj = {
            'a_vals': a_vals_grid,
            'r_tilde_plus_vals': r_tilde_plus_vals_grid,
            'theta_val': theta_val,
            'admissible_map': admissible_map
        }
        with open(filename, 'wb') as f:
            pickle.dump(file_obj, f)


class AnalyticalRoots(AdmissibilityEvaluator):

    def admissibility(self, pgrid, roots, discrim, theta_val=None, filename=None):
        """Determine whether the metric is admissible for given values of the parameters.

        Args:
            pgrid: Object representing parameter grid
            roots: List of roots of the admissibility condition
            discrim: Leading term in r_tilde of the admissibility condition
            theta_val: Value of theta to evaluate admissibility for
            filename: Name of file to save admissible map and parameter grid to

        Returns:
            Grid where each element indicates admissibility for given parameter values

        """
        # restrict to given theta
        if theta_val is not None:
            roots = [root.subs({mt.theta: theta_val}) for root in roots]
            discrim = discrim.subs({mt.theta: theta_val})

        admissible_map = np.zeros(pgrid.grid_size, dtype=bool)

        with tqdm(total=np.prod(pgrid.grid_size)) as pbar:
            for i in range(pgrid.grid_size[0]):
                for j in range(pgrid.grid_size[1]):
                    admissible_map[i, j] = self._check_if_admissible(
                        pgrid.a_vals_grid[i, j], pgrid.r_tilde_plus_vals_grid[i, j], roots, discrim
                    )
                    pbar.update(1)

        if filename:
            self.save_to_file(filename, pgrid.a_vals_grid, pgrid.r_tilde_plus_vals_grid, theta_val, admissible_map)

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


class PointwiseAnalyticalRoots(AdmissibilityEvaluator):

    def admissibility(self, pgrid, theta_val, condition, discrim, filename=None):
        """Determine whether the metric is admissible for given values of the parameters.

        Args:
            pgrid: Object representing parameter grid
            theta_val: Value of theta to evaluate admissibility for
            condition: Admissibility condition that must be greater than zero
            discrim: Leading term in r_tilde of the admissibility condition
            filename: Name of file to save admissible map and parameter grid to

        Returns:
            Grid where each element indicates admissibility for given parameter values

        """
        # restrict to given theta
        condition_theta = condition.subs({mt.theta: theta_val})
        discrim_theta = discrim.subs({mt.theta: theta_val})

        admissible_map = np.zeros(pgrid.grid_size, dtype=bool)

        with tqdm(total=np.prod(pgrid.grid_size)) as pbar:
            for i in range(pgrid.grid_size[0]):
                for j in range(pgrid.grid_size[1]):
                    admissible_map[i, j] = self._check_if_admissible(
                        pgrid.a_vals_grid[i, j], pgrid.r_tilde_plus_vals_grid[i, j], condition_theta, discrim_theta
                    )
                    pbar.update(1)

        if filename:
            self.save_to_file(filename, pgrid.a_vals_grid, pgrid.r_tilde_plus_vals_grid, theta_val, admissible_map)

        return admissible_map

    def admissibility_parallel(self, pgrid, theta_val, condition, discrim, filename=None, proc_count=10):
        """Determine whether the metric is admissible for given values of the parameters using multithreading.

        Args:
            pgrid: Object representing parameter grid
            theta_val: Value of theta to evaluate admissibility for
            condition: Admissibility condition that must be greater than zero
            discrim: Leading term in r_tilde of the admissibility condition
            filename: Name of file to save admissible map and parameter grid to
            proc_count: Number of processes to run

        Returns:
            Grid where each element indicates admissibility for given parameter values

        """
        # restrict to given theta
        condition_theta = condition.subs({mt.theta: theta_val})
        discrim_theta = discrim.subs({mt.theta: theta_val})

        # construct flat list of sweep parameters
        args_parallel = [
            (a_val, r_tilde_plus_val, condition_theta, discrim_theta)
            for a_val, r_tilde_plus_val in zip(pgrid.a_vals_grid.flatten(), pgrid.r_tilde_plus_vals_grid.flatten())
        ]

        with Pool(proc_count) as pool:
            results = pool.starmap(self._check_if_admissible, tqdm(args_parallel, total=len(args_parallel)))

        admissible_map = np.array(results).reshape(pgrid.grid_size)

        if filename:
            self.save_to_file(filename, pgrid.a_vals_grid, pgrid.r_tilde_plus_vals_grid, theta_val, admissible_map)

        return admissible_map

    def _check_if_admissible(self, a_val, r_tilde_plus_val, condition, discrim):
        # check the sign of leading term in r_tilde
        D = discrim.subs({mt.a: a_val, mt.r_tilde_plus: r_tilde_plus_val}).evalf()
        if D < 0:
            return False

        # numerically compute the roots
        cond_params = condition.subs({mt.a: a_val, mt.r_tilde_plus: r_tilde_plus_val})
        roots = sympy.solve(cond_params, mt.r_tilde)

        if not roots:
            print(f"No roots found for a = {a_val}, r_tilde_plus = {r_tilde_plus_val}")

        for r in roots:
            if r != sympy.nan:
                if np.abs(r.coeff(sympy.I)) < 1e-10:
                    if r.is_real:
                        real_part = r
                    else:
                        real_part = r.args[0]
                    if real_part > r_tilde_plus_val:
                        return False

        return True
