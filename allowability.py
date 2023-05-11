"""Allowability condition as a function of theta."""

import numpy as np
import sympy


r_tilde, delta_r, delta_theta, xi = sympy.symbols("\\tilde{r} Delta_r Delta_theta Xi")
a, r_tilde_plus, theta, omega = sympy.symbols("a \\tilde{r}_+ theta Omega")


def check_if_allowable_theta(poly_cond, A_coeff, theta_val, a_val, r_tilde_plus_val):
    # check the sign of leading term in r_tilde
    A = A_coeff.subs({a: a_val, r_tilde_plus: r_tilde_plus_val, theta: theta_val}).evalf()
    if A < 0:
         return False

    # numerically compute the roots
    p = poly_cond.subs({a: a_val, r_tilde_plus: r_tilde_plus_val, theta: theta_val})
    roots = sympy.solve(p, r_tilde)

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
