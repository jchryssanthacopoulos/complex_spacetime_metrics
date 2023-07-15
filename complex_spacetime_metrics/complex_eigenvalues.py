"""Allowability condition using complex eigenvalues."""

# import numpy as np
import sympy

from complex_spacetime_metrics import metric as metric_utils


# # variable declarations
# A, B, C, D, E, F, G, H = sympy.symbols("A B C D E F G H")
# r_tilde, theta, r_tilde_plus, a = sympy.symbols("\\tilde{r} theta \\tilde{r} a")


# # beta definition
# beta = (A * F + B * E - 2 * C * D) ** 2 - 4 * (A * E - C ** 2) * (B * F - D ** 2)


def get_angles(metric_coefficients):
    """Get angles of complex eigenvalues given metric coefficients."""
    A = metric_coefficients[metric_utils.A]
    B = metric_coefficients[metric_utils.B]
    C = metric_coefficients[metric_utils.C]
    D = metric_coefficients[metric_utils.D]
    E = metric_coefficients[metric_utils.E]
    F = metric_coefficients[metric_utils.F]
    G = metric_coefficients[metric_utils.G]
    H = metric_coefficients[metric_utils.H]

    beta = (A * F + B * E - 2 * C * D) ** 2 - 4 * (A * E - C ** 2) * (B * F - D ** 2)

    # beta < 0
    lambda_plus_real_val = 1 + sympy.sqrt(-beta) / (2 * (A * E - C ** 2))
    lambda_minus_real_val = 1 - sympy.sqrt(-beta) / (2 * (A * E - C ** 2))

    lambda_imag_val = (A * F + B * E - 2 * C * D) / (2 * (A * E - C ** 2))

    # beta_val >= 0
    lambda_plus_imag_val = (A * F + B * E - 2 * C * D + sympy.sqrt(beta)) / (2 * (A * E - C ** 2))
    lambda_minus_imag_val = (A * F + B * E - 2 * C * D - sympy.sqrt(beta)) / (2 * (A * E - C ** 2))

    angle_1 = sympy.Piecewise(
        (sympy.Abs(sympy.atan2(lambda_imag_val, lambda_plus_real_val)).evalf(), beta < 0),
        (sympy.Abs(sympy.atan2(lambda_plus_imag_val, 1)).evalf(), beta >= 0)
    )

    angle_2 = sympy.Piecewise(
        (sympy.Abs(sympy.atan2(lambda_imag_val, lambda_minus_real_val)).evalf(), beta < 0),
        (sympy.Abs(sympy.atan2(lambda_minus_imag_val, 1)).evalf(), beta >= 0)
    )

    angle_3 = sympy.Abs(sympy.atan2(H / G, 1))

    return angle_1, angle_2, angle_3


# def cond_sym(theta_val, r_tilde_plus_val, a_val, A_val, B_val, C_val, D_val, E_val, F_val, G_val, H_val):
#     """Get angles for given theta, r_tilde_plus, and a_val.

#     Args:
#         theta_val: Value of theta
#         r_tilde_plus_val: Value of r_tilde_plus
#         a_val: Value of a
#         A_val: Value of A
#         B_val: Value of B
#         C_val: Value of C
#         D_val: Value of D
#         E_val: Value of E
#         F_val: Value of F
#         G_val: Value of G
#         H_val: Value of H

#     Returns:
#         Three angles

#     """
#     vals_dict = {}
#     for var_name, var_val in [(A, A_val), (B, B_val), (C, C_val), (D, D_val), (E, E_val), (F, F_val), (G, G_val), (H, H_val)]:
#         vals_dict[var_name] = var_val.subs({theta: theta_val, r_tilde_plus: r_tilde_plus_val, a: a_val})

#     beta_val = beta.subs(vals_dict).simplify().evalf()
#     lambda_r = (H / G).subs(vals_dict).evalf()

#     # beta_val < 0:
#     lambda_plus_real_val = 1 + sympy.sqrt(-beta_val) / (2 * (A * E - C ** 2))
#     lambda_plus_real_val = lambda_plus_real_val.subs(vals_dict).simplify().evalf()
#     lambda_minus_real_val = 1 - sympy.sqrt(-beta_val) / (2 * (A * E - C ** 2))
#     lambda_minus_real_val = lambda_minus_real_val.subs(vals_dict).simplify().evalf()

#     lambda_imag_val = (A * F + B * E - 2 * C * D) / (2 * (A * E - C ** 2))
#     lambda_imag_val = lambda_imag_val.subs(vals_dict).simplify().evalf()

#     # beta_val >= 0
#     lambda_plus_imag_val = (A * F + B * E - 2 * C * D + sympy.sqrt(beta_val)) / (2 * (A * E - C ** 2))
#     lambda_plus_imag_val = lambda_plus_imag_val.subs(vals_dict).simplify().evalf()
#     lambda_minus_imag_val = (A * F + B * E - 2 * C * D - sympy.sqrt(beta_val)) / (2 * (A * E - C ** 2))
#     lambda_minus_imag_val = lambda_minus_imag_val.subs(vals_dict).simplify().evalf()

#     angle_1 = sympy.Piecewise(
#         (sympy.Abs(sympy.atan2(lambda_imag_val, lambda_plus_real_val)).evalf(), beta_val < 0),
#         (sympy.Abs(sympy.atan2(lambda_plus_imag_val, 1)).evalf(), beta_val >= 0)
#     )

#     angle_2 = sympy.Piecewise(
#         (sympy.Abs(sympy.atan2(lambda_imag_val, lambda_minus_real_val)).evalf(), beta_val < 0),
#         (sympy.Abs(sympy.atan2(lambda_minus_imag_val, 1)).evalf(), beta_val >= 0)
#     )

#     angle_3 = sympy.Abs(sympy.atan2(lambda_r, 1))

#     return angle_1, angle_2, angle_3


# def is_allowable(theta_val, r_tilde_plus_val, a_val, A_val, B_val, C_val, D_val, E_val, F_val, G_val, H_val):
#     num_r_tilde_evals = 200
#     r_tilde_multiplers = np.linspace(0.001, 4, num_r_tilde_evals)

#     # get angles as a function of r_tilde for given parameter values
#     angle_1, angle_2, angle_3 = cond_sym(theta_val, r_tilde_plus_val, a_val, A_val, B_val, C_val, D_val, E_val, F_val, G_val, H_val)

#     for r_tilde_mult in r_tilde_multiplers:
#         r_tilde_val = r_tilde_plus_val + r_tilde_mult * np.abs(r_tilde_plus_val)

#         angle_1_val = angle_1.subs({r_tilde: r_tilde_val}).evalf()
#         angle_2_val = angle_2.subs({r_tilde: r_tilde_val}).evalf()
#         angle_3_val = angle_3.subs({r_tilde: r_tilde_val}).evalf()

#         try:
#             angle_1_val = float(angle_1_val)
#             angle_2_val = float(angle_2_val)
#             angle_3_val = float(angle_3_val)
#             if angle_1_val + angle_2_val + angle_3_val >= np.pi - 1e-8:
#                 return False
#         except:
#             # continue
#             return False

#     return True
