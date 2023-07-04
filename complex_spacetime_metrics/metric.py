"""Generate metrics and metric coefficients."""

import sympy


# coordinates
r_tilde, theta = sympy.symbols("\\tilde{r} theta", real=True)

# differentials
dt, dr_tilde, dphi, dtau = sympy.symbols(r"dt d\tilde{r} d\phi d\tau")

# functions of coordinates
delta_r, delta_theta, W = sympy.symbols("Delta_r Delta_theta W")

# parameters
a, xi = sympy.symbols("a Xi")


def metric_3d():
    metric = -delta_r / W * (dt - a * sympy.sin(theta) ** 2 * dphi / xi) ** 2
    metric += delta_theta * sympy.sin(theta) ** 2 / W * (a * dt - (r_tilde ** 2 + a ** 2) * dphi / xi) ** 2
    metric += W * dr_tilde ** 2 / delta_r
    metric

    return metric


def wick_rotate(metric):
    dt_val = sympy.I * dtau

    euclidean_metric = metric.subs({dt: dt_val}).expand().collect([dtau ** 2, dtau * dphi, dphi ** 2])

    return euclidean_metric


# omega = sympy.symbols("Omega")
# dphi_val = dphi_tilde + omega * dt_val
# dphi_val

# euclidean_metric_2 = euclidean_metric.subs({dphi: dphi_val}).expand().collect([dtau ** 2, dtau * dphi_tilde, dphi_tilde ** 2])
# euclidean_metric_2


# metric_list_matrix = [[0 for i in range(3)] for i in range(3)]

# # tau
# metric_list_matrix[0][0] = euclidean_metric_2.coeff(dtau ** 2)
# metric_list_matrix[0][1] = euclidean_metric_2.coeff(dtau * dr_tilde) / 2
# metric_list_matrix[0][2] = euclidean_metric_2.coeff(dtau * dphi_tilde) / 2

# # r_tilde
# metric_list_matrix[1][0] = euclidean_metric_2.coeff(dr_tilde * dtau) / 2
# metric_list_matrix[1][1] = euclidean_metric_2.coeff(dr_tilde ** 2)
# metric_list_matrix[1][2] = euclidean_metric_2.coeff(dr_tilde * dphi_tilde) / 2

# # phi_tilde
# metric_list_matrix[2][0] = euclidean_metric_2.coeff(dphi_tilde * dtau) / 2
# metric_list_matrix[2][1] = euclidean_metric_2.coeff(dphi_tilde * dr_tilde) / 2
# metric_list_matrix[2][2] = euclidean_metric_2.coeff(dphi_tilde ** 2)

# g_mat = sympy.Matrix(metric_list_matrix)
# g_mat.simplify()
# g_mat

# z_r, z_i = sympy.symbols("z_r z_i")
# delta_r_val = z_r + sympy.I * z_i
# delta_r_val

# real_delta_r = r_tilde ** 4 - r_tilde_plus ** 4 + (r_tilde_plus - r_tilde) ** 2 * (a + 1) ** 2 + 2 * a * (r_tilde_plus ** 2 - r_tilde ** 2)
# real_delta_r

# imag_delta_r = 2 * (r_tilde_plus - r_tilde) * (a + 1) * (r_tilde_plus ** 2 - a)
# imag_delta_r

# delta_theta_val = 1 - a ** 2 * sympy.cos(theta) ** 2
# xi_val = 1 - a ** 2
# W_val = r_tilde ** 2 + a ** 2 * sympy.cos(theta) ** 2
# omega_val = a * xi / (r_tilde_plus ** 2 + a ** 2)

# g_mat_subs = g_mat.subs({delta_r: delta_r_val})
# g_mat_subs
