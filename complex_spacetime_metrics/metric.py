"""Generate metrics and metric coefficients."""

import sympy


# coordinates
r_tilde, theta = sympy.symbols("\\tilde{r} theta", real=True)

# differentials
dt, dr_tilde, dphi, dphi_tilde, dtau = sympy.symbols(r"dt d\tilde{r} d\phi d\tilde{\phi} d\tau")

# functions of coordinates
delta_r = sympy.symbols("Delta_r")
delta_theta, W = sympy.symbols("Delta_theta W", real=True)

# parameters
a, xi, omega, r_tilde_plus = sympy.symbols("a Xi Omega \\tilde{r}_+", real=True)

# real and imaginary metric coefficients
A, B, C, D, E, F, G, H = sympy.symbols("A B C D E F G H")


def metric_3d():
    metric = -delta_r / W * (dt - a * sympy.sin(theta) ** 2 * dphi / xi) ** 2
    metric += delta_theta * sympy.sin(theta) ** 2 / W * (a * dt - (r_tilde ** 2 + a ** 2) * dphi / xi) ** 2
    metric += W * dr_tilde ** 2 / delta_r
    metric

    return metric


def wick_rotate_omega_shift(metric):
    """Perform Wick rotation, followed by omega shift."""
    dt_val = sympy.I * dtau
    euclidean_metric = metric.subs({dt: dt_val}).expand().collect([dtau ** 2, dtau * dphi, dphi ** 2])

    dphi_val = dphi_tilde + omega * dt_val
    euclidean_metric = euclidean_metric.subs({dphi: dphi_val}).expand().collect(
        [dtau ** 2, dtau * dphi_tilde, dphi_tilde ** 2]
    )

    return euclidean_metric


def omega_shift_wick_rotate(metric):
    """Perform omega shift, followed by Wick rotation."""
    dphi_val = dphi_tilde + omega * dt
    euclidean_metric = metric.subs({dphi: dphi_val}).expand().collect(
        [dt ** 2, dt * dphi_tilde, dphi_tilde ** 2]
    )

    dt_val = sympy.I * dtau
    euclidean_metric = euclidean_metric.subs({dt: dt_val}).expand().collect([dtau ** 2, dtau * dphi, dphi ** 2])

    return euclidean_metric


def metric_3d_to_matrix(metric):
    """Convert 3D metric into a matrix.

    Assumes metric has tau, phi_tilde, r_tilde components

    """
    metric_matrix = [[0 for i in range(3)] for i in range(3)]

    # tau
    metric_matrix[0][0] = metric.coeff(dtau ** 2)
    metric_matrix[0][1] = metric.coeff(dtau * dphi_tilde) / 2
    metric_matrix[0][2] = metric.coeff(dtau * dr_tilde) / 2

    # phi_tilde
    metric_matrix[1][0] = metric.coeff(dphi_tilde * dtau) / 2
    metric_matrix[1][1] = metric.coeff(dphi_tilde ** 2)
    metric_matrix[1][2] = metric.coeff(dphi_tilde * dr_tilde) / 2

    # r_tilde
    metric_matrix[2][0] = metric.coeff(dr_tilde * dtau) / 2
    metric_matrix[2][0] = metric.coeff(dr_tilde * dphi_tilde) / 2
    metric_matrix[2][2] = metric.coeff(dr_tilde ** 2)

    g_mat = sympy.Matrix(metric_matrix)
    g_mat.simplify()

    return g_mat


def metric_coefficients(metric):
    metric_mat = metric_3d_to_matrix(metric)

    A_val, B_val = get_real_and_imag(metric_mat[0, 0])
    C_val, D_val = get_real_and_imag(metric_mat[0, 1])
    E_val, F_val = get_real_and_imag(metric_mat[1, 1])
    G_val, H_val = get_real_and_imag(metric_mat[2, 2])

    return {
        A: A_val,
        B: B_val,
        C: C_val,
        D: D_val,
        E: E_val,
        F: F_val,
        G: G_val,
        H: H_val
    }


def get_real_and_imag(expr):
    """Get real and imaginary part of a given expression that using delta_r, which is complex."""
    # substitute delta_r = a + b * i
    expr = _subs_delta_r(expr)

    # make other substitutions
    expr = _subs_delta_theta(expr)
    expr = _subs_W(expr)
    expr = _subs_xi(expr)
    expr = _subs_omega(expr)

    # check if complex number appears in the denominator
    expr_num, expr_denom = sympy.fraction(expr)

    expr_denom_real, expr_denom_imag = expr_denom.as_real_imag()
    if expr_denom_imag == 0:
        # denominator is real
        expr = expr.expand().collect(sympy.I)
        real_and_imag = expr.as_real_imag()
        return (real_and_imag[0].simplify().factor(), real_and_imag[1].simplify().factor())

    # multiply by complex conjugate
    expr = expr_num * expr_denom.conjugate()
    expr = expr.expand().collect(sympy.I)
    real_and_imag = expr.as_real_imag()

    norm = expr_denom_real ** 2 + expr_denom_imag ** 2
    real_part = real_and_imag[0].simplify().factor() / norm
    imag_part = real_and_imag[1].simplify().factor() / norm

    return real_part, imag_part


def _subs_delta_r(metric):
    real_delta_r = (
        r_tilde ** 4 - r_tilde_plus ** 4 + (r_tilde_plus - r_tilde) ** 2 * (a + 1) ** 2 +
        2 * a * (r_tilde_plus ** 2 - r_tilde ** 2)
    )

    imag_delta_r = 2 * (r_tilde_plus - r_tilde) * (a + 1) * (r_tilde_plus ** 2 - a)

    metric = metric.subs({delta_r: real_delta_r + sympy.I * imag_delta_r})

    return metric

def _subs_delta_theta(metric):
    return metric.subs({delta_theta: 1 - a ** 2 * sympy.cos(theta) ** 2})

def _subs_W(metric):
    return metric.subs({W: r_tilde ** 2 + a ** 2 * sympy.cos(theta) ** 2})

def _subs_xi(metric):
    return metric.subs({xi: 1 - a ** 2})

def _subs_omega(metric):
    return metric.subs({omega: a * (1 - a ** 2) / (r_tilde_plus ** 2 + a ** 2)})
