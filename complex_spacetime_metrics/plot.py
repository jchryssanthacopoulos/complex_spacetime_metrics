"""Various functions for plotting admissibility."""

import matplotlib.pyplot as plt
import sympy


def plot_admissibility(pgrid, admissible_map, filename=None):
    """Print the admissibility map in parameter space to the screen or a file.

    Args:
        pgrid: Object representing parameter grid
        admissible_map: Grid where each element indicates admissibility for given parameter values
        filename: Name of file to save plot to

    """
    plt.contourf(pgrid.a_vals_grid, pgrid.r_tilde_plus_vals_grid, admissible_map, cmap='Set2')
    plt.plot(pgrid.a_vals, [sympy.sqrt(a_val) for a_val in pgrid.a_vals], 'k-')
    plt.plot(pgrid.a_vals, [-sympy.sqrt(a_val) for a_val in pgrid.a_vals], 'k-')

    plt.xlabel("$a$")
    plt.ylabel("$\\tilde{r}_+$")
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.colorbar()

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
