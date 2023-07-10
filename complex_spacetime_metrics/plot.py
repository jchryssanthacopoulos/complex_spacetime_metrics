"""Various functions for plotting admissibility."""

import matplotlib.pyplot as plt
import sympy


def plot_admissibility(a_vals_grid, r_tilde_plus_vals_grid, admissible_map, filename=None):
    a_vals = a_vals_grid[0, :]

    plt.contourf(a_vals_grid, r_tilde_plus_vals_grid, admissible_map, cmap='Set2')
    plt.plot(a_vals, [sympy.sqrt(a_val) for a_val in a_vals], 'k-')
    plt.plot(a_vals, [-sympy.sqrt(a_val) for a_val in a_vals], 'k-')
    plt.xlabel("$a$")
    plt.ylabel("$\\tilde{r}_+$")
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.colorbar()

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
