"""Various functions for plotting admissibility."""

import pickle

import matplotlib.pyplot as plt
import sympy
import tikzplotlib


def plot_admissibility(pgrid, admissible_map, filename=None):
    """Print the admissibility map in parameter space to the screen or a file.

    Args:
        pgrid: Object representing parameter grid
        admissible_map: Grid where each element indicates admissibility for given parameter values
        filename: Name of file to save plot to

    """
    _plot(admissible_map, pgrid.a_vals_grid, pgrid.r_tilde_plus_vals_grid, pgrid.a_vals)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()


def plot_admissibility_from_file(filename, plotname, tikz_plotname=None):
    """Plot admissibility map from file and save to a pdf."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    _plot(data['admissible_map'], data['a_vals'], data['r_tilde_plus_vals'], data['a_vals'][0, :])

    plt.savefig(plotname, bbox_inches='tight', dpi=300)

    if tikz_plotname:
        tikzplotlib.save(tikz_plotname)

    plt.close()


def _plot(admissible_map, a_vals_grid, r_tilde_plus_vals_grid, a_vals):
    """Plot the admissibility map given a grid."""
    plt.contourf(a_vals_grid, r_tilde_plus_vals_grid, admissible_map, cmap='Set2')
    plt.plot(a_vals, [sympy.sqrt(a_val) for a_val in a_vals], 'k-')
    plt.plot(a_vals, [-sympy.sqrt(a_val) for a_val in a_vals], 'k-')

    plt.xlabel("$a$")
    plt.ylabel("$\\tilde{r}_+$")
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.colorbar()
