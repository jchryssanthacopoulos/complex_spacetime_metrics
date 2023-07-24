"""This script loads the angular data in complex eigenvalue admissibility condition and plots it."""
import pickle

import matplotlib.pyplot as plt
import sympy
import tikzplotlib


filename = 'data/complex_eigenvalues/angles.theta_pi_4.pk'
with open(filename, 'rb') as f:
    data = pickle.load(f)


a_vals_grid = data['a_vals']
r_tilde_plus_vals_grid = data['r_tilde_plus_vals']
a_vals = a_vals_grid[0, :]


plt.contourf(a_vals_grid, r_tilde_plus_vals_grid, data['angle_1'][0])
plt.plot(a_vals, [sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.plot(a_vals, [-sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.xlabel("$a$")
plt.ylabel("$\\tilde{r}_+$")
plt.xlim([0, 1])
plt.ylim([-1, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.colorbar()
tikzplotlib.save("figures/complex_eigenvalues/angle_1.theta_pi_4.tex")


plt.contourf(a_vals_grid, r_tilde_plus_vals_grid, data['angle_2'][0])
plt.plot(a_vals, [sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.plot(a_vals, [-sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.xlabel("$a$")
plt.ylabel("$\\tilde{r}_+$")
plt.xlim([0, 1])
plt.ylim([-1, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.colorbar()
tikzplotlib.save("figures/complex_eigenvalues/angle_2.theta_pi_4.tex")


plt.contourf(a_vals_grid, r_tilde_plus_vals_grid, data['angle_3'][0])
plt.plot(a_vals, [sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.plot(a_vals, [-sympy.sqrt(a_val) for a_val in a_vals], 'k-')
plt.xlabel("$a$")
plt.ylabel("$\\tilde{r}_+$")
plt.xlim([0, 1])
plt.ylim([-1, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.colorbar()
tikzplotlib.save("figures/complex_eigenvalues/angle_3.theta_pi_4.tex")
