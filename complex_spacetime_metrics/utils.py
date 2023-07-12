"""Contains utilities that help with data processing."""

import pickle

from scipy.interpolate import RegularGridInterpolator


def interpolate_admissible_maps(file_1: str, file_2: str, new_filename: str):
    """Interpolate admissible map in file 1 to the grid of file 2 and save

    Args:
        file_1: File to interpolate
        file_2: File to interpolate to
        new_filename: Name of new file to save

    """
    # load files
    with open(file_1, 'rb') as f:
        data_1 = pickle.load(f)
    with open(file_2, 'rb') as f:
        data_2 = pickle.load(f)

    x = data_1["a_vals"][0, :]
    y = data_1["r_tilde_plus_vals"][:, 0]
    data = data_1["admissible_map"]

    interp = RegularGridInterpolator((y, x), data, method='quintic')

    x_new = data_2["a_vals"]
    y_new = data_2["r_tilde_plus_vals"]

    data_new = interp((y_new, x_new))

    # cast to bool
    data_new = data_new > 0.5

    file_obj = {
        'a_vals': x_new,
        'r_tilde_plus_vals': y_new,
        'theta_val': data_1["theta_val"],
        'admissible_map': data_new
    }
    with open(new_filename, 'wb') as f:
        pickle.dump(file_obj, f)
