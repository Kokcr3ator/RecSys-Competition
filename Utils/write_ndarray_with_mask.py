import numpy as np

def write_ndarray_with_mask(array, mask_row, mask_col, new_values_array):
    selected_rows = array[mask_row]
    selected_rows[:, mask_col] = new_values_array
    array[mask_row] = selected_rows