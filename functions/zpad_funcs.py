"""
This module includes functions for experiments with the zero-padding preprocessing pipeline (subtle inverse crime I).

Efrat Shimron (UC Berkeley, 2021).
"""

import numpy as np
from functions.utils import merge_multicoil_data, calc_pad_half
import matplotlib.pyplot as plt

################################## helper func #########################################################
def zpad_merge_scale(ksp_block_multicoil, pad_ratio):
    ''' inputs:
        kspace - numpy array of size [Ncoils, NX, NY]
        pad_ratio - numpy array (scalar) that denotes the desired padding ratio
        '''

    NX = ksp_block_multicoil.shape[1]
    NY = ksp_block_multicoil.shape[2]


    ############## zero-pad, merge & save ###################

    pad_half_dim1, N_tot_dim1 = calc_pad_half(NX, pad_ratio)
    pad_half_dim2, N_tot_dim2 = calc_pad_half(NY, pad_ratio)

    padding_lengths = ((0, 0), (pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))

    #NX_padded = int(NX * pad_ratio)
    #NY_padded = int(NY * pad_ratio)

    ksp_block_multicoil_padded = np.pad(ksp_block_multicoil, padding_lengths, mode='constant',
                                        constant_values=(0, 0))

    # compute a single *magnitude* image from the data
    im_mag = merge_multicoil_data(ksp_block_multicoil_padded)

    # normalization
    magnitude_vals = im_mag.reshape(-1)
    mag_vals_sorted = np.sort(magnitude_vals)
    k = int(round(0.98 * magnitude_vals.shape[0]))
    scale_factor = mag_vals_sorted[k]
    im_mag_scaled = im_mag / scale_factor

    return im_mag_scaled