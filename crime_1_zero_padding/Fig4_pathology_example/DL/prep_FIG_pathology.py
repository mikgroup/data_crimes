
import os
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch
import torch.nn as nn
from PIL import Image

# load results that were generated by OLD_Test_MoDL_pathology_example.py

R_vec = np.array([4])
pad_ratio_vec = np.array([1, 2])
var_dens_type_vec = np.array([0, 1])  # 0 = weak var dens, 1 = strong var dens

for ex_i in range(3):  # run over examples

    fig = plt.figure()
    cnt = 0

    for v_i in range (var_dens_type_vec.shape[0]):

        if var_dens_type_vec[v_i]==0:
            var_dens_flag = 'weak'
        elif var_dens_type_vec[v_i]==1:
            var_dens_flag = 'strong'

        for pad_i in range(pad_ratio_vec.shape[0]):
            pad_ratio = pad_ratio_vec[pad_i]
            cnt += 1

            rec_zoomed = np.load('test_figs_pathology/example_{}_pad_{}_var_dens_{}_rec_zoomed.npy'.format(ex_i, pad_ratio, var_dens_flag))
            #NRMSE_full_FOV = np.load('test_figs_pathology/example_{}_pad_{}_var_dens_{}_NRMSE.npy'.format(iter, MoDL_err.NRMSE, pad_ratio, var_dens_flag),
            #        MoDL_err.NRMSE)

            plt.subplot(2,2,cnt)
            plt.imshow(rec_zoomed,cmap="gray")
            plt.title('{} var-dens; pad {}'.format(var_dens_flag,pad_ratio))
            plt.axis('off')


    plt.suptitle('example {}'.format(ex_i))
    plt.show()
    fig.savefig('test_figs_pathology/ALL_zooms_example_{}'.format(ex_i))