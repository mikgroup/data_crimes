'''
This script loads the results of the hyperparameter search performed by the previous script:
1_knee_calib_CS_lamda_FatSatPD.py

Make sure that the data_filename defined here is identical to the one defined in the previous script.

(c) Efrat Shimron (UC Berkeley, 2021)
'''

import matplotlib.pyplot as plt
import numpy as np

# -------- Experiment set up ---------

# strong VD & weak VD, lamda range larger
R = 4
num_slices = 10  #
Nsamp = 2
data_filename = 'knee_lamda_calib_R{}_num_slices{}_Nsamp{}_FatSat.npz'.format(R, num_slices, Nsamp)

# ---------------------- load results --------------------

im_str = 'knee im'

container = np.load(data_filename, allow_pickle=True)
print('container files:')
container.files

NRMSE_arr = container['NRMSE_arr']
pad_ratio_vec = container['pad_ratio_vec']
sampling_type_vec = container['sampling_type_vec']
lamda_vec = container['lamda_vec']
num_slices = container['num_slices']

NRMSE_arr_av = np.mean(NRMSE_arr, axis=0)

# --------------------- display NRMSE vs. lamda vec ----------------------
# display results vs. lambda
mycolor = ['k', 'b', 'r', 'g']

r = 0  # there is only a single R here

for j in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[j] == 0:
        samp_label = 'unif-random'
    elif sampling_type_vec[j] == 1:
        samp_label = 'VD weak'
    elif sampling_type_vec[j] == 2:
        samp_label = 'VD strong'

    fig = plt.figure()

    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        if pad_ratio == 1:
            pad_label = 'ImSize orig'
        else:
            pad_label = 'ImSizex{}'.format(pad_ratio)

        label = pad_label

        CS_vs_lamda_vec = NRMSE_arr_av[pad_i, r, j, :].squeeze()

        plt.plot(lamda_vec, CS_vs_lamda_vec, marker='o', markersize=7.50, label=label, linewidth=2,
                 color=mycolor[pad_i])
        plt.xlabel('lamda')
        plt.ylabel('NRMSE')

    title_str = 'FatSatPD knee N_images={}'.format(num_slices) + ' ' + 'R={} '.format(
        R) + samp_label + ' - CS recon error vs lamda'
    plt.title(title_str)
    ax = plt.gca()
    # ax.set_xscale('log')
    ax.legend(fontsize=18, loc='upper center')
    plt.show()

# --------------------- display NRMSE vs. lamda vec - lamda in LOG scale----------------------
# display results vs. lambda
mycolor = ['k', 'b', 'r', 'g']

r = 0  # there is only a single R here

for j in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[j] == 0:
        samp_label = 'unif-random'
    elif sampling_type_vec[j] == 1:
        samp_label = 'VD weak'
    elif sampling_type_vec[j] == 2:
        samp_label = 'VD strong'

    fig = plt.figure()

    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        if pad_ratio == 1:
            pad_label = 'ImSize orig'
        else:
            pad_label = 'ImSizex{}'.format(pad_ratio)

        label = pad_label

        CS_vs_lamda_vec = NRMSE_arr_av[pad_i, r, j, :].squeeze()

        plt.plot(lamda_vec, CS_vs_lamda_vec, marker='o', markersize=7.50, label=label, linewidth=2,
                 color=mycolor[pad_i])
        plt.xlabel('lamda')
        plt.ylabel('NRMSE')

    title_str = 'FatSatPD knee N_images={}'.format(num_slices) + ' ' + 'R={} '.format(
        R) + samp_label + ' - CS recon error vs lamda'
    plt.title(title_str)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.legend(fontsize=18, loc='upper center')
    plt.show()

###################################### compare results for optimal value of lambda for each case & fixed lamda (lamda=0.005) ###################################

lamda_fixed = 1e-5  # original value chosen for the experiment in the paper
lam_fixed_i = np.argwhere(lamda_vec == lamda_fixed)

# prepare x ticks labels for the NRMSE and SSIM graphs
x = pad_ratio_vec
x_ticks_labels = []
for i in range(pad_ratio_vec.shape[0]):
    x_ticks_labels.append('x{}'.format(pad_ratio_vec[i]))

mycolor = ['k', 'b', 'r', 'g']
styl_list = ['-', '--', '-.', ':']

for j in range(sampling_type_vec.shape[0]):

    if sampling_type_vec[j] == 0:
        samp_label = 'unif-random'
    elif sampling_type_vec[j] == 1:
        samp_label = 'VD weak'
    elif sampling_type_vec[j] == 2:
        samp_label = 'VD strong'

    CS_NRMSE_fixed_lamda = np.zeros((pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))
    CS_NRMSE_optimal_lamda = np.zeros((pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))
    optimal_lamda_vs_pad_ratio = np.zeros(pad_ratio_vec.shape[0])

    # find the
    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        CS_NRMSE_vs_lamda_vec = NRMSE_arr_av[pad_i, r, j, :].squeeze()
        NRMSE_min = np.min(CS_NRMSE_vs_lamda_vec)
        lam_i_min = np.argmin(CS_NRMSE_vs_lamda_vec)

        CS_NRMSE_optimal_lamda[pad_i, j] = NRMSE_min

        # for comparison - save result for the "fixed" lamda
        NRMSE_lam_fixed = NRMSE_arr_av[pad_i, r, j, lam_fixed_i].squeeze()
        CS_NRMSE_fixed_lamda[pad_i, j] = NRMSE_lam_fixed

    fig = plt.figure()

    plt.plot(pad_ratio_vec, CS_NRMSE_optimal_lamda[:, j], label='lamda calibrated for each ImSize', linestyle='solid',
             color='k', marker='o')
    plt.plot(pad_ratio_vec, CS_NRMSE_fixed_lamda[:, j], label=f'fixed lamda {lamda_fixed} (paper fig)',
             linestyle='solid', color='r', marker='s')
    plt.xlabel('imSize factor')
    plt.ylabel('NRMSE')
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks_labels, fontsize=12)
    ax.legend()
    # title_str = im_str + ' ' + 'R={} '.format(R) + samp_label + ' - CS recon - OPTIMAL lamda for each imSize'
    title_str = 'CS recon ' + im_str + ' ' + 'R={} '.format(R) + samp_label
    plt.title(title_str)
    plt.show()
