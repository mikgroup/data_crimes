

import os
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

# add path to functions library - when running on mikQNAP
sys.path.append("/mikQNAP/efrat/1_inverse_crimes/1_mirror_PyCharm_CS_MoDL_merged/SubtleCrimesRepo/")


sys.path.append("/home/efrat/anaconda3/")
sys.path.append("/home/efrat/anaconda3/lib/python3.7/site-packages/")  # path to sigpy

# strong VD & weak VD, lamda range larger
R = 4
num_slices = 10 #
Nsamp = 2
data_filename = 'knee_lamda_calib_R{}_num_slices{}_Nsamp{}.npz'.format(R,num_slices,Nsamp)


# ---------------------- load results --------------------
im_str = 'knee im'

container = np.load(data_filename,allow_pickle=True)
print('container files:')
container.files

NRMSE_arr            = container['NRMSE_arr']
pad_ratio_vec        = container['pad_ratio_vec']
sampling_type_vec    = container['sampling_type_vec']
lamda_vec            = container['lamda_vec']
num_slices           = container['num_slices']

NRMSE_arr_av = np.mean(NRMSE_arr,axis=0)

mycolor =['k','b','r','g','c','m']


# --------------------- display mean NRMSE vs. lamda ----------------------
r = 0 # the experiment was performed for a single value of R (R=4)

for j in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[j] == 0:
        samp_label = 'unif-random'
    elif sampling_type_vec[j] == 1:
        samp_label = 'weak VD'
    elif sampling_type_vec[j] == 2:
        samp_label = 'strong VD'

    fig = plt.figure()

    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        if pad_ratio==1:
            pad_label = 'ImSize orig'
        else:
            pad_label = 'ImSizex{}'.format(pad_ratio)

        label = pad_label

        CS_vs_lamda_vec = NRMSE_arr_av[pad_i,r,j,:].squeeze()

        plt.plot(lamda_vec,CS_vs_lamda_vec, marker = 'o', markersize = 7.50, label = label, linewidth = 2, color = mycolor[pad_i])
        plt.xlabel('lamda')
        plt.ylabel('NRMSE')

    title_str = 'knee N_images={}'.format(num_slices) + ' ' + 'R={} '.format(R) + ' - CS recon error vs lamda'
    plt.ylim(0.005,0.03)
    plt.title(title_str)
    plt.suptitle(samp_label)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.legend(fontsize=18,loc='upper center')
    plt.show()


#################### find the optimal value of lamda ######################
optimal_lamda_arr = np.zeros((pad_ratio_vec.shape[0],sampling_type_vec.shape[0]))

print('------- Chosen lamda values ------')
for j in range(sampling_type_vec.shape[0]):
    samp_type = sampling_type_vec[j]
    if samp_type == 0:
        samp_label = 'unif-random'
    elif samp_type == 1:
        samp_label = 'weak VD'
    elif samp_type == 2:
        samp_label = 'strong VD' #

    # find the optimal lamda for
    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        CS_NRMSE_vs_lamda_vec = NRMSE_arr_av[pad_i,r, j, :].squeeze()
        lam_i_min = np.argmin(CS_NRMSE_vs_lamda_vec)
        lamda_chosen = lamda_vec[lam_i_min]

        optimal_lamda_arr[pad_i,j] = lamda_chosen

        print(f'pad={pad_ratio} {samp_label} lamda={lamda_chosen :.6f}')

np.savez('CS_optimal_lamda',pad_ratio_vec=pad_ratio_vec,sampling_type_vec=sampling_type_vec,optimal_lamda_arr=optimal_lamda_arr,R=R)
