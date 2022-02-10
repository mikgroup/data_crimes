'''
This code loads the results for the CS, DictL and DL algorithms and prepares the graphs shown in Fig 5.

(c) Efrat Shimron (UT Berkeley) (2021)
'''

import numpy as np
import sys
import os
sys.path.append("../")  # add folder above
import matplotlib.pyplot as plt

pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])
sampling_type_vec = np.array([1,2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens


####################################### load CS results ##########################################
CS_data_filename = './CS/CS_test_results.npz'
CS_container = np.load(CS_data_filename,allow_pickle=True)
CS_test_set_NRMSE_arr = CS_container['CS_test_set_NRMSE_arr']
CS_test_set_SSIM_arr  = CS_container['CS_test_set_SSIM_arr']


####################################### load DictL results ##########################################
DictL_data_filename = 'DictL/DictL_test_results.npz'
DictL_container = np.load(DictL_data_filename,allow_pickle=True)
DictL_test_set_NRMSE_arr = DictL_container['DictL_NRMSE_test_set']
DictL_test_set_SSIM_arr  = DictL_container['DictL_SSIM_test_set']


####################################### load DL results ##########################################
DL_data_filename = './DL/DNN_results_NRMSE_and_SSIM_N_examples_122.npz'
DL_container = np.load(DL_data_filename,allow_pickle=True)
DL_test_set_NRMSE_arr = DL_container['NRMSE_test_set']
DL_test_set_SSIM_arr  = DL_container['SSIM_test_set']

#################### prep for plots #################

# prepare x ticks labels for the NRMSE and SSIM graphs
x = pad_ratio_vec
x_ticks_labels = []
for i in range(pad_ratio_vec.shape[0]):
    pad_num = pad_ratio_vec[i]
    if pad_num % 1 ==0.0:
        pad_str = str(int(pad_num))
        x_ticks_labels.append('x{}'.format(pad_str))
    elif pad_num % 1 == 0.5:
        pad_str = str(pad_num)
        x_ticks_labels.append('x{}'.format(pad_str))
    else:
        pad_str = {}
        x_ticks_labels.append('')

markers=['o', 'd', 's', 'h', '8']
colorslist = ['darkcyan','royalblue','k']

figs_path = 'stats_figs'

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

#######################################################

N_methods = 3 # 0 = CS, 1 = DictL, 2 = DNN

for method_i in range(N_methods):
    if method_i==0:
        method_str = 'CS'
        NRMSE_arr = CS_test_set_NRMSE_arr
        SSIM_arr = CS_test_set_SSIM_arr
    elif method_i==1:
        method_str = 'DictL'
        NRMSE_arr = DictL_test_set_NRMSE_arr
        SSIM_arr = DictL_test_set_SSIM_arr
    elif method_i == 2:
        method_str = 'DL'
        NRMSE_arr = DL_test_set_NRMSE_arr
        SSIM_arr = DL_test_set_SSIM_arr

    print(f'--------- {method_str} --------')


    ############ NRMSE figure ###############


    # NRMSE figure
    fig = plt.figure()
    for j in range(sampling_type_vec.shape[0]):

        NRMSE_vec = NRMSE_arr[:, :, j].squeeze()
        NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=0)
        NRMSE_std_per_pad_N = NRMSE_vec.std(axis=0)

        NRMSE_diff = NRMSE_av_per_pad_N[-1] - NRMSE_av_per_pad_N[0]
        NRMSE_drop = NRMSE_diff / NRMSE_av_per_pad_N[0] * 100

        NRMSE_diff_vec = NRMSE_av_per_pad_N - NRMSE_av_per_pad_N[0]
        NRMSE_change_vec = NRMSE_diff_vec/ NRMSE_av_per_pad_N[0] * 100

        # create labels & plot lines showing the NRMSE decline
        if sampling_type_vec[j] == 1:
            label_str = 'weak VD'
            # plot horizontal and vertical straight lines:
            plt.plot(np.array([1, 2]), np.array([NRMSE_av_per_pad_N[0], NRMSE_av_per_pad_N[0]]), color='dimgray',
                     linestyle='--')
            plt.plot(np.array([2, 2]), np.array([NRMSE_av_per_pad_N[0], NRMSE_av_per_pad_N[-1]]), color='dimgray',
                     linestyle='solid')

        elif sampling_type_vec[j] == 2:
            label_str = 'strong VD'

        plt.plot(pad_ratio_vec, NRMSE_av_per_pad_N, linestyle='solid', label=label_str,color=colorslist[j],
                     marker=markers[j])

        plt.fill_between(pad_ratio_vec, (NRMSE_av_per_pad_N - NRMSE_std_per_pad_N/2), (NRMSE_av_per_pad_N + NRMSE_std_per_pad_N/2), color=colorslist[j],alpha=0.1)

        # print results
        print(f'{method_str} {label_str}; NRMSE init {NRMSE_av_per_pad_N[0]:.4f}; NRMSE end {NRMSE_av_per_pad_N[-1]:.4f}; NRMSE drop {NRMSE_drop:.1f}%')


    plt.ylabel('NRMSE', fontsize=20)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks_labels, fontsize=18)
    plt.xlim(0.95,2.05)
    plt.ylim(0.006, 0.02)
    ax.set_yticks((0.01, 0.015, 0.02, 0.02))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if method_i==2:
        ax.legend(fontsize=20,loc='upper right')

    plt.tight_layout()
    plt.show()
    figname_NRMSE = figs_path + f'/{method_str}_NRMSE_stats'
    fig.savefig(fname=figname_NRMSE, dpi=1200)


    ############################################################################
    # SSIM figure
    fig = plt.figure()
    for j in range(sampling_type_vec.shape[0]):

        SSIM_vec = SSIM_arr[:, :, j].squeeze()
        SSIM_av_per_pad_N = SSIM_vec.mean(axis=0)
        SSIM_std_per_pad_N = SSIM_vec.std(axis=0)

        SSIM_diff = SSIM_av_per_pad_N[-1] - SSIM_av_per_pad_N[0]
        SSIM_drop = SSIM_diff / SSIM_av_per_pad_N[0] * 100

        SSIM_diff_vec = SSIM_av_per_pad_N - SSIM_av_per_pad_N[0]
        SSIM_change_vec = SSIM_diff_vec/ SSIM_av_per_pad_N[0] * 100

        # create labels
        if sampling_type_vec[j] == 1:
            label_str = 'weak VD'
        elif sampling_type_vec[j] == 2:
            label_str = 'strong VD'

        plt.plot(pad_ratio_vec, SSIM_av_per_pad_N, linestyle='solid', label=label_str,color=colorslist[j],
                     marker=markers[j])

        plt.fill_between(pad_ratio_vec, (SSIM_av_per_pad_N - SSIM_std_per_pad_N/2), (SSIM_av_per_pad_N + SSIM_std_per_pad_N/2), color=colorslist[j],alpha=0.1)

        # print results
        #print(f'SSIM av {label_str}: {SSIM_av_per_pad_N}')
        print(f'{method_str} {label_str}; SSIM init {SSIM_av_per_pad_N[0]:.2f} ; SSIM end  {SSIM_av_per_pad_N[-1]:.2f} ; SSIM rise {SSIM_drop:.1f}%')


    plt.xlabel('Zero padding', fontsize=18)
    plt.ylabel('SSIM', fontsize=20)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks_labels, fontsize=18)
    plt.xlim(0.95, 2.05)
    plt.locator_params(axis='y', nbins=4)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylim(0.847, 0.97)
    ax.set_yticks((0.85, 0.9, 0.95))
    if method_i == 2:
        ax.legend(fontsize=20, loc='lower right')

    plt.tight_layout()
    plt.show()

    figname_SSIM = figs_path + f'/{method_str}_SSIM_stats'
    fig.savefig(fname=figname_SSIM, dpi=1200)

