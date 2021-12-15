'''
This script loads the results and prepares Fig8b.

(c) Efrat Shimron, UC Berkeley, 2021
'''

import os

import matplotlib.pyplot as plt
import numpy as np

pad_ratio_vec = np.array([1, 1.25, 1.5, 1.75, 2])
sampling_type_vec = np.array([1, 2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens


####################################### load DL results ##########################################
DL_data_filename = '../Fig5_statistics_knee_data/DL/DNN_results_NRMSE_and_SSIM_N_examples_122.npz'
DL_container = np.load(DL_data_filename, allow_pickle=True)
DL_test_set_NRMSE_arr = DL_container['NRMSE_test_set']
DL_test_set_SSIM_arr = DL_container['SSIM_test_set']

DL_crime_impact_data_filename = './crime_impact_DNN_results_NRMSE_and_SSIM_N_examples_122.npz'
DL_crime_impact_container = np.load(DL_crime_impact_data_filename, allow_pickle=True)
DL_crime_impact_test_set_NRMSE_arr = DL_crime_impact_container['NRMSE_test_set']
DL_crime_impact_test_set_SSIM_arr = DL_crime_impact_container['SSIM_test_set']


#################### prep for plots #################

# prepare x ticks labels for the NRMSE and SSIM graphs
x = pad_ratio_vec
x_ticks_labels = []
for i in range(pad_ratio_vec.shape[0]):
    pad_num = pad_ratio_vec[i]
    if pad_num % 1 == 0.0:
        pad_str = str(int(pad_num))
        x_ticks_labels.append('x{}'.format(pad_str))
    elif pad_num % 1 == 0.5:
        pad_str = str(pad_num)
        x_ticks_labels.append('x{}'.format(pad_str))
    else:
        pad_str = {}
        x_ticks_labels.append('')

markers = ['o', 'd', 's', 'h', '8']
# colorslist = ['darkcyan','royalblue','r','k']
colorslist = ['darkcyan', 'r', 'k']

figs_path = 'stats_figs'

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

#######################################################

method_str = 'DL'

# --------- create chart plot ------
colorslist = ['darkcyan', 'r', 'k']

fig = plt.figure()
ax = fig.add_axes([0.16, 0.12, 0.8, 0.75])
for flag in range(2):
    if flag == 0:  # train & test on the same data
        NRMSE_arr = DL_test_set_NRMSE_arr

    elif flag == 1:
        NRMSE_arr = DL_crime_impact_test_set_NRMSE_arr

    samp_j = 1  # strongVD=1
    NRMSE_vec = NRMSE_arr[:, :, samp_j].squeeze()
    NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=0)
    NRMSE_std_per_pad_N = NRMSE_vec.std(axis=0)

    # pad_ratio_vec, NRMSE_av_per_pad_N,
    if flag == 0:
        ax.bar(pad_ratio_vec + 0.05, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N,
               color=colorslist[flag], width=0.1, label='test on processed data')
    elif flag == 1:
        ax.bar(pad_ratio_vec - 0.05, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N,
               color=colorslist[flag], width=0.1, label='test on non-processed data')
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.ylim(0, 0.018)
plt.xlim(0.82, 2.5)
ax.set_yticks((0.005, 0.01, 0.015))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.legend(fontsize=11, loc='upper right')
plt.ylabel('NRMSE', fontsize=14)
plt.xlabel('zero-padding of training data', fontsize=14)
plt.show()
figname = 'figure_8b_crime_impact'
fig.savefig(figname)

