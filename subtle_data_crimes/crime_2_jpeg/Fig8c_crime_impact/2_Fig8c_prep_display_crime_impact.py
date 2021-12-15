'''
This scripts loads the results for Fig 8c. It compares the results of testing networks that were trained on jpeg data
 using jpeg and non-jpeg data.

(c) Efrat Shimron, UC Berkeley, 2021
'''

import os

import matplotlib.pyplot as plt
import numpy as np

################################### settings ####################################

R_vec = np.array([2, 3])
N_examples = 122
var_dens_flag = 'weak'

figs_path = 'stats_figs'

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

# ############################## Load DL results - networks trained & tested on jpeg data ###################################
print('loading DNN results...')
DL_container = np.load('../Fig7_jpeg_statistics/DL/Res_MODL_JPEG_exp.npz')

DL_NRMSE_array = DL_container['DL_NRMSE_array']
DL_SSIM_array = DL_container['DL_SSIM_array']
q_vec_from_DL_container = DL_container['q_vec']

# ############################## Load DL results - crime impact - networks trained on JPEG data but tested on non-JPEG data ###################################
print('loading DNN results...')
DL_container_crime_impact = np.load('Res_MODL_JPEG_exp_crime_impact.npz')

DL_NRMSE_array_crime_impact = DL_container_crime_impact['DL_NRMSE_array']
DL_SSIM_array_crime_impact = DL_container_crime_impact['DL_SSIM_array']
q_vec_from_DL_container_crime_impact = DL_container_crime_impact['q_vec']

################ reorganize the data - keep only results corresponding to q=25,50,75,999 ####################3
# The next code deals with cases in which the CS, DictL and DL runs were performed with additional q values (e.g. q=90).
# Here we keep only the results that we wish to display in graphs later. For that, we first find the relevant indices.
# note: q=999 corresponds to the case of No Compression (NC), and it will be mapped to q=101 just for display purposes

q_vec = np.array([20, 50, 75, 999])
q_inds_CS_DictL = []
q_inds_DL = []

for q_tmp in q_vec:
    qi_DL = np.argwhere(q_vec_from_DL_container == q_tmp)
    q_inds_DL = np.append(q_inds_DL, qi_DL)

q_inds_DL = q_inds_DL.astype(int)

DL_NRMSE_array = DL_NRMSE_array[q_inds_DL, :, :]
DL_SSIM_array = DL_SSIM_array[q_inds_DL, :, :]

############################################# display ##########################################################

print('preparing figures...')

# ----------- preparation for plots -----------------
x_ticks_labels = []
for i in range(q_vec.shape[0]):

    if q_vec[(-i - 1)] == 999:
        q_label = 'NC'
    else:
        q_label = q_vec[(-i - 1)]
    x_ticks_labels.append(f'{q_label}')

print(x_ticks_labels)

markers = ['o', 'd', 's', '^', 'x']
colorslist = ['darkcyan', 'royalblue', 'darkcyan', 'k']

# q_vec_for_plot = q_vec.copy()
q_vec_for_plot = q_vec
q_vec_for_plot[q_vec == 999] = 101

print('q_vec=', q_vec)
print('q_vec_for_plot=', q_vec_for_plot)

########################## NRMSE bar chart ################################
colorslist = ['darkcyan', 'r', 'k']

fig = plt.figure()
ax = fig.add_axes([0.16, 0.12, 0.8, 0.75])
for flag in range(2):
    if flag == 0:  # train & test on the same data
        NRMSE_arr = DL_NRMSE_array

    elif flag == 1:
        NRMSE_arr = DL_NRMSE_array_crime_impact

    # samp_j = 1  # strongVD=1
    r = 1  # corresponds to R=3
    NRMSE_vec = NRMSE_arr[:, r, :].squeeze()
    NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=1)
    NRMSE_std_per_pad_N = NRMSE_vec.std(axis=1)

    # pad_ratio_vec, NRMSE_av_per_pad_N,
    if flag == 0:
        ax.bar(q_vec - 5, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N,
               color=colorslist[flag], width=10, label='test on processed data')
    elif flag == 1:
        ax.bar(q_vec + 5, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N,
               color=colorslist[flag], width=10, label='test on non-processed data')

plt.ylim(0, 0.014)
plt.xlim(-23, 117)
ax.set_yticks((0.005, 0.01))
plt.xlabel('JPEG quality', fontsize=20)
plt.ylabel('NRMSE', fontsize=20)
ax = plt.gca()
ax.set_xticks(q_vec_for_plot[::-1])
ax.set_xticklabels(x_ticks_labels, fontsize=14)
ax.invert_xaxis()
ax.legend(fontsize=11, loc='upper right')
plt.ylabel('NRMSE', fontsize=14)
plt.xlabel('JPEG', fontsize=14)
plt.show()
figname = 'figure_8c_jpeg_crime_impact'
fig.savefig(figname)
