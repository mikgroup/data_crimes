# this code loads the results and prepares figures
# note

import os

import matplotlib.pyplot as plt
import numpy as np

pad_ratio_vec = np.array([1, 1.25, 1.5, 1.75, 2])
sampling_type_vec = np.array([1, 2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens

#
# ####################################### load CS results ##########################################
# CS_data_filename = './CS/CS_test_results.npz'
# CS_container = np.load(CS_data_filename,allow_pickle=True)
# CS_test_set_NRMSE_arr = CS_container['CS_test_set_NRMSE_arr']
# CS_test_set_SSIM_arr  = CS_container['CS_test_set_SSIM_arr']
#
# ####################################### load DictL results ##########################################
# DictL_data_filename = './DictL/DictL_test_results.npz'
# DictL_container = np.load(DictL_data_filename,allow_pickle=True)
# DictL_test_set_NRMSE_arr = DictL_container['DictL_NRMSE_test_set']
# DictL_test_set_SSIM_arr  = DictL_container['DictL_SSIM_test_set']


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

# markers=['o', 'v', 's', 'h', '8']
# colorslist = ['b','g','k']


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

# N_methods = 3 # 0 = CS, 1 = DictL, 2 = DNN

# #for method_i in range(N_methods):
# for method_i in np.array([2]):
#     if method_i==0:
#         method_str = 'CS'
#         NRMSE_arr = CS_test_set_NRMSE_arr
#         SSIM_arr = CS_test_set_SSIM_arr
#     elif method_i==1:
#         method_str = 'DictL'
#         NRMSE_arr = DictL_test_set_NRMSE_arr
#         SSIM_arr = DictL_test_set_SSIM_arr
#     elif method_i == 2:
#         method_str = 'DL'
#         NRMSE_arr = DL_test_set_NRMSE_arr
# #        SSIM_arr = DL_test_set_SSIM_arr
#
#     print(f'--------- {method_str} --------')

method_str = 'DL'

# ############ NRMSE figure ###############
#
#
# # NRMSE figure
# fig = plt.figure()
# color_j = 0
#
# for flag in range(2):
#     if flag==0:  # train & test on the same data
#         NRMSE_arr = DL_test_set_NRMSE_arr
#         linestyle_str ='solid'
#         #SSIM_arr = DL_test_set_SSIM_arr
#
#     elif flag==1:
#         NRMSE_arr = DL_crime_impact_test_set_NRMSE_arr
#         linestyle_str = '--'
#         #SSIM_arr = DL_test_set_SSIM_arr
#
#     print(f'flag: {flag}')
#
#     #for j in range(sampling_type_vec.shape[0]):
#     for j in np.array([1]): #strongVD
#
#         NRMSE_vec = NRMSE_arr[:, :, j].squeeze()
#         NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=0)
#         NRMSE_std_per_pad_N = NRMSE_vec.std(axis=0)
#
#         NRMSE_diff = NRMSE_av_per_pad_N[-1] - NRMSE_av_per_pad_N[0]
#         NRMSE_drop = NRMSE_diff / NRMSE_av_per_pad_N[0] * 100
#
#         NRMSE_diff_vec = NRMSE_av_per_pad_N - NRMSE_av_per_pad_N[0]
#         NRMSE_change_vec = NRMSE_diff_vec/ NRMSE_av_per_pad_N[0] * 100
#
#         # create labels
#         if sampling_type_vec[j] == 1:
#             label_str = 'weak VD'
#
#         elif sampling_type_vec[j] == 2:
#             label_str = 'strong VD'
#
#         if flag == 0:
#             curve_str = 'test on preprocessed data'
#         elif flag==1:
#             curve_str = 'test on non-processed data'
#
#         plt.plot(pad_ratio_vec, NRMSE_av_per_pad_N, linestyle=linestyle_str, label=curve_str,color=colorslist[color_j],
#                      marker=markers[j])
#
#         plt.fill_between(pad_ratio_vec, (NRMSE_av_per_pad_N - NRMSE_std_per_pad_N/2), (NRMSE_av_per_pad_N + NRMSE_std_per_pad_N/2), color=colorslist[color_j],alpha=0.1)
#
#         # print results
#         #print(f'NRMSE av {label_str}: {NRMSE_av_per_pad_N}')
#         #print(f'{method_str} {label_str}; NRMSE init {NRMSE_av_per_pad_N[0]:.4f}; NRMSE end {NRMSE_av_per_pad_N[-1]:.4f}; NRMSE drop {NRMSE_drop:.1f}%')
#
#         color_j+=1
#
# plt.ylabel('NRMSE', fontsize=14)
# plt.xlabel('zero-padding of the training dataset',fontsize=14)
# plt.title(f'{method_str} - networks trained on preprocessed data')
# ax = plt.gca()
# ax.set_xticks(x)
# ax.set_xticklabels(x_ticks_labels, fontsize=18)
# plt.xlim(0.95,2.05)
# plt.ylim(0.0045, 0.017)
# ax.set_yticks((0.005, 0.01, 0.015))
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# ax.title('Crime impact: performance of networks trained on processed data')
# ax.legend(fontsize=11,loc='upper left')
#
# #ax.grid('on')
#
# plt.show()
# figname_NRMSE = figs_path + f'/{method_str}_NRMSE_stats'
#
# fig.savefig(fname=figname_NRMSE)
#
#
#     # ############################################################################
#     # # SSIM figure
#     # fig = plt.figure()
#     # for j in range(sampling_type_vec.shape[0]):
#     #
#     #     SSIM_vec = SSIM_arr[:, :, j].squeeze()
#     #     SSIM_av_per_pad_N = SSIM_vec.mean(axis=0)
#     #     SSIM_std_per_pad_N = SSIM_vec.std(axis=0)
#     #
#     #     SSIM_diff = SSIM_av_per_pad_N[-1] - SSIM_av_per_pad_N[0]
#     #     SSIM_drop = SSIM_diff / SSIM_av_per_pad_N[0] * 100
#     #
#     #     SSIM_diff_vec = SSIM_av_per_pad_N - SSIM_av_per_pad_N[0]
#     #     SSIM_change_vec = SSIM_diff_vec/ SSIM_av_per_pad_N[0] * 100
#     #
#     #     # create labels
#     #     if sampling_type_vec[j] == 1:
#     #         label_str = 'weak VD'
#     #     elif sampling_type_vec[j] == 2:
#     #         label_str = 'strong VD'
#     #
#     #     plt.plot(pad_ratio_vec, SSIM_av_per_pad_N, linestyle='solid', label=label_str,color=colorslist[j],
#     #                  marker=markers[j])
#     #
#     #     plt.fill_between(pad_ratio_vec, (SSIM_av_per_pad_N - SSIM_std_per_pad_N/2), (SSIM_av_per_pad_N + SSIM_std_per_pad_N/2), color=colorslist[j],alpha=0.1)
#     #
#     #     # print results
#     #     #print(f'SSIM av {label_str}: {SSIM_av_per_pad_N}')
#     #     print(f'{method_str} {label_str}; SSIM init {SSIM_av_per_pad_N[0]:.2f} ; SSIM end  {SSIM_av_per_pad_N[-1]:.2f} ; SSIM rise {SSIM_drop:.1f}%')
#     #
#     #
#     # plt.xlabel('Zero padding', fontsize=18)
#     # plt.ylabel('SSIM', fontsize=20)
#     # ax = plt.gca()
#     # ax.set_xticks(x)
#     # ax.set_xticklabels(x_ticks_labels, fontsize=18)
#     # plt.xlim(0.95, 2.05)
#     # plt.locator_params(axis='y', nbins=4)
#     # plt.yticks(fontsize=20)
#     # plt.xticks(fontsize=20)
#     # plt.ylim(0.847, 0.97)
#     # ax.set_yticks((0.85, 0.9, 0.95))
#     # if method_i == 2:
#     #     ax.legend(fontsize=20, loc='lower right')
#     # plt.show()
#     #
#     # figname_SSIM = figs_path + f'/{method_str}_SSIM_stats'
#     # fig.savefig(fname=figname_SSIM)

# --------- bart chart plot ------
colorslist = ['darkcyan', 'r', 'k']

fig = plt.figure()
ax = fig.add_axes([0.16, 0.12, 0.8, 0.75])
for flag in range(2):
    if flag == 0:  # train & test on the same data
        NRMSE_arr = DL_test_set_NRMSE_arr
        # linestyle_str ='solid'
        # SSIM_arr = DL_test_set_SSIM_arr

    elif flag == 1:
        NRMSE_arr = DL_crime_impact_test_set_NRMSE_arr
        # linestyle_str = '--'
        # SSIM_arr = DL_test_set_SSIM_arr

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
    # ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    # ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
# plt.title('Crime Impact Estimation \n Performance of networks trained on processed data',fontsize=15)
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

print('')

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# #ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# #ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# #ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
# plt.show()


print('')
