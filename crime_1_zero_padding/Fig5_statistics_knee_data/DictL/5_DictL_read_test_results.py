'''
This script reads the results of the previous script, 4_DictL_generate_test_commands.py, and prepare a graph that shows the
statistics of the DictL test runs (for Fig5 in the paper).

(c) Efrat Shimron, UC Berkeley, 2021
'''

import numpy as np
import matplotlib.pyplot as plt

R = np.array([4])

N_examples = 122 # the number of slices in our test set
#data_type_str = 'test'

pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])
sampling_type_vec = np.array([1,2])  # 0 = random, 1 = strong var-dens, 2 = weak var-dens

# initialize arrays
DictL_NRMSE_test_set = np.zeros((N_examples,pad_ratio_vec.shape[0],sampling_type_vec.shape[0]))
DictL_SSIM_test_set  = np.zeros((N_examples,pad_ratio_vec.shape[0],sampling_type_vec.shape[0]))


################ load data & plot ##################

for samp_i in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[samp_i] == 1:
        samp_str = 'weak'
    elif sampling_type_vec[samp_i] == 2:
        samp_str = 'strong'

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):
        if (pad_ratio==1.0) | (pad_ratio==2.0):
            pad_ratio = int(pad_ratio)

        logdir = 'test_logs/' + samp_str + '_pad_ratio_{}'.format(pad_ratio)

        filename = logdir + "/res_NRMSE_SSIM.npz"
        container = np.load(filename)

        DictL_NRMSE_arr = container['Dict_NRMSE_arr'] # notice: the name of the stored array begins with "Dict", not "DictL" (for convenience only)
        DictL_SSIM_arr = container['Dict_SSIM_arr'] # notice: the name of the stored array begins with "Dict", not "DictL" (for convenience only)

        DictL_NRMSE_test_set[:,pad_i,samp_i] = DictL_NRMSE_arr
        DictL_SSIM_test_set[:, pad_i, samp_i] = DictL_SSIM_arr


print('')

filename = 'DictL_test_results.npz'
np.savez(filename,DictL_NRMSE_test_set=DictL_NRMSE_test_set,DictL_SSIM_test_set=DictL_SSIM_test_set)



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


############ Display the results graph ###############
markers=['o', 'v', 's', 'h', '8']
colorslist = ['b','g','k']

# NRMSE figure
fig = plt.figure()
for j in range(sampling_type_vec.shape[0]):

    NRMSE_vec = DictL_NRMSE_test_set[:, :, j].squeeze()
    NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=0)
    NRMSE_std_per_pad_N = NRMSE_vec.std(axis=0)

    # create labels
    if sampling_type_vec[j] == 1:
        label_str = 'weak VD'
    elif sampling_type_vec[j] == 2:
        label_str = 'strong VD'

    plt.errorbar(pad_ratio_vec, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N, linestyle='solid', label=label_str,
                 marker=markers[j])


plt.ylabel('NRMSE', fontsize=20)
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.ylim(0.0035, 0.027)
ax.set_yticks((0.005, 0.01, 0.015, 0.02, 0.025))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.legend(fontsize=20)
plt.show()
figname_NRMSE = 'DictL_NRMSE_stats'
fig.savefig(fname=figname_NRMSE)
plt.show()


# SSIM figure
fig = plt.figure()
for j in range(sampling_type_vec.shape[0]):

    SSIM_vec = DictL_SSIM_test_set[:, :, j].squeeze()
    SSIM_av_per_pad_N = SSIM_vec.mean(axis=0)
    SSIM_std_per_pad_N = SSIM_vec.std(axis=0)

    # create labels
    if sampling_type_vec[j] == 1:
        label_str = 'weak VD'
    elif sampling_type_vec[j] == 2:
        label_str = 'strong VD'

    plt.errorbar(pad_ratio_vec, SSIM_av_per_pad_N, yerr=SSIM_std_per_pad_N, linestyle='solid', label=label_str,
                 marker=markers[j])

plt.xlabel('Zero padding', fontsize=18)
plt.ylabel('SSIM', fontsize=20)
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.locator_params(axis='y', nbins=4)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylim(0.72, 0.98)
ax.set_yticks((0.75, 0.8, 0.85, 0.9, 0.95))
ax.legend(fontsize=20, loc='lower right')
plt.show()

figname_SSIM = 'DictL_SSIM_stats'
fig.savefig(fname=figname_SSIM)