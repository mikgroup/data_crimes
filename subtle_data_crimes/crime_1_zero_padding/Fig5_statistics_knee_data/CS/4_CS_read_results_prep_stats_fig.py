import matplotlib.pyplot as plt
import numpy as np

R = np.array([4])

N_examples = 122  # the number of slices in our test set
pad_ratio_vec = np.array([1, 1.25, 1.5, 1.75, 2])
sampling_type_vec = np.array([1, 2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens

CS_test_set_NRMSE_arr = np.empty([N_examples, pad_ratio_vec.shape[0], sampling_type_vec.shape[0]])
CS_test_set_SSIM_arr = np.empty([N_examples, pad_ratio_vec.shape[0], sampling_type_vec.shape[0]])

for samp_i in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[samp_i] == 1:
        samp_str = 'weak_VD'
    elif sampling_type_vec[samp_i] == 2:
        samp_str = 'strong_VD'

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):
        if (pad_ratio == 1.0) | (pad_ratio == 2.0):
            pad_ratio = int(pad_ratio)

        logdir = 'logs/' + samp_str + '_pad_ratio_{}'.format(pad_ratio)
        print(logdir)

        filename = logdir + "/CS_res_NRMSE_SSIM.npz"
        container = np.load(filename)

        CS_NRMSE_arr = container['CS_NRMSE_arr']
        CS_SSIM_arr = container['CS_SSIM_arr']

        CS_test_set_NRMSE_arr[:, pad_i, samp_i] = CS_NRMSE_arr
        CS_test_set_SSIM_arr[:, pad_i, samp_i] = CS_SSIM_arr

print('')

filename = 'CS_test_results.npz'
np.savez(filename, CS_test_set_NRMSE_arr=CS_test_set_NRMSE_arr, CS_test_set_SSIM_arr=CS_test_set_SSIM_arr)

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

############ Display the results graph ###############
markers = ['o', 'd', 's', 'h', '8']
colorslist = ['c', 'm', 'k']

# NRMSE figure
fig = plt.figure()
for j in range(sampling_type_vec.shape[0]):

    NRMSE_vec = CS_test_set_NRMSE_arr[:, :, j].squeeze()
    NRMSE_av_per_pad_N = NRMSE_vec.mean(axis=0)
    NRMSE_std_per_pad_N = NRMSE_vec.std(axis=0)

    # create labels
    if sampling_type_vec[j] == 1:
        label_str = 'weak VD'
    elif sampling_type_vec[j] == 2:
        label_str = 'strong VD'

    # plt.errorbar(pad_ratio_vec, NRMSE_av_per_pad_N, yerr=NRMSE_std_per_pad_N, linestyle='solid', label=label_str,
    #             marker=markers[j])
    plt.plot(pad_ratio_vec, NRMSE_av_per_pad_N, linestyle='solid', label=label_str,
             marker=markers[j])

    plt.fill_between(pad_ratio_vec, (NRMSE_av_per_pad_N - NRMSE_std_per_pad_N / 2),
                     (NRMSE_av_per_pad_N + NRMSE_std_per_pad_N / 2), alpha=0.1)

plt.ylabel('NRMSE', fontsize=20)
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.ylim(0.0035, 0.02)
ax.set_yticks((0.005, 0.01, 0.015, 0.02))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.legend(fontsize=20, loc='lower left')
plt.show()
figname_NRMSE = 'CS_NRMSE_stats'
fig.savefig(fname=figname_NRMSE)
plt.show()

# SSIM figure
fig = plt.figure()
for j in range(sampling_type_vec.shape[0]):

    SSIM_vec = CS_test_set_SSIM_arr[:, :, j].squeeze()
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

figname_SSIM = 'CS_SSIM_stats'
fig.savefig(fname=figname_SSIM)
