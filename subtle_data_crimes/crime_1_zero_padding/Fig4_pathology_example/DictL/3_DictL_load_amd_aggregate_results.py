# This script loads the results of several runs of 2_DictL_recon_pathology.py, which were perfomred for
# pad_ratio = 1, weak VD
# pad_ratio = 2, weak VD
# pad_ratio = 1, strong VD
# pad_ratio = 2, strong VD
#
# (c) Efrat Shimron, UC Berkeley, 2021


import os

import numpy as np

R = 4  # np.array([4])

N_examples = 1  # the number of slices in our test set
data_type_str = 'pathology_1'

pad_ratio_vec = np.array([1, 2])
sampling_type_vec = np.array([1, 2])  # 0 = random, 1 = strong var-dens, 2 = weak var-dens

num_cpus = str(10)  # number of CPUs that each run can employ

pathology_slice = np.array([22])

data_type = 'pathology_1'
# logdir = data_type + '_results'
# if not os.path.exists(logdir):
#    os.makedirs(logdir)

gold_dict = {}  # a python dictionary that will contain the gold standard recons
DictL_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing

for samp_i in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[samp_i] == 1:
        samp_type = 'weak'
    elif sampling_type_vec[samp_i] == 2:
        samp_type = 'strong'

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):
        if (pad_ratio == 1.0) | (pad_ratio == 2.0):
            pad_ratio_str = int(pad_ratio)

        logdir = data_type_str + f'_results_R{int(R)}/' + samp_type + f'_pad_ratio_{pad_ratio_str}'
        print('logdir = ', logdir)

        for s_i in range(pathology_slice.shape[0]):
            s = pathology_slice[s_i]
            # load "gold standard", which was computed from zero-padded data
            gold_filename = logdir + f'/rec_gold_pad_{pad_ratio}_slice{s}.npz'
            gold_container = np.load(gold_filename)
            rec_gold = gold_container['rec_gold']

            # load reconstruction
            rec_filename = logdir + f'/rec_DictL_pad_{pad_ratio}_{samp_type}_VD_slice{s}.npz'
            rec_container = np.load(rec_filename)
            rec_DictL = rec_container['rec_DictLearn']

            rec_gold_rotated = np.rot90(np.abs(rec_gold), 2)
            rec_DictL_rotated = np.rot90(np.abs(rec_DictL), 2)

            gold_dict[pad_ratio, samp_type] = rec_gold_rotated
            DictL_recs_dict[pad_ratio, samp_type] = rec_DictL_rotated

            # # zoom-in coordinates for pathology 1
            # x1 = 335
            # x2 = 380
            # y1 = 210
            # y2 = 300
            # # scale the zoom-in coordinates to fit changing image size
            # x1s = 335 * pad_ratio
            # x2s = 380 * pad_ratio
            # y1s = 210 * pad_ratio
            # y2s = 300 * pad_ratio
            #
            # cmax = np.max(np.abs(gold_rotated))
            #
            # if (pad_i==0) & (samp_i==0):
            #     fig = plt.figure()
            #     plt.imshow(gold_rotated, cmap="gray")
            #     plt.axis('off')
            #     plt.clim(0, cmax)
            #     plt.title(f'gold standard - pad {pad_ratio_str}')
            #     plt.show()
            #     figname = logdir + f'/target_slice{s}_pad_{pad_ratio_str}.eps'
            #     fig.savefig(figname, format='eps', dpi=1000)
            #
            # fig = plt.figure()
            # plt.imshow(gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
            # plt.axis('off')
            # plt.clim(0, cmax)
            # plt.title(f'gold zoomed - pad {pad_ratio_str}')
            # plt.show()
            # figname = logdir + f'/target_slice{s}_pad_{pad_ratio_str}_zoomed.png'
            # fig.savefig(figname, dpi=1000)
            #
            # fig = plt.figure()
            # plt.imshow(rec_DictL_rotated[x1s:x2s, y1s:y2s], cmap="gray")
            # plt.axis('off')
            # plt.clim(0, cmax)
            # plt.title(f'DictL rec zoomed - pad {pad_ratio_str}  {samp_type} VD')
            # plt.show()
            # figname = logdir + f'/DL_rec_slice{s}_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed.eps'
            # fig.savefig(figname, format='eps', dpi=1000)

# save the recons
results_dir = data_type + f'_results_R{R}/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

gold_filename = results_dir + '/gold_dict.npy'
np.save(gold_filename, gold_dict)
DictL_rec_filename = results_dir + '/DictL_dict.npy'
np.save(DictL_rec_filename, DictL_recs_dict)
