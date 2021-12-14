'''
This code performs the experiments for figure 8a.
It loads DL models that were trained earlier (for Fig4) using processed versions of the FastMRI fat-saturated knee data.
Each loaded network is tested twice: with processed and un-processed versions of the underlying test set.

Before running this code, update the basic_data_folder to your processed fat-sat data folder. It should be the same one
as the one defined in the script Fig4_pathology_example/data_prep/, in this code line:
FatSat_processed_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_FatSat_data/"

(c) Efrat Shimron, UC Berkeley (2021)
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.MoDL_single import UnrolledModel
from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.utils import complex_utils as cplx
from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.utils.datasets import create_data_loaders
from subtle_data_crimes.functions.error_funcs import error_metrics

################################# crime effect experiment#################
# here we examine the case of training using zero-padded data and inference (test) with NON-PADDED data
crime_impact_exp_flag = 1

print('============================================================================== ')
print('                        crime impact estimation                   ')
print('=============================================================================== ')

###########################################################################
N_examples_stats = 1  # number of examples that will be used for computing the mean and STD
print('N_examples_stats=', N_examples_stats)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Hyper parameters
params = Namespace()
params.batch_size = 1
print('2D VAR DENS')
params.sampling_flag = 'var_dens_2D'
params.NX_full_FOV = 640
params.NY_full_FOV = 372

#  experiment settings
R = 3
params.R = R

# define desired sampling type
samp_type = 'weak'
params.var_dens_flag = samp_type

unrolls = 6
checkpoint_num = 69

pad_ratio_vec = np.array([1, 2])

data_type = 'pathology_2'

if data_type == 'pathology_1':  # h5_filename = 'file1000425.h5'
    pathology_slice = 22
elif data_type == 'pathology_2':  # h5_filename = 'file1002455.h5'
    pathology_slice = 26

# create a folder for the test figures
figs_foldername = data_type + '_figs_R{}'.format(R)
if not os.path.exists(figs_foldername):
    os.makedirs(figs_foldername)

im_type_str = 'full_im'  # inference is done on full images (training was done on blocks)

gold_dict = {}  # a python dictionary that will contain the gold standard recons
DL_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing

######################################################################################################


for pad_i in range(pad_ratio_vec.shape[0]):
    pad_ratio = pad_ratio_vec[pad_i]
    pad_ratio_str = int(pad_ratio)
    print('======== pad_ratio={} ======== '.format(pad_ratio))

    params.pad_ratio = pad_ratio  # zero-padding ratio

    if crime_impact_exp_flag == 0:
        params.pad_ratio_test = params.pad_ratio
    elif crime_impact_exp_flag == 1:
        params.pad_ratio_test = 1

    basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_FatSat_data/"

    # path to test data - notice that this depends on pad_ratio_test, not on pad_ratio
    params.data_path = basic_data_folder + data_type + "/pad_" + str(
        int(100 * params.pad_ratio_test)) + "/" + im_type_str + "/"
    print('test data path:', params.data_path)

    # define calib area - this depends on the pad ratio. Calib is assumed to be 12 for NX=640
    calib_x = int(12 * (params.NX_full_FOV / 640) * params.pad_ratio_test)
    calib_y = int(12 * (params.NY_full_FOV / 640) * params.pad_ratio_test)
    params.calib = np.array([calib_x, calib_y])

    # Remember - the data loader defines the sampling mask.
    test_loader = create_data_loaders(params, shuffle_flag=False)

    # load trained network
    print(f'loading a network trained for pad ratio {pad_ratio_str}, testing on pad ratio {params.pad_ratio_test}')

    checkpoint_file = '../Fig4_pathology_example/DL/R{}_pad_{}_unrolls_{}_{}_var_dens/checkpoints/model_{}.pt'.format(
        params.R,
        str(int(100 * pad_ratio)),
        unrolls,
        samp_type,
        checkpoint_num,
    )
    checkpoint = torch.load(checkpoint_file, map_location=device)

    params_loaded = checkpoint["params"]
    single_MoDL = UnrolledModel(params_loaded).to(device)
    single_MoDL.load_state_dict(checkpoint['model'])

    single_MoDL.eval()

    NRMSE_test_list = []
    SSIM_test_list = []

    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):

            if i_batch == pathology_slice:  # process only the first image (for the paper figure)

                input_batch, target_batch, mask_batch = data

                # move data to GPU
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                mask_batch = mask_batch.to(device)

                # forward pass - for the full batch
                out_batch = single_MoDL(input_batch.float(), mask=mask_batch)

                for i in range(params.batch_size):
                    im_input = cplx.to_numpy(input_batch.cpu())[i, :, :]
                    im_target = cplx.to_numpy(target_batch.cpu())[i, :, :]
                    im_out = cplx.to_numpy(out_batch.cpu())[i, :, :]

                    MoDL_err = error_metrics(np.abs(im_target), np.abs(im_out))
                    MoDL_err.calc_NRMSE()
                    MoDL_err.calc_SSIM()
                    # print('NRMSE={:0.3f}'.format(MoDL_err.NRMSE))

                    NRMSE_test_list.append(MoDL_err.NRMSE)
                    SSIM_test_list.append(MoDL_err.SSIM)

                    # --------- prep for display --------------
                    im_target_rotated = np.rot90(np.abs(im_target), 2)
                    rec_DL_rotated = np.rot90(np.abs(im_out), 2)

                    gold_dict[pad_ratio, samp_type] = im_target_rotated
                    DL_recs_dict[pad_ratio, samp_type] = rec_DL_rotated

                    # ----------- figures -------------

                    fig = plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(rec_DL_rotated, cmap="gray")
                    plt.colorbar(shrink=0.5)
                    plt.title('im_out MoDL')
                    # plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(im_target_rotated, cmap="gray")
                    plt.colorbar(shrink=0.5)
                    # plt.axis('off')
                    plt.title('target')

                    plt.suptitle(
                        f'DL R{R} pad_x{pad_ratio_str} - example {data_type}')
                    plt.show()
                    figname = figs_foldername + '/im{}_pad_{}'.format(i_batch, pad_ratio_str)
                    fig.savefig(figname)

# save data
results_dir = data_type + f'_results_R{R}/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

gold_filename = results_dir + '/gold_dict.npy'
np.save(gold_filename, gold_dict)
DL_rec_filename = results_dir + '/DL_dict.npy'
np.save(DL_rec_filename, DL_recs_dict)
