##############################################################################
# To run this code, use the conda virtual environment "subtle_env"
# (two identical environments were defined on mikneto or mikshoov)
###############################################################################
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch
import torch.nn as nn
from MoDL_single import UnrolledModel
# from subsample_fastmri import MaskFunc
# from subsample_var_dens import MaskFuncVarDens_1D
# from torch.utils.data import DataLoader
from utils.datasets import create_data_loaders
from PIL import Image

# import custom libraries
from utils import complex_utils as cplx
from functions.error_funcs import error_metrics

N_examples_stats = 1  # number of examples that will be used for computing the mean and STD
print('N_examples_stats=', N_examples_stats)

R_vec = np.array([4])
R = R_vec[0]
print('R={}'.format(R))



use_multiple_GPUs_flag = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# %load_ext autoreload
# %autoreload 2


##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# TODO: "params" is defind here but it is also LOADED later in the code - check why we need it both here and there
# Hyper parameters
params = Namespace()

params.batch_size = 1
print('2D VAR DENS')
params.sampling_flag = 'var_dens_2D'
params.NX_full_FOV = 640
params.NY_full_FOV = 372

######################################################################################################
#                                     experiment settings
######################################################################################################

checkpoint_vec = np.array([69])

unrolls = 6

small_dataset_flag = 0


# pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])  # np.array([1,2])
pad_ratio_vec = np.array([1,2])

# TODO: change the next vec from sampling_type_vec to samp_type_vec and change the values to [1,2] for compatibility with the rest of the code
#sampling_type_vec = np.array([0])  # np.array([0,1])    #0 = weak var dens, 1 = strong var dens
sampling_type_vec = np.array([1,2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens

# data_type = 'test'
data_type = 'pathology_1'
#data_type = 'pathology_2'

if data_type=='pathology_1':
    pathology_slice = 22


# create a folder for the test figures
figs_foldername = data_type + '_figs_R{}'.format(R)
if not os.path.exists(figs_foldername):
    os.makedirs(figs_foldername)

im_type_str = 'full_im'  # inference is done on full images (training was done on blocks)

gold_dict = {}  # a python dictionary that will contain the gold standard recons
DL_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing

######################################################################################################

NRMSE_av_vs_pad_ratio_and_R = np.zeros((R_vec.shape[0], pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))
NRMSE_std_vs_pad_ratio_and_R = np.zeros((R_vec.shape[0], pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))
SSIM_av_vs_pad_ratio_and_R = np.zeros((R_vec.shape[0], pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))
SSIM_std_vs_pad_ratio_and_R = np.zeros((R_vec.shape[0], pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))

n_test_images = 0

for r in range(R_vec.shape[0]):
    R = R_vec[r]
    print('================================================== ')
    print('                         R={}                      '.format(R))
    print('================================================== ')

    # Important - here we update R in the params in order to create masks with appropriate sampling
    # The mask is created in the DataTransform (utils/datasets
    params.R = R

    # for qi in range(q_vec.shape[0]):
    #     q = q_vec[qi]
    #     print('========= q={} ======== '.format(q))

    # for v_i in range(sampling_type_vec.shape[0]):
    #
    #     if sampling_type_vec[v_i] == 0:
    #         var_dens_flag = 'weak'
    #     elif sampling_type_vec[v_i] == 1:
    #         var_dens_flag = 'strong'

    for j in range(sampling_type_vec.shape[0]):

        if sampling_type_vec[j] == 0:  # random uniform
            samp_type = 'random'
        elif sampling_type_vec[j] == 1:  # weak variable-density
            samp_type = 'weak'
        elif sampling_type_vec[j] == 2:  # strong variable-density
            samp_type = 'strong'

        params.var_dens_flag = samp_type

        print("loading a network trained for " + samp_type + ' var dens')

        for pad_i in range(pad_ratio_vec.shape[0]):
            pad_ratio = pad_ratio_vec[pad_i]
            print('======== pad_ratio={} ======== '.format(pad_ratio))

            if (pad_ratio == 1) | (pad_ratio == 2):
                # pad_ratio = int(pad_ratio)
                pad_ratio_str = str(int(pad_ratio))  # instead of 1.0 this will give 1
            else:
                pad_ratio_str = str(int(np.floor(pad_ratio))) + 'p' + str(int(100 * (pad_ratio % 1)))
                print(pad_ratio_str)

            params.pad_ratio = pad_ratio  # zero-padding ratio


            basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/subtle_inv_crimes_zpad_data_v19_FatSatPD"

            if small_dataset_flag == 1:
                print('*** using a SMALL dataset ***')
                basic_data_folder = basic_data_folder + '_small/'
            else:
                basic_data_folder = basic_data_folder + '/'

            # path to test data
            params.data_path = basic_data_folder + data_type + "/pad_" + str(
                int(100 * params.pad_ratio)) + "/" + im_type_str + "/"

            print('test data path:', params.data_path)

            # params.calib = int(24 * pad_ratio)
            # print('calib = ',params.calib)

            # calib is assumed to be 12 for NX=640
            calib_x = int(12 * (params.NX_full_FOV / 640) * pad_ratio)
            calib_y = int(12 * (params.NY_full_FOV / 640) * pad_ratio)
            params.calib = np.array([calib_x, calib_y])

            # Remember - the data loader defines the sampling mask. The test data must undergo the same transform as train data!
            test_loader = create_data_loaders(params,shuffle_flag=False)

            N_test_images = len(test_loader.dataset)
            print('N_test_images =', N_test_images)

            for model_i in range(checkpoint_vec.shape[0]):
                checkpoint_num = int(checkpoint_vec[model_i])
                print('checkpoint_num:', checkpoint_num)

                # load trained network
                if small_dataset_flag == 1:
                    # checkpoint_file = 'R{}_pad_ratio_{}_unrolls_{}_small_dataset/checkpoints/model_{}.pt'.format(params.R, pad_ratio, unrolls,checkpoint_num)
                    checkpoint_file = 'R{}_pad_{}_unrolls_{}_{}_var_dens_small_data/checkpoints/model_{}.pt'.format(
                        params.R, str(int(100 * pad_ratio)),
                        unrolls,
                        samp_type,
                        checkpoint_num,
                    )
                elif small_dataset_flag == 0:
                    # checkpoint_file = 'R{}_pad_ratio_{}_unrolls_{}/checkpoints/model_{}.pt'.format(params.R,pad_ratio, unrolls,checkpoint_num)
                    checkpoint_file = 'R{}_pad_{}_unrolls_{}_{}_var_dens/checkpoints/model_{}.pt'.format(
                        params.R,
                        str(int(100 * pad_ratio)),
                        unrolls,
                        samp_type,
                        checkpoint_num,
                    )
                    # checkpoint_file = 'R{}_pad_{}_unrolls_{}_small_data_{}_var_dens/checkpoints/model_{}.pt'.format(
                    #     params.R, pad_ratio, unrolls, samp_type, checkpoint_num)

                print('checkpoint_file=')
                print(checkpoint_file)

                checkpoint = torch.load(checkpoint_file, map_location=device)

                params_loaded = checkpoint["params"]
                single_MoDL = UnrolledModel(params_loaded).to(device)

                #print('params.data_path: ', params.data_path)
                #print('params.batch_size: ', params.batch_size)

                # # Data Parallelism - enables running on multiple GPUs
                # if (torch.cuda.device_count() > 1) & (use_multiple_GPUs_flag == 1):
                #     print("Now using ", torch.cuda.device_count(), "GPUs!")
                #     single_MoDL = nn.DataParallel(single_MoDL, device_ids=[0, 1, 2,
                #                                                            3])  # the first index on the device_ids determines which GPU will be used as a staging area before scattering to the other GPUs
                # else:
                #     print("Now using a single GPU")

                single_MoDL.load_state_dict(checkpoint['model'])

                single_MoDL.eval()

                NRMSE_test_list = []
                SSIM_test_list = []

                with torch.no_grad():
                    for i_batch, data in enumerate(test_loader):

                        if i_batch==pathology_slice:  # process only the first image (for the paper figure)

                            input_batch, target_batch, mask_batch = data

                            in_size = input_batch.size()
                            print('in_size=', in_size)

                            if (r == 0) & (pad_i == 0) & (j == 0):
                                n_test_images += 1
                                print('n_test_images=', n_test_images)

                            # # display the mask (before converting it to torch tensor)
                            #if (i_batch == 0) & (checkpoint_num == checkpoint_vec[-1]):

                                # display mask
                                #print('mask_batch shape:',mask_batch.shape)
                                # mask_squeezed = mask_batch[0, :, :, 0].squeeze()
                                # np.save('mask_squeezed', mask_squeezed)
                                # print('saved mask squeezed')
                                # fig = plt.figure()
                                # plt.imshow(mask_squeezed, cmap="gray")
                                # plt.title(params.sampling_flag + ' epoch 0, i_batch {}'.format(i_batch))
                                # plt.show()
                                # fig.savefig('mask_i_batch{}.png'.format(i_batch))

                            # move data to GPU
                            if (torch.cuda.device_count() > 1) & (use_multiple_GPUs_flag == 1):
                                input_batch = input_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                                target_batch = target_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                                mask_batch = mask_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                            else:
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
                                im_target_rotated = np.rot90(np.abs(im_target),2)
                                rec_DL_rotated = np.rot90(np.abs(im_out), 2)

                                gold_dict[pad_ratio, samp_type] = im_target_rotated
                                DL_recs_dict[pad_ratio, samp_type] = rec_DL_rotated

                                # fig = plt.figure()
                                # plt.imshow(im_target_rotated, cmap="gray")
                                # plt.title('target')
                                # plt.axis('on')
                                # plt.show()
                                # figname = figs_foldername + '/target{}_pad_{}'.format(i_batch, pad_ratio_str)
                                # fig.savefig(figname, format='eps', dpi=1000)

                                # # zoom-in coordinates for pathology 1
                                # x1 = 335
                                # x2 = 380
                                # y1 = 210
                                # y2 = 300
                                # # scale the zoom-in coordinates to fit changing image size
                                # x1s = 335*pad_ratio
                                # x2s = 380*pad_ratio
                                # y1s = 210*pad_ratio
                                # y2s = 300*pad_ratio
                                #
                                # cmax = np.max(np.abs(im_target_rotated))
                                #
                                # fig = plt.figure()
                                # plt.imshow(im_target_rotated, cmap="gray")
                                # plt.axis('off')
                                # plt.clim(0,cmax)
                                # plt.show()
                                # figname = figs_foldername + f'/target_im{i_batch}_pad_{pad_ratio_str}.eps'
                                # fig.savefig(figname, format='eps', dpi=1000)
                                #
                                # fig = plt.figure()
                                # plt.imshow(im_target_rotated[x1s:x2s,y1s:y2s], cmap="gray")
                                # plt.axis('off')
                                # plt.clim(0, cmax)
                                # plt.show()
                                # figname = figs_foldername + f'/target_im{i_batch}_pad_{pad_ratio_str}_zoomed.png'
                                # fig.savefig(figname, dpi=1000)
                                #
                                #
                                # fig = plt.figure()
                                # plt.imshow(rec_DL_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                                # plt.axis('off')
                                # plt.clim(0, cmax)
                                # plt.show()
                                # figname = figs_foldername +  f'/DL_rec_im{i_batch}_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed.eps'
                                # fig.savefig(figname, format='eps', dpi=1000)
                                #

                                # ---------------------------


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
                                figname = figs_foldername + '/im{}_pad_{}'.format(i_batch,pad_ratio_str)
                                fig.savefig(figname)


                                # fig = plt.figure()
                                # plt.imshow(np.rot90(np.abs(im_target), 2), cmap="gray")
                                # plt.title(f'target - im{i_batch} - pad x{pad_ratio_str}')
                                # figname = figs_foldername + '/target{}_pad_{}'.format(i_batch,pad_ratio_str)
                                # fig.savefig(figname,format='png', dpi=1000)

                                # save in .eps format
                                # fig = plt.figure()
                                # plt.imshow(np.rot90(np.abs(im_target), 2), cmap="gray")
                                # plt.title(f'target - example {i_batch} - zero-padding x{pad_ratio_str}')
                                # figname = figs_foldername + '/target{}_pad_{}'.format(i_batch,pad_ratio_str)
                                # fig.savefig(figname,format='eps', dpi=1000)

results_dir =  data_type + f'_results_R{R}/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

gold_filename = results_dir + '/gold_dict.npy'
np.save(gold_filename , gold_dict)
DL_rec_filename = results_dir + '/DL_dict.npy'
np.save(DL_rec_filename, DL_recs_dict)


# #################### prep for plots #################33
#
# # NOTICE: the following code creates and saves the figures, but for some reason the axis labels aren't
# # displayed properly.
# # for better figures run the script fig4ISMRM_Test_MoDL_run5 after running this script
#
# # prepare x ticks labels for the NRMSE and SSIM graphs
# x = pad_ratio_vec
# x_ticks_labels = []
# for i in range(pad_ratio_vec.shape[0]):
#     x_ticks_labels.append('x{}'.format(pad_ratio_vec[i]))
#
# markers = ['o', 's', 'v', 'h', '8']
# # colorslist = ['b','r']

# #########################   plots  ####################################################################
#
# # display NRMSE
# fig = plt.figure()
# for j in range(sampling_type_vec.shape[0]):
#     if sampling_type_vec[j] == 0:
#         samp_type = 'weak'
#     elif sampling_type_vec[j] == 1:
#         samp_type = 'strong'
#
#     label_str = samp_type + " var-dens"
#
#     for r in range(R_vec.shape[0]):
#         plt.errorbar(pad_ratio_vec, NRMSE_av_vs_pad_ratio_and_R[r, :, j].squeeze(),
#                      yerr=NRMSE_std_vs_pad_ratio_and_R[r, :, j].squeeze(), linestyle='solid', label=label_str,
#                      marker=markers[j])
#
# # plt.ylim((0,0.075))
# plt.xlabel('Zero padding ratio', fontsize=18)
# plt.ylabel('NRMSE', fontsize=20)
# ax = plt.gca()
# ax.set_xticks(x)
# ax.set_xticklabels(x_ticks_labels, fontsize=18)
# plt.locator_params(axis='y', nbins=3)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# ax.legend(fontsize=20)
# plt.title('N_examples_stats={}'.format(N_examples_stats))
# plt.show()
# fig.savefig('RESULTS_imSize372_NRMSE_N_examples_stats_{}'.format(N_examples_stats))
#
# # Display SSIM
# fig = plt.figure()
# for j in range(sampling_type_vec.shape[0]):
#     if sampling_type_vec[j] == 0:
#         samp_type = 'weak'
#     elif sampling_type_vec[j] == 1:
#         samp_type = 'strong'
#
#     label_str = samp_type + " var-dens"
#
#     for r in range(R_vec.shape[0]):
#         plt.errorbar(pad_ratio_vec, SSIM_av_vs_pad_ratio_and_R[r, :, j].squeeze(),
#                      yerr=SSIM_std_vs_pad_ratio_and_R[r, :, j].squeeze(), linestyle='solid', label=label_str,
#                      marker=markers[j])
#
# # plt.ylim((0,0.075))
# plt.xlabel('Zero padding ratio', fontsize=18)
# plt.ylabel('SSIM', fontsize=20)
# ax = plt.gca()
# ax.set_xticks(x)
# ax.set_xticklabels(x_ticks_labels, fontsize=18)
# plt.locator_params(axis='y', nbins=3)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# ax.legend(fontsize=20, loc="lower right")
# plt.title('N_examples_stats={}'.format(N_examples_stats))
# plt.show()
# fig.savefig('RESULTS_imSize372_SSIM_N_examples_stats_{}'.format(N_examples_stats))
#
# print('saved figures successfully')

# ################################ save data ##################################333
# # save SSIM_av & SSIM
# results_filename = 'DNN_results_NRMSE_and_SSIM_N_examples_stats_{}.npz'.format(N_examples_stats)
# np.savez(results_filename, R_vec=R_vec, pad_ratio_vec=pad_ratio_vec, params=params, checkpoint_num=checkpoint_num,
#          NRMSE_av_vs_pad_ratio_and_R=NRMSE_av_vs_pad_ratio_and_R,
#          NRMSE_std_vs_pad_ratio_and_R=NRMSE_std_vs_pad_ratio_and_R,
#          SSIM_av_vs_pad_ratio_and_R=SSIM_av_vs_pad_ratio_and_R,
#          SSIM_std_vs_pad_ratio_and_R=SSIM_std_vs_pad_ratio_and_R)
# print('saved .npz file')
# # np.savez(tmp_filename, loss_vs_i_batch_vec=loss_vs_i_batch_vec, NRMSE_vs_i_batch_vec=NRMSE_vs_i_batch_vec)
