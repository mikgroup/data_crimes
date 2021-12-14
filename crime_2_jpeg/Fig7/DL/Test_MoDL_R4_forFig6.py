'''
This code is used for testing MoDL on JPEG-compressed data, for the results shown in figures 6, 7 and 8c in the paper.

Before running this script you should update the following:
basic_data_folder - it should be the same as the output folder defined in the script /crime_2_jpeg/data_prep/jpeg_data_prep.py

(c) Efrat Shimron, UC Berkeley, 2021
'''
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch
import torch.nn as nn
from MoDL_single import UnrolledModel
#from subsample_fastmri import MaskFunc
#from subsample_var_dens import MaskFuncVarDens_1D
#from torch.utils.data import DataLoader
from utils.datasets import create_data_loaders

# import custom libraries
from utils import complex_utils as cplx
# import custom classes
from utils.datasets import SliceData

from functions.error_funcs import error_metrics

use_multiple_GPUs_flag = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %load_ext autoreload
# %autoreload 2

# create a folder for the test figures
if not os.path.exists('test_figs'):
    os.makedirs('test_figs')

##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Hyper parameters
params = Namespace()
params.batch_size = 1

# image dimensions
params.NX = 640
params.NY = 372

# calib is assumed to be 12 for NX=640
calib_x = int(12)
calib_y = int(12 * params.NY / params.NX)
params.calib = np.array([calib_x, calib_y])

params.shuffle_flag = False # should be True for training, False for testing. Notice that this is not a string, semicolons aren't necessary.

params.sampling_flag = 'var_dens_2D'
params.var_dens_flag = 'strong'  # 'weak' / 'strong'
checkpoint_num = int(69)  # for loading a trained network

q_vec = np.array([20,50,75,999])
R_vec = np.array([4])

N_examples_4display=15 # number of examples to display
N_examples_stats = 15 # number of examples over which the mean and STD will be computed

NRMSE_av_vs_q_and_R = np.zeros((R_vec.shape[0],q_vec.shape[0]))
NRMSE_std_vs_q_and_R = np.zeros((R_vec.shape[0],q_vec.shape[0]))
SSIM_av_vs_q_and_R = np.zeros((R_vec.shape[0],q_vec.shape[0]))
SSIM_std_vs_q_and_R = np.zeros((R_vec.shape[0],q_vec.shape[0]))

N_calc_err = 200

NRMSE_examples_4display = np.zeros((R_vec.shape[0],q_vec.shape[0],N_calc_err))
SSIM_examples_4display = np.zeros((R_vec.shape[0],q_vec.shape[0],N_calc_err))

small_dataset_flag = 0

for r in range(R_vec.shape[0]):
    R = R_vec[r]
    print('================================================== ')
    print('                         R={}                      '.format(R))
    print('================================================== ')

    # Important - here we update R in the params in order to create masks with appropriate sampling
    # The mask is created in the DataTransform (utils/datasets
    params.R = R

    for qi in range(q_vec.shape[0]):
        q = q_vec[qi]
        params.q = q

        basic_data_folder = "/mikQNAP/NYU_knee_data/multicoil_efrat/5_JPEG_compressed_data/"

        data_type = 'test'
        im_type_str = 'full_im'  # training & validation is done on blocks (to accelerate training). Test is done on full-size images.

        params.data_path = basic_data_folder + data_type + "/q" + str(params.q) + "/" + im_type_str + "/"

        test_loader = create_data_loaders(params)

        N_test_batches = len(test_loader.dataset)
        print('N_test_batches =', N_test_batches)

        checkpoint_file = 'R{}_q{}/checkpoints/model_{}.pt'.format(R, q, checkpoint_num)

        checkpoint = torch.load(checkpoint_file,map_location=device)

        # load the parameters of the trained network
        params_loaded = checkpoint["params"]
        single_MoDL = UnrolledModel(params_loaded).to(device)

        single_MoDL.load_state_dict(checkpoint['model'])

        single_MoDL.eval()

        NRMSE_test_list = []
        SSIM_test_list = []

        cnt = 0
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                if iter % 10 == 0:
                    print('loading test batch ',iter)

                #input_batch, target_batch, mask_batch, target_no_JPEG_batch = data
                input_batch, target_batch, mask_batch = data

                # display the mask (before converting it to torch tensor)
                if (iter == 0):
                    # print('mask_batch shape:',mask_batch.shape)
                    mask_squeezed = mask_batch[0, :, :, 0].squeeze()
                    # fig = plt.figure()
                    # plt.imshow(mask_squeezed, cmap="gray")
                    # plt.title(params.sampling_flag + ' epoch 0, iter {}'.format(iter))
                    # plt.show()
                    # fig.savefig('mask_iter{}.png'.format(iter))

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
                    cnt += 1  # counts the number of test images
                    print('cnt={}'.format(cnt))

                    im_input = cplx.to_numpy(input_batch.cpu())[i, :, :]
                    im_target = cplx.to_numpy(target_batch.cpu())[i, :, :]
                    im_out = cplx.to_numpy(out_batch.cpu())[i, :, :]

                    MoDL_err = error_metrics(np.abs(im_target),np.abs(im_out))
                    MoDL_err.calc_NRMSE()
                    MoDL_err.calc_SSIM()

                    NRMSE_test_list.append(MoDL_err.NRMSE)
                    SSIM_test_list.append(MoDL_err.SSIM)

                    if cnt<N_calc_err:
                        NRMSE_examples_4display[r, qi, cnt - 1] = MoDL_err.NRMSE
                        SSIM_examples_4display[r, qi, cnt - 1] = MoDL_err.SSIM

                    if cnt<=N_examples_4display:
                        target_im_rotated = np.rot90(np.abs(im_target),2)
                        im_out_rotated = np.rot90(np.abs(im_out),2)
                        NX = im_out_rotated.shape[0]
                        NY = im_out_rotated.shape[1]


                        if (r==0) & (qi==0) & (iter==0):
                            TARGETS = np.zeros((NX,NY,q_vec.shape[0],N_examples_4display))
                            RECS = np.zeros((NX,NY,R_vec.shape[0],q_vec.shape[0],N_examples_4display))

                        TARGETS[:,:,qi,iter] = target_im_rotated
                        RECS[:,:,r,qi,iter] = im_out_rotated


                        #if iter==0:
                        fig = plt.figure()
                        plt.imshow(target_im_rotated, cmap="gray")
                        plt.colorbar(shrink=0.5)
                        plt.axis('off')
                        plt.title('target - iter={} - R{} q{}'.format(iter,R,q))
                        plt.show()
                        figname = 'check3_target_R{}_q{}_iter{}'.format(R,q,iter)
                        fig.savefig(figname)

                if iter >= N_examples_stats:
                    break


            # NRMSE - calc av & std                
            NRMSE_test_array = np.asarray(NRMSE_test_list)
            NRMSE_av = np.mean(NRMSE_test_array[0:N_examples_stats].squeeze())
            NRMSE_std = np.std(NRMSE_test_array[0:N_examples_stats].squeeze())
            NRMSE_av_vs_q_and_R[r,qi] = NRMSE_av
            NRMSE_std_vs_q_and_R[r, qi] = NRMSE_std

            # SSIM - calc av & std                
            SSIM_test_array = np.asarray(SSIM_test_list)
            SSIM_av = np.mean(SSIM_test_array[0:N_examples_stats].squeeze())
            SSIM_std = np.std(SSIM_test_array[0:N_examples_stats].squeeze())
            SSIM_av_vs_q_and_R[r,qi] = SSIM_av
            SSIM_std_vs_q_and_R[r, qi] = SSIM_std

            print('q={} NRMSE_av = {}, SSIM_av = {}'.format(q, NRMSE_av,SSIM_av))



# save NRMSE_av & SSIM
print('saving results')
results_filename = 'Res_for_Fig6.npz'
np.savez(results_filename, R_vec=R_vec,q_vec=q_vec,params=params,checkpoint_num=checkpoint_num,
         NRMSE_av_vs_q_and_R=NRMSE_av_vs_q_and_R,
         NRMSE_std_vs_q_and_R=NRMSE_std_vs_q_and_R,
         SSIM_av_vs_q_and_R=SSIM_av_vs_q_and_R,
         SSIM_std_vs_q_and_R=SSIM_std_vs_q_and_R,
         NRMSE_examples_4display=NRMSE_examples_4display,
         SSIM_examples_4display=SSIM_examples_4display,
         N_examples_stats=N_examples_stats,
         N_examples_4display=N_examples_4display,
         TARGETS=TARGETS,
         RECS=RECS,
         )
