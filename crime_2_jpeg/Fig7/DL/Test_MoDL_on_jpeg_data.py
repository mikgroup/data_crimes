##############################################################################
# To run this code, use the conda virtual environment "subtle_env"
# (two identical environments were defined on mikneto or mikshoov)
###############################################################################
# TODO: remove the option for small dataset

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
figs_foldername = 'test_figs'
if not os.path.exists(figs_foldername):
    os.makedirs(figs_foldername)

##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# TODO: "params" is defind here but it is also LOADED later in the code - check why we need it both here and there
# Hyper parameters
params = Namespace()
#params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_data_w_Ke_preprocessing/test/"  # NO JPEG
params.batch_size = 1

# image dimensions
params.NX = 640
params.NY = 372

# calib is assumed to be 12 for NX=640
calib_x = int(12)
calib_y = int(12 * params.NY / params.NX)
params.calib = np.array([calib_x, calib_y])
params.shuffle_flag = False # should be True for training, False for testing. Notice that this is not a string, semicolons aren't necessary.

# params.sampling_flag = 'random_uniform'
# params.sampling_flag = 'var_dens_1D'
print('2D VAR DENS')
params.sampling_flag = 'var_dens_2D'
params.var_dens_flag = 'weak'  # 'weak' / 'strong'

checkpoint_num = int(69)

#q_vec = np.array([10,20,50,100])
q_vec = np.array([10,20,50,75,100])
q_vec = np.array([20,50,75,999])
#q_vec = np.array([10,20,50,75,100])
#q_vec = np.array([100])

#R_vec = np.array([4])
R_vec = np.array([2,3,4])

print('R_vec=',R_vec)

N_examples_4display=20 # number of examples to display & save recs

N_examples = 122 # number of examples over which the mean and STD will be computed

DL_NRMSE_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])
DL_SSIM_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])

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
        print('========= q={} ======== '.format(q))

        if q == 100:
            use_multiple_GPUs_flag = 0 # can be 1 if there are multiple GPUs available
        else:
            use_multiple_GPUs_flag = 0

        basic_data_folder = "/mikQNAP/NYU_knee_data/multicoil_efrat/5_JPEG_compressed_data"

        if small_dataset_flag == 1:
            print('using SMALL DATASET')
            basic_data_folder = basic_data_folder + '_small/'
            run_foldername = 'R{}_q{}_small_dataset'.format(params.R, params.q)
        else:
            basic_data_folder = basic_data_folder + '/'
            run_foldername = 'R{}_q{}'.format(params.R, params.q)

        data_type = 'test'
        im_type_str = 'full_im'  # training & validation is done on blocks (to accelerate training). Test is done on full-size images.

        params.data_path = basic_data_folder + data_type + "/q" + str(params.q) + "/" + im_type_str + "/"

        print(f'CHECK THIS: params.data_path= {params.data_path}')
        test_loader = create_data_loaders(params)  # params.R is important here! it defines the sampling mask

        N_test_batches = len(test_loader.dataset)
        print('N_test_batches =', N_test_batches)


        if small_dataset_flag == 1:
            checkpoint_file = 'R{}_q{}_small_dataset/checkpoints/model_{}.pt'.format(R,q,checkpoint_num)
        elif small_dataset_flag == 0:
            checkpoint_file = 'R{}_q{}/checkpoints/model_{}.pt'.format(R, q, checkpoint_num)

        checkpoint = torch.load(checkpoint_file,map_location=device)

        params_loaded = checkpoint["params"]
        single_MoDL = UnrolledModel(params_loaded).to(device)

        print('params.data_path: ', params.data_path)
        print('params.batch_size: ', params.batch_size)

        # Data Parallelism - enables running on multiple GPUs
        if (torch.cuda.device_count() > 1) & (use_multiple_GPUs_flag == 1):
            print("Now using ", torch.cuda.device_count(), "GPUs!")
            single_MoDL = nn.DataParallel(single_MoDL, device_ids=[0,1,2,3])  # the first index on the device_ids determines which GPU will be used as a staging area before scattering to the other GPUs
        else:
            print("Now using a single GPU")


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

                #print('input_batch ',input_batch.shape)
                #print('target_batch ', target_batch.shape)


                # # display the mask (before converting it to torch tensor)
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
                    #print('im_input.shape',im_input.shape)
                    #print('im_target.shape', im_target.shape)

                    # # normalize the target image (no need to normalize the output of the network)
                    # scale = calc_scaling_factor(kspace)
                    #
                    # kspace = kspace / scale
                    # im_target = im_target / scale
                    # #target_no_JPEG = target_no_JPEG / scale


                    MoDL_err = error_metrics(np.abs(im_target),np.abs(im_out))
                    MoDL_err.calc_NRMSE()
                    MoDL_err.calc_SSIM()
                    #print('NRMSE={:0.3f}'.format(MoDL_err.NRMSE))

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


                        # if iter==0:
                        #     fig = plt.figure()
                        #     plt.imshow(target_im_rotated, cmap="gray")
                        #     plt.colorbar(shrink=0.5)
                        #     plt.axis('off')
                        #     plt.title('target - R{} q{} i{} iter{} cnt{}'.format(R,q,i,iter,cnt))
                        #     plt.show()
                        #     figname = figs_foldername + 'sanity_check_target_R{}_q{}_iter{}'.format(R,q,iter)
                        #     fig.savefig(figname)
                        #     print('............saved fig iter 0.............')

                        # fig = plt.figure()
                        # plt.subplot(1,2,1)
                        # plt.imshow(target_im_rotated,cmap="gray")
                        # plt.colorbar(shrink=0.5)
                        # plt.axis('off')
                        # plt.title('target')
                        #
                        # plt.subplot(1,2,2)
                        # plt.imshow(im_out_rotated,cmap="gray")
                        # plt.colorbar(shrink=0.5)
                        # plt.title('MoDL rec (test session) NRMSE={:.3f}'.format(MoDL_err.NRMSE))
                        # plt.axis('off')
                        # plt.suptitle('Results R{} q{} MoDL{} - example {}'.format(R,q,checkpoint_num,i))
                        # plt.show()
                        #
                        # plt.suptitle('Test images: q {} R {} MoDL checkpoint {}'.format(q,R,checkpoint_num))
                        # fig.savefig('test_figs/test_images_R{}_q{}_MoDL{}_example{}'.format(R,q,checkpoint_num,i))

                        # # display concatenated figure
                        #a = np.concatenate((im_target, im_out), axis=1)
                        #s1 = im_target.shape[1]
                        # fig = plt.figure()
                        # plt.imshow(np.abs(a), cmap="gray", origin='lower')
                        # plt.title(
                        #     'Run10 ' + params.sampling_flag + ': target ; MoDL single coil rec, checkpoint #{}'.format(
                        #         checkpoint_num))
                        # plt.text(  # position text relative to Axes
                        #     1.1 * s1, 0.05 * s1, 'NRMSE {:.3f}'.format(MoDL_err.NRMSE),
                        #     ha='left', va='center',
                        #     color="yellow", fontsize=12)
                        # plt.colorbar()
                        # plt.show()
                        # fname = 'fig_checkpoint_{}.png'.format(checkpoint_num)
                        # fig.savefig(fname)

                        print('debug')

                if iter >= N_examples:
                    break


            # Store results in NRMSE & SSIM arrays (for saving)
            NRMSE_test_array = np.asarray(NRMSE_test_list)
            DL_NRMSE_array[qi,r,:] = NRMSE_test_array

            SSIM_test_array = np.asarray(SSIM_test_list)
            DL_SSIM_array[qi, r, :] = SSIM_test_array


            # calc av & std for figures in this script (not for saving)
            NRMSE_av = np.mean(NRMSE_test_array[0:N_examples].squeeze())
            NRMSE_std = np.std(NRMSE_test_array[0:N_examples].squeeze())
            NRMSE_av_vs_q_and_R[r, qi] = NRMSE_av
            NRMSE_std_vs_q_and_R[r, qi] = NRMSE_std

            SSIM_av = np.mean(SSIM_test_array[0:N_examples].squeeze())
            SSIM_std = np.std(SSIM_test_array[0:N_examples].squeeze())
            SSIM_av_vs_q_and_R[r,qi] = SSIM_av
            SSIM_std_vs_q_and_R[r, qi] = SSIM_std

            #print('q={} NRMSE_av = {}, SSIM_av = {}'.format(q, NRMSE_av,SSIM_av))



            # # plot
            # if checkpoint_num==69:
            #     a = np.concatenate((im_target, im_out), axis=1)
            #     s1 = target.shape[1]
            #
            #     fig = plt.figure()
            #     plt.imshow(np.abs(a),cmap="gray",origin='lower')
            #     plt.title('Run10 ' + params.sampling_flag + ': target ; MoDL single coil rec, checkpoint #{}'.format(checkpoint_num))
            #     plt.text(  # position text relative to Axes
            #                         1.1 * s1, 0.05 * s1, 'NRMSE {:.3f}'.format(MoDL_err.NRMSE),
            #                         ha='left', va='center',
            #                         color="yellow", fontsize=12)
            #     plt.colorbar()
            #     plt.show()
            #     fname = 'fig_checkpoint_{}.png'.format(checkpoint_num)
            #     fig.savefig(fname)


compression_vec = (100-q_vec)/100
print('compression_vec=',compression_vec)

# display NRMSE
fig = plt.figure()
for r in range(R_vec.shape[0]):
    #plt.plot(compression_vec,NRMSE_av_vs_q_and_R[r,:],fmt='-o',label='R={}'.format(R_vec[r]))
    plt.errorbar(compression_vec,NRMSE_av_vs_q_and_R[r,:].squeeze(),yerr=NRMSE_std_vs_q_and_R[r,:].squeeze(),fmt='-o',label='R={}'.format(R_vec[r]))
    #plt.xlabel('JPEG Compresseion - (100-q)/100')
    plt.xlabel('JPEG Compresseion')
    plt.ylabel('NRMSE')
    plt.ylim((0,0.075))
ax = plt.gca()
ax.legend(fontsize=14,loc="upper right")
plt.title('N_examples = '.format(N_examples))
plt.show()
fig.savefig(f'RESULTS_NRMSE_vs_q_{params.var_dens_flag}_VD')

# Display SSIM
fig = plt.figure()
for r in range(R_vec.shape[0]):
    #plt.plot(compression_vec,SSIM_av_vs_q_and_R[r,:],fmt='-o',label='R={}'.format(R_vec[r]))
    plt.errorbar(compression_vec,SSIM_av_vs_q_and_R[r,:].squeeze(),yerr=SSIM_std_vs_q_and_R[r,:].squeeze(),fmt='-o',label='R={}'.format(R_vec[r]))
    #plt.xlabel('JPEG Compresseion - (100-q)/100')
    plt.xlabel('JPEG Compresseion')
    plt.ylabel('SSIM')
    plt.ylim((0,1))
ax = plt.gca()
ax.legend(fontsize=14,loc="lower right")
plt.title('N_examples = {}'.format(N_examples))
plt.show()
fig.savefig(f'RESULTS_SSIM_vs_q_{params.var_dens_flag}_VD')


# save NRMSE_av & SSIM
print('saving results')
results_filename = 'Res_MODL_JPEG_exp.npz'
np.savez(results_filename, R_vec=R_vec,q_vec=q_vec,params=params,checkpoint_num=checkpoint_num,
         # NRMSE_av_vs_q_and_R=NRMSE_av_vs_q_and_R,
         # NRMSE_std_vs_q_and_R=NRMSE_std_vs_q_and_R,
         # SSIM_av_vs_q_and_R=SSIM_av_vs_q_and_R,
         # SSIM_std_vs_q_and_R=SSIM_std_vs_q_and_R,
         # NRMSE_examples_4display=NRMSE_examples_4display,
         # SSIM_examples_4display=SSIM_examples_4display,
         DL_NRMSE_array=DL_NRMSE_array,
         DL_SSIM_array=DL_SSIM_array,
         N_examples=N_examples,
         N_examples_4display=N_examples_4display
         )

# # the arrays of images (TARGETS and RECS) are huge so they are saved separately
#results_filename = 'RECS_and_TARGETS.npz'
# np.savez(results_filename, TARGETS=TARGETS,RECS=RECS,R_vec=R_vec,q_vec=q_vec,)

#np.savez(tmp_filename, loss_vs_iter_vec=loss_vs_iter_vec, NRMSE_vs_iter_vec=NRMSE_vs_iter_vec)


print('results saved successfully')
