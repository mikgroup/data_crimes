##############################################################################
# To run this code, use the conda virtual environment "subtle_env"
# (two identical environments were defined on mikneto or mikshoov)
###############################################################################
# documentation:
# moved the code to PRE_SUBMISSION/Fig5../DL/
# renaming:
# NRMSE_av_vs_pad_ratio_and_R --> NRMSE_test_set_av
# NRMSE_std_vs_pad_ratio_and_R --> NRMSE_test_set_std
# SSIM_av_vs_pad_ratio_and_R --> SSIM_test_set_av
# SSIM_std_vs_pad_ratio_and_R --> SSIM_test_set_std
##########################################################################


import logging
import os
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
from PIL import Image

# import custom libraries
from utils import complex_utils as cplx
from functions.error_funcs import error_metrics

N_examples = 122 # number of examples that will be used for computing the mean and STD
print('N_examples=',N_examples)

#R_vec = np.array([4])
R = int(4)
#print('R={}'.format(R))

# create a folder for the test figures
if not os.path.exists('test_figs_R{}'.format(R)):
    os.makedirs('test_figs_R{}'.format(R))

use_multiple_GPUs_flag = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# TODO: "params" is defind here but it is also LOADED later in the code - check why we need it both here and there
# Hyper parameters
params = Namespace()
#params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_data_w_Ke_preprocessing/test/"  # NO JPEG
#params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_data_w_Ke_preprocessing_q10/test/"
#params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_data_w_Ke_preprocessing_q10/train/"
params.batch_size = 1
#params.num_grad_steps = 6  # number of unrolls
#params.num_cg_steps = 8
#params.share_weights = True
#params.modl_lamda = 0.05
#params.lr = 0.0001
#params.weight_decay = 0
#params.lr_step_size = 500
#params.lr_gamma = 0.5
#params.epoch = 70
#params.calib = 24  # TODO: calib should depend on pad_ratio
#params.R = 4
# params.sampling_flag = 'random_uniform'
#params.sampling_flag = 'var_dens'
# params.sampling_flag = 'var_dens_1D'
print('2D VAR DENS')
params.sampling_flag = 'var_dens_2D'
#params.var_dens_flag = args.var_dens_flag
params.NX_full_FOV = 640
params.NY_full_FOV = 372


######################################################################################################
#                                     experiment settings
######################################################################################################

# Remember: the test data must go through the same transforms (e.g. sampling) as the training data

#checkpoint_vec = np.array([0])
#print('###### CHANGE THIS - checkpoint is {}'.format(checkpoint_vec))
checkpoint_vec = np.array([69])




unrolls = 6



small_dataset_flag = 0

#pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])  # np.array([1,2])
pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])

#TODO: change the next vec from var_dens_type_vec to samp_type_vec and change the values to [1,2] for compatibility with the rest of the code
var_dens_type_vec = np.array([0,1]) # np.array([0,1])    #0 = weak var dens, 1 = strong var dens


######################################################################################################

NRMSE_test_set = np.zeros((N_examples,pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set  = np.zeros((N_examples,pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))

NRMSE_test_set_av  = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
NRMSE_test_set_std = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set_av   = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set_std  = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))

n_test_images = 0

# for r in range(R_vec.shape[0]):
#     R = R_vec[r]
#     print('================================================== ')
#     print('                         R={}                      '.format(R))
#     print('================================================== ')


# Important - here we update R in the params in order to create masks with appropriate sampling
# The mask is created in the DataTransform (utils/datasets
params.R = R

# for qi in range(q_vec.shape[0]):
#     q = q_vec[qi]
#     print('========= q={} ======== '.format(q))

for v_i in range (var_dens_type_vec.shape[0]):

    if var_dens_type_vec[v_i]==0:
        var_dens_flag = 'weak'
    elif var_dens_type_vec[v_i]==1:
        var_dens_flag = 'strong'

    params.var_dens_flag = var_dens_flag

    print("loading a network trained for " + var_dens_flag + ' var dens')

    for pad_i in range(pad_ratio_vec.shape[0]):
        pad_ratio = pad_ratio_vec[pad_i]
        print('======== pad_ratio={} ======== '.format(pad_ratio))

        if (pad_ratio == 1) | (pad_ratio == 2):
            #pad_ratio = int(pad_ratio)
            pad_ratio_str = str(int(pad_ratio))  # instead of 1.0 this will give 1
        else:
            pad_ratio_str = str(int(np.floor(pad_ratio))) + 'p' + str(int(100*(pad_ratio % 1)))
            print(pad_ratio_str)

        params.pad_ratio = pad_ratio  # zero-padding ratio

        if pad_ratio >= 3:
            use_multiple_GPUs_flag = 1
        else:
            use_multiple_GPUs_flag = 0

        # use_multiple_GPUs_flag = 0

        # load test data
        # if small_dataset_flag == 1:
        #     print('*** using a SMALL dataset ***')
        #     params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_zpad_data_v15_x" + str(int(100*pad_ratio)) + "_small_dataset/test/"
        #
        # elif small_dataset_flag == 0:
        #
        #      params.data_path = "/mikQNAP/NYU_knee_data/singlecoil_efrat/1_zpad_data_v15_x" + str(int(100*pad_ratio)) + "/test/"

        basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/subtle_inv_crimes_zpad_data_v18"

        if small_dataset_flag == 1:
            print('*** using a SMALL dataset ***')
            basic_data_folder = basic_data_folder + '_small/'
        else:
            basic_data_folder = basic_data_folder + '/'

        # path to test data
        data_type = 'test'
        im_type_str = 'full_im'
        params.data_path = basic_data_folder + data_type + "/pad_" + str(
            int(100 * params.pad_ratio)) + "/" + im_type_str + "/"

        print('test data path:',params.data_path )

        #params.calib = int(24 * pad_ratio)
        #print('calib = ',params.calib)

        # calib is assumed to be 12 for NX=640
        calib_x = int(12 * (params.NX_full_FOV/640) * pad_ratio)
        calib_y = int(12 * (params.NY_full_FOV/640) * pad_ratio)
        params.calib = np.array([calib_x, calib_y])

        # Remember - the data loader defines the sampling mask. The test data must undergo the same transform as train data!
        test_loader = create_data_loaders(params,shuffle_flag=False)

        N_test_images = len(test_loader.dataset)
        print('N_test_images =', N_test_images)

        for model_i in range(checkpoint_vec.shape[0]):
            checkpoint_num = int(checkpoint_vec[model_i])
            print('checkpoint_num:',checkpoint_num)



            # load trained network
            if small_dataset_flag==1:
                #checkpoint_file = 'R{}_pad_ratio_{}_unrolls_{}_small_dataset/checkpoints/model_{}.pt'.format(params.R, pad_ratio, unrolls,checkpoint_num)
                checkpoint_file = 'R{}_pad_{}_unrolls_{}_{}_var_dens_small_data/checkpoints/model_{}.pt'.format(
                    params.R, str(int(100*pad_ratio)),
                    unrolls,
                    var_dens_flag,
                    checkpoint_num,
                    )
            elif small_dataset_flag==0:
                #checkpoint_file = 'R{}_pad_ratio_{}_unrolls_{}/checkpoints/model_{}.pt'.format(params.R,pad_ratio, unrolls,checkpoint_num)
                checkpoint_file = 'R{}_pad_{}_unrolls_{}_{}_var_dens/checkpoints/model_{}.pt'.format(
                                                                                       params.R,
                                                                                       str(int(100*pad_ratio)),
                                                                                       unrolls,
                                                                                       var_dens_flag,
                                                                                       checkpoint_num,
                                                                                       )
                # checkpoint_file = 'R{}_pad_{}_unrolls_{}_small_data_{}_var_dens/checkpoints/model_{}.pt'.format(
                #     params.R, pad_ratio, unrolls, var_dens_flag, checkpoint_num)

            print('checkpoint_file=')
            print(checkpoint_file)

            checkpoint = torch.load(checkpoint_file,map_location=device)

            params_loaded = checkpoint["params"]
            single_MoDL = UnrolledModel(params_loaded).to(device)


            # single_MoDL.display_zf_image_flag = 1
            # single_MoDL.zf_im_foldername = "test_figs"

            print('params.data_path: ', params.data_path)
            print('params.batch_size: ', params.batch_size)

            # Data Parallelism - enables running on multiple GPUs
            if (torch.cuda.device_count() > 1) & (use_multiple_GPUs_flag == 1):
                print("Now using ", torch.cuda.device_count(), "GPUs!")
                single_MoDL = nn.DataParallel(single_MoDL, device_ids=[0,1,2, 3])  # the first index on the device_ids determines which GPU will be used as a staging area before scattering to the other GPUs
            else:
                print("Now using a single GPU")


            single_MoDL.load_state_dict(checkpoint['model'])

            single_MoDL.eval()

            NRMSE_test_list = []
            SSIM_test_list = []

            with torch.no_grad():
                for i_batch, data in enumerate(test_loader):
                    if i_batch % 10 == 0:
                        print('loading test batch ',i_batch)


                    #input_batch, target_batch, mask_batch, target_no_JPEG_batch = data
                    input_batch, target_batch, mask_batch = data

                    # # debugging - check target
                    # im_target = cplx.to_numpy(target_batch.cpu())[0, :, :]
                    # fig = plt.figure()
                    # plt.imshow(np.abs(im_target),cmap="gray")
                    # plt.title('target')
                    # plt.colorbar()
                    # plt.show()


                    in_size =  input_batch.size()
                    print('in_size=',in_size)


                    if (pad_i==0) & (v_i==0):
                        n_test_images += 1
                        print('n_test_images=',n_test_images)

                    #print('input_batch ',input_batch.shape)
                    #print('target_batch ', target_batch.shape)


                    # # display the mask (before converting it to torch tensor)
                    if (i_batch == 0) & (checkpoint_num == checkpoint_vec[-1]):
                        # print('mask_batch shape:',mask_batch.shape)
                        mask_squeezed = mask_batch[0, :, :, 0].squeeze()
                        np.save('mask_squeezed',mask_squeezed)
                        print('saved mask squeezed')
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


                        if (i_batch>=5) &  (i_batch<=15) & (i<=3):
                            # print('input_batch.shape: ', input_batch.shape)
                            # print('out_batch.shape: ', out_batch.shape)
                            # print('target_batch.shape: ', target_batch.shape)
                            # print('mask_batch.shape: ', mask_batch.shape)

                            im_out_rotated = np.rot90(np.abs(im_out),2)
                            im_gold_rotated = np.rot90(np.abs(im_target),2)

                            fig = plt.figure()
                            plt.subplot(1,2,1)
                            plt.imshow(im_out_rotated,cmap="gray")
                            plt.colorbar(shrink=0.5)
                            plt.title('im_out MoDL')
                            #plt.axis('off')

                            plt.subplot(1, 2, 2)
                            plt.imshow(im_gold_rotated, cmap="gray")
                            plt.colorbar(shrink=0.5)
                            # plt.axis('off')
                            plt.title('target')
                            plt.suptitle('Results R{} pad_x{} MoDL{} - example {}'.format(R, pad_ratio_str, checkpoint_num,i))
                            plt.show()
                            fig.savefig('test_figs_R{}/test_images_pad_x{}_MoDL{}_example{}'.format(R,pad_ratio_str,checkpoint_num,i))

                            # zoom-in
                            x1 = 325
                            x2 = 370
                            y1 = 70
                            y2 = 160
                            # scale the zoom-in coordinates to fit changing image size
                            x1s = int(x1 * pad_ratio)
                            x2s = int(x2 * pad_ratio)
                            y1s = int(y1 * pad_ratio)
                            y2s = int(y2 * pad_ratio)

                            fig = plt.figure()
                            plt.subplot(1,2,1)
                            plt.imshow(im_out_rotated[x1s:x2s, y1s:y2s],cmap="gray")
                            plt.colorbar(shrink=0.5)
                            plt.title('im_out MoDL')
                            #plt.axis('off')

                            plt.subplot(1, 2, 2)
                            plt.imshow(im_gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                            plt.colorbar(shrink=0.5)
                            # plt.axis('off')
                            plt.title('target')
                            plt.suptitle('Results R{} pad_x{} MoDL{} - example {}'.format(R, pad_ratio_str, checkpoint_num,i))
                            plt.show()
                            fig.savefig('test_figs_R{}/test_images_pad_x{}_MoDL{}_example{}_zoom'.format(R,pad_ratio_str,checkpoint_num,i))



                            #print('debug')


                NRMSE_array = np.asarray(NRMSE_test_list)
                NRMSE_av = np.mean(NRMSE_array[0:N_examples].squeeze())
                NRMSE_std = np.std(NRMSE_array[0:N_examples].squeeze())

                NRMSE_test_set[:,pad_i,v_i] = NRMSE_array
                NRMSE_test_set_av[pad_i,v_i] = NRMSE_av
                NRMSE_test_set_std[pad_i,v_i] = NRMSE_std

                SSIM_array = np.asarray(SSIM_test_list)
                SSIM_av = np.mean(SSIM_array[0:N_examples].squeeze())
                SSIM_std = np.std(SSIM_array[0:N_examples].squeeze())

                SSIM_test_set[:, pad_i, v_i] = SSIM_array
                SSIM_test_set_av[pad_i,v_i] = SSIM_av
                SSIM_test_set_std[pad_i,v_i] = SSIM_std


                print('pad_ratio={} NRMSE_av={} SSIM_av={}'.format(pad_ratio, NRMSE_av, SSIM_av))



#################### prep for plots #################33

# NOTICE: the following code creates and saves the figures, but for some reason the axis labels aren't
# displayed properly.
# for better figures run the script fig4ISMRM_Test_MoDL_run5 after running this script

# prepare x ticks labels for the NRMSE and SSIM graphs
x = pad_ratio_vec
x_ticks_labels = []
for i in range(pad_ratio_vec.shape[0]):
    x_ticks_labels.append('x{}'.format(pad_ratio_vec[i]))


markers=['o', 's', 'v', 'h', '8']


#########################   plots  ####################################################################

# display NRMSE
fig = plt.figure()
for v_i in range(var_dens_type_vec.shape[0]):
    if var_dens_type_vec[v_i] == 0:
        var_dens_flag = 'weak'
    elif var_dens_type_vec[v_i] == 1:
        var_dens_flag = 'strong'

    label_str = var_dens_flag + " var-dens"


    plt.errorbar(pad_ratio_vec,NRMSE_test_set_av[:,v_i].squeeze(),yerr=NRMSE_test_set_std[:,v_i].squeeze(),linestyle='solid', label=label_str,
                     marker=markers[v_i])


#plt.ylim((0,0.075))
plt.ylim(0.0035, 0.027)
plt.xlabel('Zero padding ratio', fontsize=18)
plt.ylabel('NRMSE', fontsize=20)
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.locator_params(axis='y', nbins=3)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.legend(fontsize=20)
plt.title('N_examples={}'.format(N_examples))
plt.show()
fig.savefig('RESULTS_imSize372_NRMSE_N_examples_{}'.format(N_examples))


# Display SSIM
fig = plt.figure()
for v_i in range(var_dens_type_vec.shape[0]):
    if var_dens_type_vec[v_i] == 0:
        var_dens_flag = 'weak'
    elif var_dens_type_vec[v_i] == 1:
        var_dens_flag = 'strong'

    label_str = var_dens_flag + " var-dens"

    plt.errorbar(pad_ratio_vec,SSIM_test_set_av[:,v_i].squeeze(),yerr=SSIM_test_set_std[:,v_i].squeeze(),linestyle='solid', label=label_str,
                     marker=markers[v_i])


#plt.ylim((0,0.075))
plt.ylim(0.72, 0.98)
plt.xlabel('Zero padding ratio', fontsize=18)
plt.ylabel('SSIM', fontsize=20)
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.locator_params(axis='y', nbins=3)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.legend(fontsize=20,loc="lower right")
plt.title('N_examples={}'.format(N_examples))
plt.show()
fig.savefig('RESULTS_imSize372_SSIM_N_examples_{}'.format(N_examples))

print('saved figures successfully')

################################ save data ##################################333
# save SSIM_av & SSIM
results_filename = 'DNN_results_NRMSE_and_SSIM_N_examples_{}.npz'.format(N_examples)
np.savez(results_filename, R=R, pad_ratio_vec=pad_ratio_vec, params=params, checkpoint_num=checkpoint_num,
         var_dens_type_vec = var_dens_type_vec,
         NRMSE_test_set = NRMSE_test_set,
         NRMSE_test_set_av=NRMSE_test_set_av,
         NRMSE_test_set_std=NRMSE_test_set_std,
         SSIM_test_set = SSIM_test_set,
         SSIM_test_set_av = SSIM_test_set_av,
         SSIM_test_set_std = SSIM_test_set_std)
print('saved .npz file')
# np.savez(tmp_filename, loss_vs_i_batch_vec=loss_vs_i_batch_vec, NRMSE_vs_i_batch_vec=NRMSE_vs_i_batch_vec)
