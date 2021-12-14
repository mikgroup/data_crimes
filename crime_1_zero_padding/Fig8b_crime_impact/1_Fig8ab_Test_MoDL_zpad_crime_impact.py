'''
This code performs the experiments for figure 8b.
It loads DL models that were trained earlier (for Fig5) using processed versions of the FastMRI proton-density (PD) knee data.
Each loaded network is tested twice: with processed and un-processed versions of the underlying test set.

Before running this code, update the basic_data_folder to your processed PD data folder. It should be the same one
as the one defined in the script Fig5_statistics_knee_data/data_prep/data_prep_zero_pad_crime.py
in this code line:
basic_out_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_data/"

(c) Efrat Shimron, UC Berkeley (2021)
'''
import logging
import os
import sys
# add path to functions library - when running on mikQNAP
sys.path.append("/mikQNAP/efrat/1_inverse_crimes/1_mirror_PyCharm_CS_MoDL_merged/SubtleCrimesRepo/")
sys.path.append("../")  # add folder above
sys.path.append("../Fig5_statistics_knee_data/DL/") # add folder in which the trained networks are saved


import matplotlib.pyplot as plt
import numpy as np
import torch
from functions.error_funcs import error_metrics

from MoDL_single import UnrolledModel
from utils import complex_utils as cplx
from utils.datasets import create_data_loaders


##################################################################################
#                 update these paths to YOUR paths
##################################################################################
basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_data/"
trained_models_folder = '../Fig5_statistics_knee_data/DL/'


##################################################################################
# Experiment set up
N_examples = 122 # number of examples that will be used for computing the mean and STD
print('N_examples=',N_examples)
R = int(4)

# create a folder for the test figures
if not os.path.exists('test_figs_R{}'.format(R)):
    os.makedirs('test_figs_R{}'.format(R))

use_multiple_GPUs_flag = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##################### create test loader ###########################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Hyper parameters
params = Namespace()
params.batch_size = 1
params.sampling_flag = 'var_dens_2D'
params.NX_full_FOV = 640
params.NY_full_FOV = 372


######################################################################################################
#                                     experiment settings
######################################################################################################

checkpoint_num= 69
unrolls = 6
pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])
var_dens_type_vec = np.array([0,1])

############################## distribution shift experiment ##########################################
### experiment - crime impact (distribution shift)
crime_impact_exp_flag = 1

######################################################################################################

NRMSE_test_set = np.zeros((N_examples,pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set  = np.zeros((N_examples,pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))

NRMSE_test_set_av  = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
NRMSE_test_set_std = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set_av   = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))
SSIM_test_set_std  = np.zeros((pad_ratio_vec.shape[0],var_dens_type_vec.shape[0]))

n_test_images = 0


# Important - here we update R in the params in order to create masks with appropriate sampling
# The mask is created in the DataTransform (utils/datasets
params.R = R


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

        # create a string label for the pad ratio
        if (pad_ratio == 1) | (pad_ratio == 2):
            #pad_ratio = int(pad_ratio)
            pad_ratio_str = str(int(pad_ratio))  # instead of 1.0 this will give 1
        else:
            pad_ratio_str = str(int(np.floor(pad_ratio))) + 'p' + str(int(100*(pad_ratio % 1)))
            print(pad_ratio_str)

        params.pad_ratio = pad_ratio  # zero-padding ratio

        if crime_impact_exp_flag==0:
            params.pad_ratio_test = params.pad_ratio
        elif crime_impact_exp_flag==1:
            params.pad_ratio_test = 1

        # define path to test data
        data_type = 'test'
        im_type_str = 'full_im'

        params.data_path = basic_data_folder + data_type + "/pad_" + str(
            int(100 * params.pad_ratio_test)) + "/" + im_type_str + "/"

        print('test data path:',params.data_path )

        # calib area is defined w.r.t. to TEST data size.
        calib_x = int(12 * (params.NX_full_FOV / 640) * params.pad_ratio_test)
        calib_y = int(12 * (params.NY_full_FOV / 640) * params.pad_ratio_test)
        params.calib = np.array([calib_x, calib_y])

        # Remember - the data loader defines the sampling mask. The test data must undergo the same transform as train data!
        test_loader = create_data_loaders(params,shuffle_flag=False)

        N_test_images = len(test_loader.dataset)
        print('N_test_images =', N_test_images)


        checkpoint_file = trained_models_folder + 'R{}_pad_{}_unrolls_{}_{}_var_dens/checkpoints/model_{}.pt'.format(
                                                                                   params.R,
                                                                                   str(int(100*pad_ratio)),
                                                                                   unrolls,
                                                                                   var_dens_flag,
                                                                                   checkpoint_num,
                                                                                   )
        checkpoint = torch.load(checkpoint_file,map_location=device)

        params_loaded = checkpoint["params"]
        single_MoDL = UnrolledModel(params_loaded).to(device)
        print('params.data_path: ', params.data_path)

        single_MoDL.load_state_dict(checkpoint['model'])

        single_MoDL.eval()

        NRMSE_test_list = []
        SSIM_test_list = []

        with torch.no_grad():
            for i_batch, data in enumerate(test_loader):
                if i_batch % 10 == 0:
                    print('loading test batch ',i_batch)


                input_batch, target_batch, mask_batch = data

                in_size =  input_batch.size()

                if (pad_i==0) & (v_i==0):
                    n_test_images += 1

                # # display the mask (before converting it to torch tensor)
                if (i_batch == 0):
                    # print('mask_batch shape:',mask_batch.shape)
                    mask_squeezed = mask_batch[0, :, :, 0].squeeze()
                    np.save('mask_squeezed',mask_squeezed)
                    print('saved mask squeezed')

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
fig.savefig('Fig_crime_impact_N_examples_{}'.format(N_examples))


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
fig.savefig('Fig_crime_impact_examples_{}'.format(N_examples))

print('saved figures successfully')

################################ save data ##################################333
# save SSIM_av & SSIM
results_filename = 'crime_impact_DNN_results_NRMSE_and_SSIM_N_examples_{}.npz'.format(N_examples)
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
