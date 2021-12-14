'''
This scripts reads the results that were saved during the hyperparameters grid search.

Notice: before running this script, you need to update all the parameters (e.g. block_shape_MIN, block_shape_MAX etc.)
such that they will match those that appear in the script gen_commands.sh

(c) Efrat Shimron (UC Berkeley, 2021)

'''

import numpy as np

# DictL parameters:

block_shape_MIN=8
block_shape_MAX=32
block_shape_STEP=8
block_shape_vec = np.arange(block_shape_MIN,block_shape_MAX,block_shape_STEP)
block_shape_vec = np.append(block_shape_vec,block_shape_MAX) # add the last element to the array

num_filters_MIN=100
num_filters_MAX=300
num_filters_STEP=100
num_filters_vec = np.arange(num_filters_MIN,num_filters_MAX,num_filters_STEP)
num_filters_vec = np.append(num_filters_vec,num_filters_MAX) # add the last element to the array

nnz_MIN=5
nnz_MAX=11
nnz_STEP=2
nnz_vec = np.arange(nnz_MIN,nnz_MAX,nnz_STEP)
nnz_vec = np.append(nnz_vec,nnz_MAX)

max_iter_MIN=7
max_iter_MAX=13
max_iter_STEP=2
max_iter_vec = np.arange(max_iter_MIN,max_iter_MAX,max_iter_STEP)
max_iter_vec = np.append(max_iter_vec,max_iter_MAX) # add the last element to the array

#lamda_vec= np.array([1e-4,1e-3,0.05,1e-2,1e-1])
lamda_vec= np.array([1e-5,1e-4,1e-3,1e-2])

# Data & sampling-related parameters:

pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])

samp_type_vec = np.array([1,2])  # 1 = weak VD, 2 = strong VD

# ------------------ find the lowest NRMSE for each combination of pad_ratio & sampling_type -----------------

N_pad_ratio = pad_ratio_vec.shape[0]
N_samp = samp_type_vec.shape[0]
N_block_shape = block_shape_vec.shape[0]
N_num_filters = num_filters_vec.shape[0]
N_nnz = nnz_vec.shape[0]
N_max_iter= max_iter_vec.shape[0]
N_lamda = lamda_vec.shape[0]

num_slices = 5

NRMSE_all_params_array = np.zeros([N_pad_ratio,N_samp,N_block_shape,N_num_filters,N_nnz,N_max_iter,N_lamda,num_slices])


for pad_i in range(N_pad_ratio):
    pad_ratio = pad_ratio_vec[pad_i]
    if (pad_ratio==1) | (pad_ratio==2):
        pad_ratio = int(pad_ratio)

    for samp_i in range(N_samp):
        if samp_type_vec[samp_i]==1:
            samp_type = 'weak'
        elif samp_type_vec[samp_i]==2:
            samp_type = 'strong'

        for block_i in range(N_block_shape):
            block_shape = block_shape_vec[block_i]

            for num_filter_i in range(N_num_filters):
                num_filters = num_filters_vec[num_filter_i]

                for nnz_i in range(N_nnz):
                    nnz = nnz_vec[nnz_i]

                    for max_iter_i in range(N_max_iter):
                        max_iter = max_iter_vec[max_iter_i]

                        for lamda_i in range(N_lamda):
                            lamda = lamda_vec[lamda_i]
                            if lamda==0.00001:
                                lamda_str = '1e-5'
                            elif lamda==0.0001:
                                lamda_str = '1e-4'
                            elif lamda==0.001:
                                lamda_str = '1e-3'
                            elif lamda==0.01:
                                lamda_str = '1e-2'
                            elif lamda==0.1:
                                lamda_str = '1e-1'



                            logdir = f'logs/{samp_type}_VD_pad_ratio_{pad_ratio}_lamda_{lamda_str}_block_shape_{block_shape}_num_filters_{num_filters}_nnz_{nnz}_max_iter_{max_iter}/'

                            filename = logdir + "res_NRMSE_SSIM.npz"


                            container = np.load(filename,allow_pickle=True)
                            NRMSE_arr = container['Dict_NRMSE_arr']
                            #SSIM_arr = container['Dict_SSIM_arr']

                            NRMSE_all_params_array[pad_i,samp_i,block_i,num_filter_i,nnz_i,max_iter_i,lamda_i,:] = NRMSE_arr


NRMSE_av_over_imgs = np.mean(NRMSE_all_params_array,7)  # compute average over the images axis

# find & save the optimal hyperparameters
dict_all = {}

for pad_i in range(N_pad_ratio):
    pad_ratio = pad_ratio_vec[pad_i]
    if (pad_ratio==1) | (pad_ratio==2):
        pad_ratio = int(pad_ratio)

    for samp_i in range(N_samp):

        if samp_type_vec[samp_i]==1:
            samp_type = 'weak'
        elif samp_type_vec[samp_i]==2:
            samp_type = 'strong'

        print('-------')
        dict0 = {}

        NRMSE_sub_array = NRMSE_av_over_imgs[pad_i,samp_i,::] #,:,:,:,:,:,:]
        #print(f'NRMSE_sub_array shape: {NRMSE_sub_array.shape}')

        min_NRMSE = np.min(NRMSE_sub_array)
        inds = np.unravel_index(np.argmin(NRMSE_sub_array, axis=None), NRMSE_sub_array.shape)
        #print(inds)

        # get the optimal value for each param
        opt_block_shape = block_shape_vec[inds[0]]
        opt_num_filters = num_filters_vec[inds[1]]
        opt_nnz =  nnz_vec[inds[2]]
        opt_max_iter = max_iter_vec[inds[3]]
        opt_lamda = lamda_vec[inds[4]]

        dict0['block_shape']=opt_block_shape
        dict0['num_filters']=opt_num_filters
        dict0['nnz']=opt_nnz
        dict0['max_iter']=opt_max_iter
        dict0['lamda']=opt_lamda

        print(f'pad_ratio={pad_ratio} samp_type={samp_type} - min_NRMSE={min_NRMSE} params: {dict0}' )

        dict_all[pad_ratio,samp_type] = dict0

np.save('opt_params_dict.npy',dict_all)

