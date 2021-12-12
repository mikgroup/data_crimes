# useful commands for checking the runs:
# 1) how to find number of directions with the string "strong_VD_pad_ratio2" in their name:
# find strong_VD_pad_ratio_2* -type d | wc -l
# 2) how to find the number of above directions that have files with ".npz" extension:
# find logs/weak_VD_pad_ratio_2*/*.npz -type f | wc -l

# TODO: edit this script and add explanations about "gen commands" etc.


import numpy as np

# block size - the blocks were symmetric, e.g. 8x8
block_shape_MIN=8
block_shape_MAX=32
block_shape_STEP=8
block_shape_vec = np.arange(block_shape_MIN,block_shape_MAX,block_shape_STEP)
block_shape_vec = np.append(block_shape_vec,block_shape_MAX) # add the last element

# number of dictionary atoms
num_filters_MIN=100
num_filters_MAX=300
num_filters_STEP=100
num_filters_vec = np.arange(num_filters_MIN,num_filters_MAX,num_filters_STEP)
num_filters_vec = np.append(num_filters_vec,num_filters_MAX) # add the last element to the array

# sparsity level K (also denoted as Number of Non-Zero elements = NNS)
nnz_MIN=5
nnz_MAX=11
nnz_STEP=2
nnz_vec = np.arange(nnz_MIN,nnz_MAX,nnz_STEP)
nnz_vec = np.append(nnz_vec,nnz_MAX) # add the last element to the array

# number of outer iterations of the alternating minimization algorithm
max_iter_MIN=7
max_iter_MAX=13
max_iter_STEP=2
max_iter_vec = np.arange(max_iter_MIN,max_iter_MAX,max_iter_STEP)
max_iter_vec = np.append(max_iter_vec,max_iter_MAX) # add the last element to the array


#lamda_vec= np.array([1e-4,1e-3,0.01,1e-2,1e-1])
lamda_vec= np.array([1e-5,1e-4,1e-3,1e-2])


pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])

num_slices = np.array([5])

n_samp_types = np.array([2])  # weak VD, strong VD



################ search sent to a single server ##############
# here we assume that there is a single pad_ratio and single VD type
N_npz_files_server = block_shape_vec.shape[0]*num_filters_vec.shape[0]*nnz_vec.shape[0]*max_iter_vec.shape[0]*lamda_vec.shape[0]

print(f'N_npz_files_server=',N_npz_files_server)

t_run_minutes = 2.5*num_slices
N_cpus = 20
t_server_in_min = N_npz_files_server*t_run_minutes/N_cpus
t_server_in_days = t_server_in_min/(60*24) # this is for a SINGLE CPU!

print(f'compute time per server ({N_cpus} cpus) = {t_server_in_min} hours')
print(f'compute time per server ({N_cpus} cpus)= {t_server_in_days} days')

# ################### overall search space #################
N_npz_tot = N_npz_files_server*pad_ratio_vec.shape[0]*n_samp_types

t_single_npz_in_minutes = 2.5*num_slices

T_compute_tot_in_min = N_npz_tot*t_single_npz_in_minutes  #[minutes]
T_compute_tot_in_days = T_compute_tot_in_min/(60*24)

print(f'T_compute_tot on a single cpu: {T_compute_tot_in_days} days')



N_runs_tot = N_npz_tot*num_slices  # each ".npz" files contains the results for num_slices images
print(f'N_runs_tot: {N_runs_tot}')

