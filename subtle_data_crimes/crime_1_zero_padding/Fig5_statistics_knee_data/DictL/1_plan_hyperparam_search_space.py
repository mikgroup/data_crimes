'''
This code  is used only for *planning* the DictL hyperparameter search space, not for running it.
It does not save any output. I used it only for planning the grid search and estimating the number of required runs.

To conduct the hyperparam search, do this:
1. Edit the gen_commands.sh script located in this folder. Notice that it is a bash script - DO NOT edit it in PyCharm
or in any windows software, because that could create problems. The best practice is to edit it directly from a linux
or Windows Subsystem for Linux (WSL) terminal.

2. After editing, run it as follows:
./gen_commands.sh
This will create a set of other bash scripts, with names such as run_pad_1.75_strong.sh - this specific script will
run a grid search for padding ratio of 1.75 and strong variable density.
Each of these scripts will contain lines for sending dozens of other runs, which are required for the hyperparam search.
Each run will be conducted separaetly, and the results will be saved in a subfolder of the logs directory (the subfolders
will be created automatically).

3. Then, run the above scripts as follows:
./run_pad_1.75_strong.sh

The advantage of the above approach is that these runs can be sent in parallel to many CPUs, on different servers.
Tip:
In order to send 20 runs in parallel (i.e. to 20 CPUs on one server), run this (in the linux command line):
cat run_xxxxxx.sh | xargs -n1 -I{} -P20 bash -c {} > log.txt

Notice that this huge set of runs is expected to take a VERY LONG TIME! In our lab it was conducted over 200 CPUs in
parallel, and it required about 4 weeks.

If you don't have the required resources, you can skip the grid search stage (i.e. skip scripts 1-3 in this fodler)
and use the optimal parameters that we found, which are stored in the file opt_params_dict.npy
However, please notice that they were calibrated for our specific datasets and sampling schemes, and that usage of every
other dataset/sampling scheme will require a new hyperparam search.

See more instructions in the next script.

(c) Efrat Shimron (UC Berkeley, 2021)
'''


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

