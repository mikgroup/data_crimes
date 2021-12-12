# This script prepares the python commands for performing the DictL test runs
# The commands should be sent manually from the linux command line. The advantage of this approach is that these runs
# can be sent in parallel to many CPUs, on different servers.

# Alternatively, it's possible to use the DictL_recon script and run everything inside the loops. However, this
# computation will probably take several days.

# (c) Efrat Shimron, UC Berkeley, 2021

import numpy as np

# load results of the hyperparam search
container = np.load('opt_params_dict.npy',allow_pickle=True)


R = np.array([4])

N_examples = 122 # the number of slices in our test set
data_type_str = 'test'

pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])
sampling_type_vec = np.array([1,2])  # 0 = random, 1 = strong var-dens, 2 = weak var-dens

num_cpus = str(10) # number of CPUs that each run can employ


for samp_i in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[samp_i] == 1:
        samp_str = 'weak'
    elif sampling_type_vec[samp_i] == 2:
        samp_str = 'strong'

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):
        if (pad_ratio==1.0) | (pad_ratio==2.0):
            pad_ratio = int(pad_ratio)

        #print(f'------- pad{pad_ratio} {samp_str} VD -----')

        # get optimal parameters from the container
        dict0 = container.item()[(pad_ratio, samp_str)]
        block_shape = dict0['block_shape']
        num_filters = dict0['num_filters']
        nnz = dict0['nnz']
        max_iter = dict0['max_iter']
        lamda = dict0['lamda']

        logdir = 'test_logs/' + samp_str + '_pad_ratio_{}'.format(pad_ratio)

        # print command for test run - run these commands manually from the linux/unix termianl
        command_str = f'python3 DictL_recon.py --pad_ratio {pad_ratio} --samp_type {samp_str} --lamda {lamda} --block_shape {block_shape} --num_filters {num_filters} --nnz {nnz} --max_iter {max_iter} --num_slices {N_examples} --data_type {data_type_str} --num_cpus {num_cpus} --logdir {logdir}'
        print(command_str)



#print(container)