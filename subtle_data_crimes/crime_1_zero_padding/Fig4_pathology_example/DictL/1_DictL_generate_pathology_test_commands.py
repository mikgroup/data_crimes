'''
This script prepares the python commands for performing the DictL test runs.
It uses the DictL parameters that were found as optimal during the huge grid search that was performed for Fig5 (that
huge search was conducted over 200 CPUs for over 4 weeks).

The commands will be printed as the output of this script. They should be copied and then pasted into a linux terminal,
to send the runs manually. This approach has two advantages: (1) these runs can be sent in parallel to many CPUs, on
different servers, and (2) they can be sent using tmux, such that they will continue to run remotely even if the local
terminal is closed. We highly recommend using this appraoch, because the DictL runs usually take a long time!

Notice: before sending these runs, you should edit the script 2_DictL_recon_pathology.py and make sure that the
basic_data_folder defined there is identical to the output path defined in the data preparation script, which is:
Fig4_pathology_example/data_prep/data_prep_FatSatPD.py

(c) Efrat Shimron, UC Berkeley, 2021
'''

import numpy as np

# load results of the hyperparam search
container = np.load('../../Fig5_statistics_knee_data/DictL/opt_params_dict.npy',allow_pickle=True)


R = np.array([4])

N_examples = 1 # the number of slices in our test set
data_type_str = 'pathology_1'

pad_ratio_vec = np.array([1,2])
sampling_type_vec = np.array([1,2])  # 0 = random, 1 = strong var-dens, 2 = weak var-dens

num_cpus = str(10) # number of CPUs that each run can employ


for samp_i in range(sampling_type_vec.shape[0]):
    if sampling_type_vec[samp_i] == 1:
        samp_type = 'weak'
    elif sampling_type_vec[samp_i] == 2:
        samp_type = 'strong'

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):
        if (pad_ratio==1.0) | (pad_ratio==2.0):
            pad_ratio = int(pad_ratio)

        #print(f'------- pad{pad_ratio} {samp_type} VD -----')

        # get optimal parameters from the container
        dict0 = container.item()[(pad_ratio, samp_type)]
        block_shape = dict0['block_shape']
        num_filters = dict0['num_filters']
        nnz = dict0['nnz']
        max_iter = dict0['max_iter']
        lamda = dict0['lamda']

        logdir = data_type_str + '_results/' + samp_type + '_pad_ratio_{}_R'.format(pad_ratio,int(R))

        # print command for test run - run these commands manually from the linux/unix termianl
        command_str = f'python3 2_DictL_recon_pathology.py --pad_ratio {pad_ratio} --samp_type {samp_type} --lamda {lamda} --block_shape {block_shape} --num_filters {num_filters} --nnz {nnz} --max_iter {max_iter} --num_slices {N_examples} --data_type {data_type_str} --num_cpus {num_cpus} --logdir {logdir}'
        print(command_str)

