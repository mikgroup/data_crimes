'''
To run the CS test session, please run the following from the linux command line:
./gen.commands.sh
cat run.sh | xargs -n1 -I{} -P20 bash -c {} > log.txt   (replace 20 with the number of CPUs that you want to use)


Documentation
This script runs the set of experiments for the CS statistics in Fig 5 (Subtle Inverse Crimes paper).
Before running this script, please run the following scripts:
1_hyperparam_search_CS_lamda_zpad_crime.py - that script performs the hyperparam (lamda) search
2_find_optimal_lamda_and_display_results - scripts takes the results of the previous one and finds the optimal lamda for each image size.

Instructions - how to run the CS reconstruction experiments in parallel by using this script:
1. Edit the bash script gen_commands.sh - edit ot only in linux, NOT in WINDOWS/WSL/PYCHARM! (or it will not compile).
2. run it: ./gen_commands.sh
   This will create a scritp named run.sh which includes all the runs required for the statistics
3. Execute the script "run.sh" from the linux command line as follows (P20 will send 20 runs in paralle):
cat run.sh | xargs -n1 -I{} -P20 bash -c {} > log.txt

Notice: before running this script, make sure that the basic_data_folder defined here is identical to the output path
defined in Fig5../data_prep/data_prep_zero_pad_crime.py

(c) Efrat Shimron, UC Berkeley, 2021
'''
####################################################################

import os
from optparse import OptionParser

import h5py
import matplotlib.pyplot as plt
# limit the number of CPUs that the process can take
import mkl
import numpy as np
import sigpy as sp
from sigpy import mri as mr

from subtle_data_crimes.functions.error_funcs import error_metrics
from subtle_data_crimes.functions.sampling_funcs import gen_2D_var_dens_mask

mkl.set_num_threads(
    5)  # the number in the brackets determines the number of CPUs. 1 is recommended for the DictL algorithm! Otherwise there's a lot of overhead (when the run is spread accross multiple cpus) and the comptuation time becomes longer.

# limit the number of CPUs that the process can take to 1
import mkl

mkl.set_num_threads(
    1)  # the number in the brackets determines the number of CPUs. 1 is recommended for the DictL algorithm! Otherwise there's a lot of overhead (when the run is spread accross multiple cpus) and the comptuation time becomes longer.


def get_args():
    parser = OptionParser()

    parser.add_option('--R', '--R', type='int', default=[4], help='desired R')
    parser.add_option('--pad_ratio', '--pad_ratio', type='float', default=1,
                      help='zero-padding ratio (preprocessed data will be chosen accordingly)')
    parser.add_option('--num_slices', '--num_slices', type='int', default=122, help='number of slices')
    parser.add_option('--samp_type', '--samp_type', type='str', default='weak', help=' random / weak / strong /')
    parser.add_option('--data_type', '--data_type', type='str', default='test',
                      help='train/val/test/pathology_1/pathology_2')

    parser.add_option('--logdir', default="./", type=str,
                      help='log dir')  # this is useful for sending many runs in parallel (e.g. for parameter calibration)

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    # data-related variables
    # The following params/flags determine the type of preprocessed data that will be loaded. All the datasets were already preprocessed and stored.
    pad_ratio = args.pad_ratio
    data_type = args.data_type  # 'train' / 'val' / 'test'.
    im_type_str = 'full_im'  # Options: 'full_im' / 'blocks' (blocks are used for training Deep Learning models, not for CS & DictL).

    # hard-coded variables
    device = sp.cpu_device  # which device to use (not all is supported on GPU)
    # mode = 'omp'  # for the dictionary learning algorithm

    # sampling parameters
    R = np.asfarray(args.R).astype('int')

    # experiment setup
    num_slices = args.num_slices
    small_dataset_flag = 0

    samp_type = args.samp_type  # 'random' = random uniform / 'weak' = weak variable-density / 'strong' = strong var-dens

    print(f'samp_type={samp_type}')
    print(f'pad_ratio={pad_ratio}')

    # get the value of the CS_lamda (load results saved during a hyperparam calibration run)
    container = np.load('CS_optimal_lamda.npz')
    loaded_pad_ratio_vec = container['pad_ratio_vec']
    loaded_sampling_type_vec = container['sampling_type_vec']
    loaded_optimal_lamda_arr = container['optimal_lamda_arr']

    if samp_type == 'weak':
        samp_i = np.argwhere(loaded_sampling_type_vec == 1)  # 0 = random, 1 = strong var-dens, 2 = weak var-dens
    elif samp_type == 'strong':
        samp_i = np.argwhere(loaded_sampling_type_vec == 2)  # 0 = random, 1 = strong var-dens, 2 = weak var-dens

    pad_i = np.argwhere(loaded_pad_ratio_vec == pad_ratio)

    CS_lamda = loaded_optimal_lamda_arr[pad_i, samp_i].item()
    print(f'CS lambda = {CS_lamda}')

    # initialize dictionaries and arrays
    # Note: we use dictionaries (not numpy arrays) for storing the reconstructions beause the zero-padded images have different sizes
    gold_dict = {}  # a python dictionary that will contain the "gold standard" recons
    CS_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing

    CS_NRMSE_arr = np.empty([num_slices])
    CS_SSIM_arr = np.empty([num_slices])

    # Create log directory - this is useful when sending many runs in parallel
    logdir = args.logdir

    print('logdir:', logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # #################################################################################
    # ##                                Experiments
    # #################################################################################

    t = 0  # counts loaded scans. each scan contains multiple slices.
    ns = 0  # counts loaded slices

    basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_data/"

    if small_dataset_flag == 1:
        basic_data_folder = basic_data_folder + '_small/'
    else:
        basic_data_folder = basic_data_folder + '/'

    data_path = basic_data_folder + data_type + "/pad_" + str(
        int(100 * pad_ratio)) + "/" + im_type_str + "/"

    files_list = os.listdir(data_path)

    while ns < num_slices:

        print(' === loading h5 file {} === '.format(t))
        # Load k-space data
        filename_h5 = data_path + files_list[t]

        print('t=', t)
        print('filename_h5=', filename_h5)
        f = h5py.File(filename_h5, 'r')

        t += 1  # update the number of LOADED scans. Each scan contains multiple slices

        kspace_preprocessed_multislice = f["kspace"]
        im_RSS_multislice = f[
            "reconstruction"]  # these are the RSS images produced from the zero-padded k-space - see fig. 1 in the paper

        n_slices_in_scan = kspace_preprocessed_multislice.shape[0]

        for s_i in range(n_slices_in_scan):
            # for s_i in slices_to_use:
            print(f'slice {s_i}')

            kspace_slice = kspace_preprocessed_multislice[s_i, :, :].squeeze()
            im_RSS = im_RSS_multislice[s_i, :, :].squeeze()

            ns += 1  # number of slices
            print(f'ns {ns}/{num_slices}')

            # ------- run recon experiment -----------------
            # print('Gold standard rec from fully sampled data...')

            rec_gold = sp.ifft(kspace_slice)

            # display
            # if ns<5:
            #     fig = plt.figure()
            #     plt.imshow(np.abs(np.rot90(rec_gold, 2)), cmap="gray")
            #     plt.title('rec_gold')
            #     plt.colorbar()
            #     plt.show()

            # check NaN values
            assert np.isnan(rec_gold).any() == False, 'there are NaN values in rec_gold! scan file {} slice {}'.format(
                filename_h5,
                s_i)
            gold_dict[ns - 1] = rec_gold
            imSize = im_RSS.shape

            # calib is assumed to be 12 for NX=640
            calib_x = int(12 * im_RSS.shape[0] / 640)
            calib_y = int(12 * im_RSS.shape[1] / 640)
            calib = np.array([calib_x, calib_y])

            # ========== sampling mask =========
            mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, samp_type, calib=calib)

            ksp_padded_sampled = np.multiply(kspace_slice, mask)

            # # # sanity check
            # if ns<=3:
            #     fig = plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(np.log(np.abs(kspace_slice)),cmap="gray")
            #     plt.title('kspace_slice')
            #     plt.subplot(1,2,2)
            #     plt.title('kspace sampled')
            #     plt.imshow(np.log(np.abs(ksp_padded_sampled)),cmap="gray")
            #     #plt.subplot(1, 3, 3)
            #     #plt.imshow(mask, cmap="gray")
            #     plt.show()

            # # ###################################### CS rec  ################################################

            mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils
            ksp_padded_sampled_expanded = np.expand_dims(ksp_padded_sampled, axis=0)
            virtual_sens_maps = np.ones_like(
                ksp_padded_sampled_expanded)  # sens maps are all ones because we have a "single-coil" magnitude image.

            # CS recon from sampled data
            print('CS rec from sub-sampled data...')
            rec_CS = mr.app.L1WaveletRecon(ksp_padded_sampled_expanded, virtual_sens_maps, lamda=CS_lamda,
                                           show_pbar=False).run()

            CS_Err = error_metrics(rec_gold, rec_CS)
            CS_Err.calc_NRMSE()
            CS_Err.calc_SSIM()

            CS_NRMSE_arr[ns - 1] = CS_Err.NRMSE
            CS_SSIM_arr[ns - 1] = CS_Err.SSIM

            print(f'NRMSE {CS_Err.NRMSE}')

            if ns <= 10:
                cmax = np.max(np.abs(rec_gold))
                fig = plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(np.rot90(np.abs(rec_gold), 2), cmap="gray")
                plt.title('rec_gold')
                plt.clim(0, cmax)
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.imshow(np.rot90(np.abs(rec_CS), 2), cmap="gray")
                plt.title('CS - NRMSE={:0.2}'.format(CS_Err.NRMSE))
                plt.clim(0, cmax)
                plt.colorbar()
                plt.suptitle(f'zero-padding x{pad_ratio} R{R} {samp_type} VD - scan {ns} slice s_i={s_i}')
                plt.show()

            if ns == num_slices:
                break

NRMSE_av = np.mean(CS_NRMSE_arr)

print(f'done - n_slices={ns}')
print('saving NRMSE and SSIM arrays')
filename = logdir + "/CS_res_NRMSE_SSIM"
np.savez(filename, CS_NRMSE_arr=CS_NRMSE_arr, CS_SSIM_arr=CS_SSIM_arr)

print(f'RES: {samp_type} VD pad {pad_ratio} CS_NRMSE_Av={CS_Err.NRMSE :.4f}')
