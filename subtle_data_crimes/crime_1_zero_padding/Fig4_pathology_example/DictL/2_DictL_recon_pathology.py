# Instructions - how to calibrate parameters by sending many parallel runs:
# 1. gen_commands.sh - edit this script using linux only, DO NOT EDIT IT IN WINDOWS/PYCHARM! it donesn't compile afterwards.
# 2. make it excutable by running:
# 3. run it: ./gen_commands.sh
#    This will create a scritp named run.sh
# 4. Mkae the print_timestamp script exectuable by running:
#    chmod +x print_timestamp.sh
# 5. TODO: explain here that "gen_commands" will create different run files for each pad_ratio, and they can be sent separately
# In order to send 20 runs in parallel, run this (in the linux command line):
# cat run.sh | xargs -n1 -I{} -P20 bash -c {} > log.txt

####################################################################
# version documentation:
# version 5 - (1) now using data prepared by data_prep_v17,
#             (2) data is taken from the VALIDATION set only
#             (3) pad_ratio is added as an input variable - this makes the script specific to zero-padding experiments
#             (4) lamda is added as an input argument to this script, and also as an input to the script functions.dict_learn_funcs.DictionaryLearningMRI (previously it used the default value)
#             (4) num_slices = 10 by default, i.e. the script runs computations for 10 images and saves both their specific NRMSEs and the average NRMSE over these images.
#             (5) the default of block_shape was changed from [8,8] to 8, in order to make it a single input variable. later in the code it's converted from 8 to [8,8]
#             (6) the iterations over R and pad_ratio_vec were cancelled. Instead, we take a single value for R and a value for pad_ratio from the input args
# version 6 - identical to version 5
# version 7 - identical to version 6
# version 8 - identical to version 8
##########################################################################################
import os
from optparse import OptionParser

import h5py
# add path to functions library - when running on mikQNAP
import matplotlib.pyplot as plt
import mkl
import numpy as np
import sigpy as sp

from subtle_data_crimes.functions import error_metrics, gen_2D_var_dens_mask
from subtle_data_crimes.functions.dict_learn_funcs import DictionaryLearningMRI


def get_args():
    # parser = argparse.ArgumentParser(description="Script for Dictionary Learning.")
    parser = OptionParser()

    parser.add_option('--R', '--R', type='int', default=[4], help='desired R')
    parser.add_option('--nnz', '--num_nonzero_coeffs', type='int', default=7,
                      help='num_nonzero_coeffs controls the sparsity level when  Dictionary Learning runs with A_mode=''omp'' ')
    parser.add_option('--num_filters', '--num_filters', type='int', default=141, help='num_filters for Dict Learning')
    parser.add_option('--max_iter', '--max_iter', type='int', default=10, help='number of iterations')
    parser.add_option('--batch_size', '--batch_size', type='int', default=500, help='batch_size')
    parser.add_option('--block_shape', '--block_shape', type='int', default=8, help='block_shape')
    parser.add_option('--block_strides', '--block_strides', type='int', default=[4, 4], help='block_strides')
    # parser.add_option('--nu', '--nu', type='int', default=0.1, help='nu for Dict Learning')

    # new in version 5:
    parser.add_option('--pad_ratio', '--pad_ratio', type='float', default=1,
                      help='zero-padding ratio (preprocessed data will be chosen accordingly)')
    parser.add_option('--lamda', '--lamda', type='float', default=0.01,
                      help='lamda (tradeoff between data consistency and regularization terms')
    parser.add_option('--num_slices', '--num_slices', type='int', default=10, help='number of slices')
    parser.add_option('--samp_type', '--samp_type', type='str', default='weak', help=' random / weak / strong /')
    parser.add_option('--data_type', '--data_type', type='str', default='pathology_1',
                      help='train/val/test/pathology_1/pathology_2')

    parser.add_option('--num_cpus', '--num_cpus', type='int', default='1', help='number of CPUs that will be employed')

    parser.add_option('--logdir', default="./", type=str,
                      help='log dir')  # this is useful for sending many runs in parallel (e.g. for parameter calibration)

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    # print(args)

    # Create log directory - this is useful when sending many runs in parallel
    logdir = args.logdir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Dictionary-Learning algorithm parameters
    # num_nonzero_coeffs = args.nnz
    nnz_lamda = args.nnz
    max_iter = args.max_iter
    num_filters = args.num_filters
    batch_size = args.batch_size
    block_shape = [args.block_shape, args.block_shape]  # this should be a list, not a numpy array
    block_strides = args.block_strides
    lamda = args.lamda
    nu_lamda = args.lamda  # this is the regularization paramter that controls the tradeoff between Data Consistency and regularization terms. In the DictL app it's called "nu"

    # limit the number of CPUs that the process can take
    num_cpus = args.num_cpus
    # It is recommended to use only 1 CPU per run during the hyperparam search, because this search includes thousands of runs.
    # During the test there are only several runs, so more CPUs per run can be used
    mkl.set_num_threads(num_cpus)

    # data-related variables
    # The following params/flags determine the type of preprocessed data that will be loaded. All the datasets were already preprocessed and stored.
    pad_ratio = args.pad_ratio
    data_type = args.data_type  # 'train' / 'val' / 'test'.
    im_type_str = 'full_im'  # Options: 'full_im' / 'blocks' (blocks are used for training Deep Learning models, not for CS & DictL).

    if (pad_ratio == 1) | (pad_ratio == 2):
        pad_ratio_str = int(pad_ratio)
    else:
        pad_ratio_str = pad_ratio

    # hard-coded variables
    # n_proc = 40  # number of cpu cores to use, when possible
    device = sp.cpu_device  # which device to use (not all is supported on GPU)
    mode = 'omp'  # for the dictionary learning algorithm

    # sampling parameters
    R = np.asfarray(args.R).astype('int')
    samp_type = args.samp_type  # 'random' = random uniform / 'weak' = weak variable-density / 'strong' = strong var-dens

    # experiment setup
    num_slices = args.num_slices
    small_dataset_flag = 0

    print(f'samp_type={samp_type}')
    print(f'pad_ratio={pad_ratio}')

    # initialize dictionaries and arrays

    gold_dict = {}  # a python dictionary that will contain the gold standard recons
    # CS_recs_dict = {}    # a python dictionary that will contain the reconstructions obtained with Compressed Sensing
    DictL_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Dictionary learning

    # CS_NRMSE_arr = np.empty([num_slices])
    # CS_SSIM_arr = np.empty([num_slices])
    Dict_NRMSE_arr = np.empty([num_slices])
    Dict_SSIM_arr = np.empty([num_slices])

    pathology_slice = np.array([22])

    # #################################################################################
    # ##                                Experiments
    # #################################################################################

    t = 0  # counts loaded scans. each scan contains multiple slices.
    ns = 0  # counts loaded slices

    basic_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_FatSat_data/"

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

        for s_i in range(pathology_slice.shape[0]):
            s = pathology_slice[s_i]
            print(f'slice {s}')

            kspace_slice = kspace_preprocessed_multislice[s, :, :].squeeze()
            im_RSS = im_RSS_multislice[s, :, :].squeeze()

            ns += 1  # number of slices
            print(f'ns={ns}')

            # ------- run recon experiment -----------------
            print('Gold standard rec from fully sampled data...')
            rec_gold = sp.ifft(kspace_slice)

            # -------------- display & save -------------

            gold_rotated = np.rot90(np.abs(rec_gold), 2)
            cmax = np.max(np.abs(gold_rotated))
            # zoom-in coordinates for pathology 1
            x1 = 335
            x2 = 380
            y1 = 210
            y2 = 300
            # scale the zoom-in coordinates to fit changing image size
            x1s = int(335 * pad_ratio)
            x2s = int(380 * pad_ratio)
            y1s = int(210 * pad_ratio)
            y2s = int(300 * pad_ratio)

            # save gold as ".png"
            fig = plt.figure()
            plt.imshow(gold_rotated, cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/gold_slice{ns}_pad_{pad_ratio_str}'
            fig.savefig(figname, dpi=1000)

            # save gold as ".eps"
            fig = plt.figure()
            plt.imshow(gold_rotated, cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/gold_slice{ns}_pad_{pad_ratio_str}.eps'
            fig.savefig(figname, format='eps', dpi=1000)

            fig = plt.figure()
            plt.imshow(gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/gold_zoomed_slice{ns}_pad_{pad_ratio_str}.png'
            fig.savefig(figname, dpi=1000)

            # check NaN values
            assert np.isnan(rec_gold).any() == False, 'there are NaN values in rec_gold! scan {} slice {}'.format(n, s)

            gold_dict[
                ns - 1] = rec_gold  # store the results in a dictionary (note: we use a dictionary instead of a numpy array beause

            imSize = im_RSS.shape

            # calib is assumed to be 12 for NX=640
            calib_x = int(12 * im_RSS.shape[0] / 640)
            calib_y = int(12 * im_RSS.shape[1] / 640)
            calib = np.array([calib_x, calib_y])

            # ========== sampling mask =========
            mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, samp_type, calib=calib)

            ksp_padded_sampled = np.multiply(kspace_slice, mask)

            ################################## Dictionary Learning rec #####################################3
            _maps = None
            img_ref = None

            with device:
                app = DictionaryLearningMRI(ksp_padded_sampled,
                                            mask,
                                            _maps,
                                            num_filters,
                                            batch_size,
                                            block_shape,
                                            block_strides,
                                            nnz_lamda,  # sparsity level, i.e. number of nonzero coefficients
                                            nu=nu_lamda,
                                            A_mode='omp',
                                            # A_mode='l1',
                                            D_mode='ksvd',
                                            D_init_mode='data',
                                            DC=True,
                                            max_inner_iter=20,
                                            # max_inner_iter=10,
                                            max_iter=max_iter,
                                            img_ref=img_ref,
                                            device=device)
                out = app.run()

            rec_DictLearn = app.img_out

            # save the results
            filename = logdir + f'/rec_gold_pad_{pad_ratio_str}_slice{s}'
            np.savez(filename, rec_gold=rec_gold)
            filename = logdir + f'/rec_DictL_pad_{pad_ratio_str}_{samp_type}_VD_slice{s}'
            np.savez(filename, rec_DictLearn=rec_DictLearn)

            DictL_ERR = error_metrics(rec_gold, rec_DictLearn)
            DictL_ERR.calc_NRMSE()
            DictL_ERR.calc_SSIM()

            Dict_NRMSE_arr[ns - 1] = DictL_ERR.NRMSE
            Dict_SSIM_arr[ns - 1] = DictL_ERR.SSIM

            # # ================== display results - DL =================
            # if (pad_ratio==1) & (sampling_type_vec[j]==0):  # display only for the case without padding & sampling is 'weak variable densit'
            cmax = np.max([np.abs(rec_gold), np.abs(rec_DictLearn)])
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.rot90(np.abs(rec_gold), 2), cmap="gray")
            plt.title('gold standard')
            plt.colorbar(shrink=0.5)
            plt.clim(0, cmax)

            plt.subplot(1, 2, 2)
            plt.imshow(np.rot90(np.abs(rec_DictLearn), 2), cmap="gray")
            Dict_str = 'NRMSE={:.3f}'.format(DictL_ERR.NRMSE)
            plt.text(0.05 * imSize[1], 0.99 * imSize[0], Dict_str, color="yellow")
            plt.title("Dictionary Learning")
            plt.suptitle(
                f'R={int(R)} {samp_type} VD samp; pad {pad_ratio}; subject {t} ; slice {s}; {data_type} data; \n lamda={lamda}, sparsity_level={nnz_lamda}, block size {block_shape}, num_atoms {num_filters}')
            plt.colorbar(shrink=0.5)
            plt.clim(0, cmax)
            plt.show()
            fig_name = logdir + '/DictL_rec_slice{}.png'.format(ns - 1)
            fig.savefig(fig_name)

            # -------------- display & save -------------

            # gold_rotated = np.rot90(np.abs(rec_gold), 2)
            rec_DL_rotated = np.rot90(np.abs(rec_DictLearn), 2)

            fig = plt.figure()
            plt.imshow(gold_rotated, cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/target_im{ns}_pad_{pad_ratio_str}.eps'
            fig.savefig(figname, format='eps', dpi=1000)

            fig = plt.figure()
            plt.imshow(gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/target_im{ns}_pad_{pad_ratio_str}_zoomed.png'
            fig.savefig(figname, dpi=1000)

            fig = plt.figure()
            plt.imshow(rec_DL_rotated[x1s:x2s, y1s:y2s], cmap="gray")
            plt.axis('off')
            plt.clim(0, cmax)
            plt.show()
            figname = logdir + f'/DL_rec_im{ns}_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed.eps'
            fig.savefig(figname, format='eps', dpi=1000)

            if ns == num_slices:
                break

            gold_dict[pad_ratio, samp_type] = gold_rotated
            DictL_recs_dict[pad_ratio, samp_type] = rec_DL_rotated

            # save results
            # filename =

print(f'done - n_slices={ns}')
print('saving NRMSE and SSIM arrays')

# save the NRMSE and SSIM arrays
# filename = logdir + "/NRMSE_SSIM"
# np.savez(filename,Dict_NRMSE_arr=Dict_NRMSE_arr,Dict_SSIM_arr=Dict_SSIM_arr)  # notice: the names of the stored arrays begin with "Dict", not "DictL" (for convenience only)

# # save the recons
# gold_filename = logdir + '/gold_dict.npy'
# np.save(gold_filename, gold_dict)
# DictL_filename = logdir +'DictL_recs_dict.npy'
# np.save(DictL_filename, DictL_recs_dict)
