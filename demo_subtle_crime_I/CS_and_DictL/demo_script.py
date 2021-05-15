'''
This module defines the experiments for demonstrating subtle_crime_I for Compressed Sensing (CS) and Dictionary Learning (DictL) algorithms.

(c) Efrat Shimron, UC Berkeley (2021)
'''

import os
import numpy as np
import h5py
import sys

# add path to functions library - when running on mikQNAP
sys.path.insert(0,'..')

import matplotlib.pyplot as plt
from PIL import Image

import sigpy as sp
from sigpy import mri as mr
from functions.error_funcs import error_metrics
from functions.sampling_funcs import gen_2D_var_dens_mask

from functions.dict_learn_funcs import DictionaryLearningMRI
from functions.zpad_funcs import zpad_merge_scale
from optparse import OptionParser
import argparse

sys.path.append("/home/efrat/anaconda3/")
sys.path.append("/home/efrat/anaconda3/lib/python3.7/site-packages/")  # path to sigpy


def get_args():
    # parser = argparse.ArgumentParser(description="Script for Dictionary Learning.")
    parser = OptionParser()
    parser.add_option('--sim_flag', '--sim_flag', type='int', default=1, help='simulation type') # see options in the code below. 1 = pathology case A, 2 = pathology case B, 3 = experiment with 50 images
    parser.add_option('--DictL_flag', '--DictL_flag', type='int', default=[0], help='flag for DictL reconstruction')  # 1 = run DictL recon, 0 = do not run it
    parser.add_option('--num_slices', '--num_slices', type='int', default=1, help='number of slices')
    parser.add_option('--R_vec', '--R_vec', type='int', default=[4], help='desired R')
    parser.add_option('--nnz', '--num_nonzero_coeffs', type='int', default=6,
                      help='num_nonzero_coeffs controls the sparsity level when  Dictionary Learning runs with A_mode=''omp'' ')
    parser.add_option('--num_filters', '--num_filters', type='int', default=224, help='num_filters for Dict Learning')
    parser.add_option('--max_iter', '--max_iter', type='int', default=5, help='number of iterations')
    parser.add_option('--batch_size', '--batch_size', type='int', default=500, help='batch_size')
    parser.add_option('--block_shape', '--block_shape', type='int', default=[8, 8], help='block_shape')
    parser.add_option('--block_strides', '--block_strides', type='int', default=[4, 4], help='block_strides')
    parser.add_option('--nu', '--nu', type='int', default=0.1, help='nu for Dict Learning')

    parser.add_option('--logdir', default="./", type=str,
                      help='log dir')  # this is useful for sending many runs in parallel (e.g. for parameter calibration)

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    print(args)

    # Create log directory - this is useful when sending many runs in parallel
    logdir = args.logdir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # input variables
    simulation_flag = args.sim_flag
    # options:
    # 1 = pathology case #1 (from FastMRI)
    # 2 = pathology case #2 (from FastMRI)
    # 3 = statistics over many examples (without the above pathologies

    DictL_flag = args.DictL_flag

    num_slices = args.num_slices
    # num_realizations = args.num_realizations
    R_vec = np.asfarray(args.R_vec).astype('int')
    num_nonzero_coeffs = args.nnz
    max_iter = args.max_iter
    num_filters = args.num_filters
    batch_size = args.batch_size
    block_shape = args.block_shape
    block_strides = args.block_strides
    nu = args.nu # nu = lambda*2 (it is a paramter the controls the tradeoff between Data Consistency and sparsity terms)

    # hard-coded variables
    n_proc = 40  # number of cpu cores to use, when possible
    device = sp.cpu_device  # which device to use (not all is supported on GPU)
    mode = 'omp'

    show_flag = 1

    #################################################################################
    ## Experiment set-up
    #################################################################################

    R_vec = np.array([4])
    #pad_ratio_vec = np.array([1])
    #pad_ratio_vec = np.array([1])
    pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])
    #pad_ratio_vec = np.array([1, 1.5, 2, 2.5])
    print('pad_ratio_vec=',pad_ratio_vec)

    sampling_type_vec = np.array([1,2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens
    sampling_flag = '2D'

    # # alpha_vec = np.array([0.5,1,50])
    # alpha_vec = np.array([0, 0.3, 100])
    # R_vec = np.array([4])
    # num_realizations = 1;  # number of sampling masks that will be generated for each case
    #
    # # Define the desired padding ratios here:
    # pad_ratio_vec = np.arange(1, 3.1,
    #                           1)  # this defines a numpy array with the desired padding ratios:  ratio=N_padded/N_original



    if simulation_flag == 1:
        print('running PATHOLOGY CASE #1')
        N_examples = 1  # desired number of slices (for statistics)
    elif simulation_flag == 2:
        print('running PATHOLOGY CASE #2')
        N_examples = 1  # desired number of slices (for statistics)
    else:
        N_examples = 50  # desired number of slices (for statistics)

    print('N_examples wanted= ', N_examples)

    # #################################################################################
    # ##                              Initialize arrays & dicts
    # #################################################################################

    lamda = 1e-5

    gold_dict = {}  # a python dictionary that will contain the gold standard recons
    CS_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing
    Dict_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Dictionary learning
    masks_dict = {}  # a python dictionary that will contain the sampling masks

    CS_NRMSE_arr = np.empty([N_examples, pad_ratio_vec.shape[0], R_vec.shape[0], sampling_type_vec.shape[0]])
    CS_SSIM_arr = np.empty([N_examples, pad_ratio_vec.shape[0], R_vec.shape[0], sampling_type_vec.shape[0]])
    Dict_NRMSE_arr = np.empty([N_examples, pad_ratio_vec.shape[0], R_vec.shape[0], sampling_type_vec.shape[0]])
    Dict_SSIM_arr = np.empty([N_examples, pad_ratio_vec.shape[0], R_vec.shape[0], sampling_type_vec.shape[0]])

    home_dir = os.listdir("/mikQNAP/NYU_knee_data/multicoil_train/")

    n = 0  # runs over all scans in the data folder Each scan has 20-30 slices.
    ns = 0  # counts the number of loaded SLICES (not scans). All of them have shape (670,372).

    for r in range(R_vec.shape[0]):
        R = R_vec[r]
        print('R=', R)



        while ns < (N_examples):  # TODO: check the stopping criterion
            kspace_loaded_flag = 0

            # #################################################################################
            # ##                        Load Data - kspace
            # #################################################################################

            # ------- choose slices ---------
            if simulation_flag == 1:  # pathology case #1
                print('loading pathology case #1, ns={}'.format(ns))
                # Load k-space data
                f = h5py.File("/mikQNAP/NYU_knee_data/multicoil_train/file1000425.h5", 'r')
                kspace_orig = np.array(f["kspace"])
                kspace_loaded_flag = 1
                fig_str = 'pathology_1'

                # Here we add zero-padding with a few pixels in order to make the image size divide perfectly by 8x8 blocks (which are used by the DictL algorithm).
                # Specifically for the FastMRI knee data, we change the im_size from [640x356] to [640x376] because 376 is a multiplication of 8.
                kspace = np.zeros((kspace_orig.shape[0], kspace_orig.shape[1], kspace_orig.shape[2], 376),
                                  dtype=kspace_orig.dtype)
                kspace[:, :, :, 2:(376 - 2)] = kspace_orig

                slices_to_store = np.array([22])  # choose the slice with the pathology (this was hard-coded)

                # Here we define factors to get the coordinates for zoom-in on the pathology area later (for display purposes)
                # we define factors rather than specific pixel locations because images that have different zero-padding facotrs will have different sizes.
                xf1 = 0.5
                xf2 = 0.15
                yf1 = 0.58
                yf2 = 0.4


            elif simulation_flag == 2:  # pathology case #1
                print('loading pathology case #2, ns={}'.format(ns))

                f = h5py.File("/mikQNAP/NYU_knee_data/multicoil_train/file1002455.h5", 'r')
                kspace_orig = np.array(f["kspace"])
                kspace = kspace_orig
                kspace_loaded_flag = 1
                fig_str = 'pathology_2'
                # ns += 1

                # Here we add zero-padding with a few pixels in order to make the image size divide perfectly by 8x8 blocks (which are used by the DictL algorithm).
                # Specifically for the FastMRI knee data, we change the im_size from [640x356] to [640x376] because 376 is a multiplication of 8.
                kspace = np.zeros((kspace_orig.shape[0], kspace_orig.shape[1], kspace_orig.shape[2], 360),
                                  dtype=kspace_orig.dtype)
                kspace[:, :, :, 2:(360 - 2)] = kspace_orig

                slices_to_store = np.array([26])  # choose the slice with the pathology (this was hard-coded)

                # define zoom-in factors for displaying the pathology area later
                xf1 = 0.47  # x-factor 1
                xf2 = 0.55
                yf1 = 0.58
                yf2 = 0.9

            else:  # run statistics
                print('statistics run - loading FastMRI data, n={}'.format(n))
                # load data from the directory home_dir

                if home_dir[n] == 'file1000425.h5':
                    print('skipping pathology example #1')  # pathology case 1
                elif home_dir[n] == 'file1002455.h5':
                    print('skipping pathology example #3')  # pathology case 2
                else:

                    print(' ###### ===================== loading scan n={} ==================== ######'.format(n))

                    # Load k-space data
                    f = h5py.File("/mikQNAP/NYU_knee_data/multicoil_train/" + home_dir[n], 'r')

                    kspace_orig = np.array(f["kspace"])

                    if kspace_orig.shape[3] == 372:  # we use only slices that have size [640x372]
                        kspace_loaded_flag = 1

                        # Add padding with a few pixels - change from size [640x372] to [640x376] because 376 is a multiplication of 8 (later we'll use 8x8 filters in the dictionary learning)
                        kspace = np.zeros((kspace_orig.shape[0], kspace_orig.shape[1], kspace_orig.shape[2], 376),
                                          dtype=kspace_orig.dtype)
                        kspace[:, :, :, 2:(376 - 2)] = kspace_orig

                        N_slices = kspace.shape[0]

                        # throw out the 5 edge slices on each side of the scan, because they contain mostly noise.
                        slices_to_store = np.arange(7, (N_slices - 7), 1)

            ###########################################################################################
            #                                 Recon Experiments
            ###########################################################################################

            if kspace_loaded_flag == 1:


                N_slices_to_store = slices_to_store.shape[0]
                print('N_slices_to_store=', N_slices_to_store)

                print('kspace.shape: ', kspace.shape)

                NX = kspace.shape[2]
                NY = kspace.shape[3]


                for s_i in range(N_slices_to_store):
                    if ns<N_examples:
                        s = slices_to_store[s_i]
                        print('load slice {}'.format(s))
                        #print(' ------------------------- slice {} ---------------------'.format(s))
                        ns += 1  # count the number of loaded SLICES.
                        print('============= ns (valid slice) {} ===================='.format(ns))

                        for i, pad_ratio in enumerate(pad_ratio_vec):

                            if (pad_ratio == 1) | (pad_ratio == 2) | (pad_ratio==3):
                                pad_ratio = int(pad_ratio)
                                pad_str = str(int(pad_ratio))  # instead of 1.0 this will give 1
                            else:
                                pad_str = str(int(np.floor(pad_ratio))) + 'p' + str(int(100 * (pad_ratio % 1)))
                                print(pad_str)



                            print(' ------------------- pad ratio {} -----------------------'.format(pad_ratio))

                            # new code (following data_prep_v11)
                            ksp_block_multicoil = kspace[s, :, :, :].squeeze()
                            im_mag_scaled = zpad_merge_scale(ksp_block_multicoil, pad_ratio)

                            ################### go back to k-space #####################

                            # kspace_padded = sp.fft(im_mag_scaled, axes=(1, 2))
                            kspace_padded = sp.fft(im_mag_scaled)

                            # add zero-pading (with only a few pixels) such that the image size will be s multiplication of 8, because DL uses 8x8 blocks.
                            NX_padded = kspace_padded.shape[0]
                            NY_padded = kspace_padded.shape[1]

                            if np.mod(NY_padded,8)!=0:  # this is currently implemented only for the y axis, it can be implemented similary for the x axis.
                                print('inside')
                                NY_padded_new = int(8*np.ceil(NY_padded/8))
                                diff = NY_padded_new - NY_padded
                                diff_half = int(diff/2)
                                kspace_padded_new = np.zeros((NX_padded,NY_padded_new),dtype=kspace.dtype)
                                kspace_padded_new[:,diff_half:(diff_half+NY_padded)] = kspace_padded
                                kspace_padded = kspace_padded_new
                                NX_padded = kspace_padded.shape[0]
                                NY_padded = kspace_padded.shape[1]

                            assert np.mod(NX_padded, 8) == 0, 'NX_padded={} is not a multiplication of 8 so DictLearn cannot run with block size 8x8'.format(NX_padded)
                            assert np.mod(NY_padded,8)==0, 'NY_padded={} is not a multiplication of 8 so DictLearn cannot run with block size 8x8'.format(NY_padded)


                            # normalize kspace
                            kspace_padded = kspace_padded / np.max(np.abs(kspace_padded))

                            # ------- run recon experiment -----------------
                            # gold standard recon (fully-sampled data, with the current zero padding length)
                            print('Gold standard rec from fully sampled data...')

                            rec_gold = sp.ifft(kspace_padded)
                            cmax = np.max(np.abs(rec_gold))


                            #
                            # # display
                            # fig = plt.figure()
                            # plt.imshow(np.abs(np.rot90(rec_gold, 2)), cmap="gray")
                            # plt.title('rec_gold')
                            # plt.clim(0, cmax)
                            # plt.colorbar()
                            # plt.show()

                            # check NaN values
                            assert np.isnan(
                                rec_gold).any() == False, 'there are NaN values in rec_gold! scan {} slice {}'.format(n, s_i)

                            gold_dict[
                                i, ns - 1] = rec_gold  # store the results in a dictionary (note: we use a dictionary instead of a numpy array beause

                            imSize = kspace_padded.shape

                            # for r in range(R_vec.shape[0]):
                            #
                            #     R = R_vec[r]
                            #     print('R=', R)

                            # mkdir for figures
                            if simulation_flag == 1:
                                figs_dir = logdir + 'test_figs_pathology_1_R{}'.format(R)
                            elif simulation_flag == 2:
                                figs_dir = logdir + 'test_figs_pathology_2_R{}'.format(R)
                            else:
                                figs_dir = logdir + 'test_figs_R{}'.format(R)
                            if not os.path.exists(figs_dir):
                                os.makedirs(figs_dir)

                            for j in range(sampling_type_vec.shape[0]):

                                if sampling_type_vec[j] == 0:
                                    var_dens_flag = 'random'
                                elif sampling_type_vec[j] == 1:
                                    var_dens_flag = 'weak'
                                elif sampling_type_vec[j] == 2:
                                    var_dens_flag = 'strong'

                                print('--------- {} sampling -------'.format(var_dens_flag))

                                # ========== sampling mask =========
                                calib_x = int(12 * (NX / 640) * pad_ratio)
                                calib_y = int(12 * (NY / 640) * pad_ratio)

                                calib = np.array([calib_x, calib_y])

                                mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, var_dens_flag, calib=calib)

                                # # dispay
                                # fig = plt.figure()
                                # plt.imshow(mask,cmap="gray")
                                # plt.axis('off')
                                # plt.show()
                                # mask_figname = 'mask_R{}_{}'.format(R,var_dens_flag)
                                # fig.savefig(mask_figname)

                                # mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils
                                # ksp_padded_sampled = np.multiply(ksp2, mask_expanded)

                                ksp_padded_sampled = np.multiply(kspace_padded, mask)

                                # # display
                                # fig = plt.figure()
                                # plt.imshow(np.log(np.abs(ksp_padded_sampled)), cmap="gray")
                                # plt.axis('off')
                                # plt.show()
                                # ksp_figname = 'kspace_pad_x{}'.format(pad_ratio)
                                # fig.savefig(ksp_figname)


                                # # sanity check
                                # fig = plt.figure()
                                # plt.subplot(1,3,1)
                                # plt.imshow(np.log(np.abs(kspace_padded)),cmap="gray")
                                # plt.subplot(1,3,2)
                                # plt.imshow(np.log(np.abs(ksp_padded_sampled)),cmap="gray")
                                # plt.subplot(1, 3, 3)
                                # plt.imshow(mask, cmap="gray")
                                # plt.show()

                                # # ###################################### CS rec  ################################################

                                mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils
                                ksp_padded_sampled_expanded = np.expand_dims(ksp_padded_sampled, axis=0)
                                virtual_sens_maps = np.ones_like(
                                    ksp_padded_sampled_expanded)  # sens maps are all ones because we have a "single-coil" magnitude image.

                                # CS recon from sampled data
                                print('CS rec from sub-sampled data...')
                                rec_CS = mr.app.L1WaveletRecon(ksp_padded_sampled_expanded, virtual_sens_maps, lamda=lamda,
                                                               show_pbar=False).run()

                                CS_Err = error_metrics(rec_gold, rec_CS)
                                CS_Err.calc_NRMSE()
                                CS_Err.calc_SSIM()

                                if n == 0:
                                    fig = plt.figure()
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(np.rot90(np.abs(rec_gold), 2), cmap="gray")
                                    plt.title('"gold standard" (preprocessed)')
                                    plt.colorbar(shrink=0.5)
                                    #plt.clim(0, cmax)

                                    plt.subplot(1, 2, 2)
                                    plt.imshow(np.rot90(np.abs(rec_CS), 2), cmap="gray")
                                    plt.title('CS rec - NRMSE={:0.2}'.format(CS_Err.NRMSE))
                                    plt.colorbar(shrink=0.5)
                                    #plt.clim(0, cmax)
                                    plt.suptitle('CS Experiment - x{} zero-padding, R={}, {} var dens '.format(pad_ratio,R, var_dens_flag))
                                    plt.show()



                                    fig_name = figs_dir + '/CS_recs_slice{}_pad_{}_samp_{}'.format(ns - 1,
                                                                                                  pad_str,
                                                                                                  var_dens_flag)
                                    print('fig_name=',fig_name)
                                    fig.savefig(fig_name)


                                print('CS_NRMSE_arr.shape=',CS_NRMSE_arr.shape)
                                print('ns={}, i={}, r={}, j={}'.format(ns,i,r,j))
                                CS_NRMSE_arr[ns - 1, i, r, j] = CS_Err.NRMSE
                                CS_SSIM_arr[ns - 1, i, r, j] = CS_Err.SSIM

                                ################################## Dictionary Learning rec #####################################3
                                _maps = None

                                # # sanity checks
                                # fig = plt.figure()
                                # plt.imshow(np.abs(img_ref),cmap="gray")
                                # plt.title('img_ref')
                                # plt.show()

                                if DictL_flag == 1:
                                    with device:
                                        app = DictionaryLearningMRI(ksp_padded_sampled,
                                                                    mask,
                                                                    _maps,
                                                                    num_filters,
                                                                    batch_size,
                                                                    block_shape,
                                                                    block_strides,
                                                                    num_nonzero_coeffs,
                                                                    nu=nu,
                                                                    A_mode='omp',
                                                                    # A_mode='l1',
                                                                    D_mode='ksvd',
                                                                    D_init_mode='data',
                                                                    DC=True,
                                                                    max_inner_iter=20,
                                                                    # max_inner_iter=10,
                                                                    max_iter=max_iter,
                                                                    img_ref=None,
                                                                    device=device)
                                        out = app.run()

                                    rec_DictLearn = app.img_out

                                    DL_Err = error_metrics(rec_gold, rec_DictLearn)
                                    DL_Err.calc_NRMSE()
                                    DL_Err.calc_SSIM()

                                    Dict_NRMSE_arr[ns - 1, i, r, j] = DL_Err.NRMSE
                                    Dict_SSIM_arr[ns - 1, i, r, j] = DL_Err.SSIM

                                    # # ================== display results - DL =================
                                    # if (pad_ratio==1) & (sampling_type_vec[j]==0):  # display only for the case without padding & sampling is 'weak variable densit'
                                    if n == 0:
                                        fig = plt.figure()
                                        plt.subplot(1, 3, 1)
                                        plt.imshow(np.rot90(np.abs(rec_gold), 2), cmap="gray")
                                        plt.title('"gold standard" (preprocessed)')
                                        plt.colorbar(shrink=0.5)
                                        plt.clim(0, cmax)

                                        plt.subplot(1, 3, 2)
                                        plt.imshow(np.rot90(np.abs(rec_CS), 2), cmap="gray")
                                        CS_str = 'NRMSE={:.3f}'.format(CS_Err.NRMSE)
                                        plt.text(0.05 * imSize[1], 0.99 * imSize[0], CS_str, color="yellow")
                                        plt.title("CS")
                                        plt.colorbar(shrink=0.5)
                                        plt.clim(0, cmax)

                                        plt.subplot(1, 3, 3)
                                        plt.imshow(np.rot90(np.abs(rec_DictLearn), 2), cmap="gray")
                                        Dict_str = 'NRMSE={:.3f}'.format(DL_Err.NRMSE)
                                        plt.text(0.05 * imSize[1], 0.99 * imSize[0], Dict_str, color="yellow")
                                        plt.title('Dictionary Learning - NRMSE={}'.format(Dict_str))
                                        plt.suptitle('DictL experiment - x{} zero padding, R {}, {} var dens'.format(pad_ratio,R, var_dens_flag))
                                        plt.colorbar(shrink=0.5)
                                        plt.clim(0, cmax)
                                        plt.show()
                                        fig_name = figs_dir + '/CS_DL_recs_slice{}_pad_{}_samp_{}'.format(ns - 1, pad_str,
                                                                                                          var_dens_flag)
                                        fig.savefig(fig_name)

                                        ####### save with PIL ##########
                                        # rescale and save each image separately
                                        rec_gold_mag = np.rot90(np.abs(rec_gold), 2)
                                        rec_CS_mag = np.rot90(np.abs(rec_CS), 2)
                                        rec_DL_mag = np.rot90(np.abs(rec_DictLearn), 2)

                                        cmin_all = np.min([rec_gold_mag.min(), rec_CS_mag.max(),
                                                           rec_DL_mag.min()])  # cmax was already defined above
                                        cmax_all = np.max([rec_gold_mag.max(), rec_CS_mag.max(),
                                                           rec_DL_mag.max()])  # cmax was already defined above

                                        rec_gold_rescaled = (255.0 / cmax_all * (rec_gold_mag - cmin_all)).astype(np.uint8)
                                        rec_CS_rescaled = (255.0 / cmax_all * (rec_CS_mag - cmin_all)).astype(np.uint8)
                                        rec_DL_rescaled = (255.0 / cmax_all * (rec_DL_mag - cmin_all)).astype(np.uint8)

                                        rec_gold_PILLOW = Image.fromarray(rec_gold_rescaled)
                                        rec_CS_PILLOW = Image.fromarray(rec_CS_rescaled)
                                        rec_DL_PILLOW = Image.fromarray(rec_DL_rescaled)

                                        # save
                                        rec_gold_PILLOW.save(
                                            figs_dir + '/PIL_gold_zoomed.png')
                                        rec_CS_PILLOW.save(
                                            figs_dir + '/PIL_CS_pad{}_{}_NRMSE_{:.3f}.png'.format(pad_str,
                                                                                                         var_dens_flag,
                                                                                                         CS_Err.NRMSE))
                                        rec_DL_PILLOW.save(
                                            figs_dir + '/PIL_DL_pad{}_{}_NRMSE_{:.3f}.png'.format(pad_str,
                                                                                                         var_dens_flag,DL_Err.NRMSE))

                                        if simulation_flag<=2:
                                            # find coordinates to zoom-in on the pathology
                                            s1 = int(xf1 * rec_gold_mag.shape[0])
                                            s2 = int(xf2 * rec_gold_mag.shape[1])
                                            s3 = int(yf1 * rec_gold_mag.shape[0])
                                            s4 = int(yf2 * rec_gold_mag.shape[1])

                                            rec_gold_rescaled_zoomed = rec_gold_rescaled[s1:s3, s2:s4]
                                            rec_CS_rescaled_zoomed = rec_CS_rescaled[s1:s3, s2:s4]
                                            rec_DL_rescaled_zoomed = rec_DL_rescaled[s1:s3, s2:s4]

                                            rec_gold_PILLOW_zoomed = Image.fromarray(rec_gold_rescaled_zoomed)
                                            rec_CS_PILLOW_zoomed = Image.fromarray(rec_CS_rescaled_zoomed)
                                            rec_DL_PILLOW_zoomed = Image.fromarray(rec_DL_rescaled_zoomed)

                                            # save
                                            rec_gold_PILLOW_zoomed.save(
                                                figs_dir + '/PIL_gold_zoomed.png')
                                            rec_CS_PILLOW_zoomed.save(
                                                figs_dir + '/PIL_CS_pad{}_{}_NRMSE_{:.3f}_zoomed.png'.format(pad_str,
                                                                                                             var_dens_flag,
                                                                                                             CS_Err.NRMSE))
                                            rec_DL_PILLOW_zoomed.save(
                                                figs_dir + '/PIL_DL_pad{}_{}_NRMSE_{:.3f}_zoomed.png'.format(pad_str,
                                                                                                             var_dens_flag,
                                                                                                             DL_Err.NRMSE))

                                elif DictL_flag == 0:
                                    ####### save with PIL ##########
                                    # rescale and save each image separately
                                    rec_gold_mag = np.rot90(np.abs(rec_gold), 2)
                                    rec_CS_mag = np.rot90(np.abs(rec_CS), 2)


                                    cmin_all = np.min([rec_gold_mag.min(), rec_CS_mag.max(),
                                                       ])  # cmax was already defined above
                                    cmax_all = np.max([rec_gold_mag.max(), rec_CS_mag.max(),
                                                       ])  # cmax was already defined above

                                    rec_gold_rescaled = (255.0 / cmax_all * (rec_gold_mag - cmin_all)).astype(np.uint8)
                                    rec_CS_rescaled = (255.0 / cmax_all * (rec_CS_mag - cmin_all)).astype(np.uint8)

                                    rec_gold_PILLOW = Image.fromarray(rec_gold_rescaled)
                                    rec_CS_PILLOW = Image.fromarray(rec_CS_rescaled)

                                    # save
                                    rec_gold_PILLOW.save(
                                        figs_dir + '/PIL_gold.png')
                                    rec_CS_PILLOW.save(
                                        figs_dir + '/PIL_CS_pad{}_{}_NRMSE_{:.3f}.png'.format(pad_str, var_dens_flag,
                                                                                                     CS_Err.NRMSE))

                                    if simulation_flag <= 2:
                                   # zoom-in on pathology

                                       s1 = int(xf1 * rec_gold_mag.shape[0])
                                       s2 = int(xf2 * rec_gold_mag.shape[1])
                                       s3 = int(yf1 * rec_gold_mag.shape[0])
                                       s4 = int(yf2 * rec_gold_mag.shape[1])

                                       rec_gold_rescaled_zoomed = rec_gold_rescaled[s1:s3, s2:s4]
                                       rec_CS_rescaled_zoomed = rec_CS_rescaled[s1:s3, s2:s4]

                                       rec_gold_PILLOW_zoomed = Image.fromarray(rec_gold_rescaled_zoomed)
                                       rec_CS_PILLOW_zoomed = Image.fromarray(rec_CS_rescaled_zoomed)

                                       # save
                                       rec_gold_PILLOW_zoomed.save(
                                           figs_dir + '/PIL_gold_zoomed.png')
                                       rec_CS_PILLOW_zoomed.save(
                                           figs_dir + '/PIL_CS_pad{}_{}_NRMSE_{:.3f}_zoomed.png'.format(pad_str,var_dens_flag,CS_Err.NRMSE))


            n += 1
            if ns==N_examples:
                break


        # --------------------- save results ----------------------
        # save results
        print('saving results for current R')
        data_filename = 'zpad_CS_DictL_results_R{}'.format(R)
        np.savez(data_filename,CS_NRMSE_arr=CS_NRMSE_arr,CS_SSIM_arr=CS_SSIM_arr,Dict_NRMSE_arr=Dict_NRMSE_arr,Dict_SSIM_arr=Dict_SSIM_arr,R_vec=R_vec,pad_ratio_vec=pad_ratio_vec,sampling_type_vec=sampling_type_vec,sampling_flag=sampling_flag,DictL_flag=DictL_flag)

        # the dictionaries cannot be saved with np.savez, but they can be saved separately, e.g.:
        # np.save('gold_dict.npy',gold_dict)
        # np.save('CS_recs_dict.npy',CS_recs_dict)
        # np.save('masks_dict.npy',masks_dict)
        # np.save('Dict_recs_dict.npy',Dict_recs_dict)


