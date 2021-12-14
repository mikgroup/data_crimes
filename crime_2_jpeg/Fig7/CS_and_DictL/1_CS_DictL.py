'''
This code demonstrates the scenario of Subtle Data Crime II for CS and DictL reconstruction algorithms.

Here we use the hyperparameters values that were found in the hyperparam search performed for subtle data crime I,
for zero-padding with factor x1 (which means that there was no zero padding - because the JPEG images were not zero-padded
 in k-space), and a "weak" VD sampling scheme (which is implemented here). These parameters are:
CS: lamda=1e-3
DictL: block_shape=16, num_filters=300, nnz=11, max_iter=13, nu_lamda=1e-05 (this is the DictL lamada, not CS lamda)

Note: these parameters were found when the algorithms were implemented on *magnitude* images (due to the subtle
crimes scenarios), hence if your research involves *complex* MR images, a new hyperparameter search should be performed.

Before running this script you should update the following:
  project_full_path
  basic_data_folder - it should be the same as the output folder defined in the script /crime_2_jpeg/data_prep/jpeg_data_prep.py

Example - how to run this code from the linux command line:
$ python3 1_CS_DictL.py --sim_flag 0 --R 2

# (c) Efrat Shimron, UC Berkeley, 2021
'''


import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import mkl
import sigpy as sp
import sigpy.plot as pl
import sys
import h5py
import os
from sigpy import mri as mr

# add path to functions library
project_full_path = "/mikQNAP/efrat/1_inverse_crimes/1_mirror_PyCharm_CS_MoDL_merged/SubtleCrimesRepo/"
sys.path.append(project_full_path)
sys.path.insert(1, './mri-sim-py/epg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functions.dict_learn_funcs import DictionaryLearningMRI
from functions.sampling_funcs import gen_2D_var_dens_mask
from functions.error_funcs import error_metrics

num_CPUs = 75
mkl.set_num_threads(num_CPUs)  # the number in the brackets determines the number of CPUs. 1 is recommended for the DictL algorithm! Otherwise there's a lot of overhead (when the run is spread accross multiple cpus) and the comptuation time becomes longer.


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_args():
    parser = OptionParser()
    parser.add_option('--simulation_flag', '--sim_flag', type='int', default=0,
                      help='0 statistics; 1=pathology #1; 2=pathology 2')
    parser.add_option('--R', '--R', type='int', default=[3], help='desired R')
    parser.add_option('--q_vec', '--q_vec', type='int', default=[20, 50, 75, 999],nargs='+',
                      help='q vals for JPEG compression')#default=[20, 50, 75, 100, 999]
    parser.add_option('--nnz', '--num_nonzero_coeffs', type='int', default=11,
                      help='num_nonzero_coeffs controls the sparsity level when  Dictionary Learning runs with A_mode=''omp'' ')
    parser.add_option('--num_filters', '--num_filters', type='int', default=300, help='num_filters for Dict Learning')
    parser.add_option('--max_iter', '--max_iter', type='int', default=13, help='number of iterations')
    parser.add_option('--batch_size', '--batch_size', type='int', default=500, help='batch_size')
    parser.add_option('--block_shape', '--block_shape', type='int', default=[16, 16], help='block_shape')
    parser.add_option('--block_strides', '--block_strides', type='int', default=[4, 4], help='block_strides')
    parser.add_option('--nu', '--nu', type='int', default=1e-05, help='nu for Dict Learning')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    print(args)

    # input variables for the experiments. q = JPEG quality factor (determines the compression level)
    R = np.asfarray(args.R).astype('int')  # Reducation factor (k-space subsampling rate)
    print('R =', R)
    print(f'type(R) {type(R)}')
    print(f'R.shape {R.shape}')

    q_vec = np.asfarray(args.q_vec).astype('int')

    # input variables for Dictionary Learning
    num_nonzero_coeffs = args.nnz
    max_iter = args.max_iter
    num_filters = args.num_filters
    batch_size = args.batch_size
    block_shape = args.block_shape
    block_strides = args.block_strides
    nu = args.nu  # nu = lambda*2 (it is a paramter the controls the tradeoff between Data Consistency and sparsity terms)

    # hard-coded options for Dictionary Learning
    device = sp.cpu_device  # which device to use (not all is supported on GPU)
    mode = 'omp'

    var_dens_flag = 'weak' # can be 'strong' / 'weak'

    # Compressed Sensing parameter
    lamda = 1e-3

    # simulation_flag = 1
    simulation_flag = args.simulation_flag
    DictL_flag = 1
    print('DictL_flag=', DictL_flag)

    if simulation_flag == 0: # statistics
        N_examples = 122  # desired number of slices (for statistics)
        print('Running statistics on {} slices'.format(N_examples))
        data_type = 'test'
        # image dimensions
        NX = 640
        NY = 372
    elif simulation_flag == 1:
        print('running PATHOLOGY CASE #1')
        N_examples = 1  # desired number of slices (for statistics)
        data_type = 'pathology_1'

    elif simulation_flag == 2:
        print('running PATHOLOGY CASE #2')
        N_examples = 1  # desired number of slices (for statistics)
        data_type = 'pathology_2'

    if (simulation_flag == 1) | (simulation_flag == 2):
        # coordinates for zoom-in on the pathology area
        s1 = int(a * NX)
        s2 = int(b * NY)
        s3 = int(c * NX)
        s4 = int(d * NY)

    print('N_examples wanted= ', N_examples)

    imSize = np.array([NX, NY])

    # calib is the fully-sampled calibration area in k-space center. Here it is assumed to be 12 for NX=640
    calib_x = int(12)
    calib_y = int(12*NY/NX)
    calib = np.array([calib_x, calib_y])

    N_recs_to_store = 25


    if (simulation_flag == 1) | (simulation_flag == 2):
        run_foldername = f'R{int(R)}_{var_dens_flag}_VD_{data_type}'
    else:
        run_foldername = f'R{int(R)}_{var_dens_flag}_VD_{N_examples}_{data_type}_imgs'

    # create folder:
    if not os.path.exists(run_foldername):
        print('creating folder {}'.format(run_foldername))
        os.makedirs(run_foldername)


    # ---------- initialize arrays --------
    CS_NRMSE_vs_quality = np.empty([q_vec.shape[0], N_examples])
    CS_SSIM_vs_quality = np.empty([q_vec.shape[0], N_examples])
    DictL_NRMSE_vs_quality = np.empty([q_vec.shape[0], N_examples])
    DictL_SSIM_vs_quality = np.empty([q_vec.shape[0], N_examples])


    gold_recs_array = np.empty([q_vec.shape[0], N_recs_to_store, NX, NY],
                               dtype='complex')
    CS_recs_array = np.empty([q_vec.shape[0], N_recs_to_store, NX, NY],
                             dtype='complex')
    DictL_recs_array = np.empty([q_vec.shape[0], N_recs_to_store, NX, NY],
                                dtype='complex')

    ###########################################################################################
    #                                 Recon Experiments
    ###########################################################################################

    for qi in range(q_vec.shape[0]):
        q = q_vec[qi]
        print('========= q={} ======== '.format(q))

        # Notice: update the next folder to YOUR data folder. It should be the same as the one defined in the script crime_2_jpeg/data_prep/jpeg_data_prep.py
        basic_data_folder = "/mikQNAP/NYU_knee_data/multicoil_efrat/5_JPEG_compressed_data/"
        data_folder = basic_data_folder + data_type + "/q" + str(q) + "/full_im/"
        files_list = os.listdir(data_folder)

        n_init = 0  # change this if you want to load scans that do not begin from the first one, e.g. scans 4,...7

        n = n_init # counts number of loaded scan. Each scan contains ~20 slices
        ns = 0  # counts loaded examples (slices, not scans!)

        while ns < N_examples:
                kspace_loaded_flag = 0

                # #################################################################################
                # ##                        Load Data - kspace
                # #################################################################################
                filename_h5 = data_folder + files_list[n]
                f = h5py.File(filename_h5, 'r')

                print(f'loaded file {filename_h5}')

                kspace_all_slices = np.array(f["kspace"])
                img_jpeg_all_slices = np.array(f["img_jpeg"])

                print('kspace_all_slices.shape: ', kspace_all_slices.shape)
                N_slices = kspace_all_slices.shape[0]

                n=+1 #

                #################### Lopp over slices (using data of a single scan) ################
                for s_i in range(N_slices):

                    if ns < N_examples:

                        ns += 1

                        print(f'slice ns={ns}')

                        kspace_slice = kspace_all_slices[s_i, :, :].squeeze()
                        rec_gold = img_jpeg_all_slices[s_i, :, :].squeeze()

                        # -------- subsample kspace ------------
                        mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, var_dens_flag, calib=calib)

                        ksp_sampled = mask * kspace_slice  # in this example we use only the first coil

                        fig = plt.figure()
                        plt.imshow(np.log(np.abs(kspace_slice)),cmap="gray")
                        plt.show()

                        #################################### Compressed Sensing ##################################
                        # we add an empty coils dimension for compatibility with Sigpy's dimensions convention
                        mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils
                        ksp_padded_sampled_expanded = np.expand_dims(ksp_sampled, axis=0)
                        virtual_sens_maps = np.ones_like(
                            ksp_padded_sampled_expanded)  # sens maps are all ones because we have a "single-coil" magnitude image.

                        # CS recon from sampled data
                        print('CS rec from sub-sampled data...')
                        rec_CS = mr.app.L1WaveletRecon(ksp_padded_sampled_expanded, virtual_sens_maps, lamda=lamda,
                                                       show_pbar=False).run()

                        # fig = plt.figure()
                        # plt.imshow(np.rot90(np.abs(rec_CS),2), cmap="gray")
                        # plt.title('CS rec')
                        # plt.show()

                        if ns<=5:
                            fig = plt.figure()
                            plt.subplot(1,2,1)
                            plt.imshow(np.rot90(np.abs(rec_gold),2),cmap="gray")
                            plt.colorbar()
                            plt.axis('off')
                            plt.title('img_jpeg (ref)')

                            plt.subplot(1,2,2)
                            plt.imshow(np.rot90(np.abs(rec_CS),2),cmap="gray")
                            plt.colorbar()
                            plt.axis('off')
                            plt.title('CS')
                            plt.suptitle('R={} q={} ns={}'.format(R,q,ns))
                            plt.show()


                        CS_Err = error_metrics(rec_gold, rec_CS)
                        CS_Err.calc_NRMSE()
                        CS_Err.calc_SSIM()

                        print('qi={} ns={}'.format(qi, ns))
                        print(f'CS NRMSE {CS_Err.NRMSE :.4f}')
                        CS_NRMSE_vs_quality[qi, ns - 1] = CS_Err.NRMSE
                        CS_SSIM_vs_quality[qi, ns - 1] = CS_Err.SSIM

                        ######################################## Dictionary Learning MRI Recon ################################################

                        if DictL_flag == 1:
                            _maps = None

                            with device:
                                app = DictionaryLearningMRI(ksp_sampled,
                                                            mask,
                                                            _maps,
                                                            num_filters,
                                                            batch_size,
                                                            block_shape,
                                                            block_strides,
                                                            num_nonzero_coeffs,
                                                            nu=nu,
                                                            A_mode='omp',
                                                            D_mode='ksvd',
                                                            D_init_mode='data',
                                                            DC=True,
                                                            max_inner_iter=20,
                                                            max_iter=max_iter,
                                                            img_ref=None,
                                                            device=device)
                                out = app.run()

                            loss_vs_iter_vec = np.array(app.residuals)

                            D_hat_flat, A_hat_flat = sp.to_device(out[0], sp.cpu_device), sp.to_device(out[1],
                                                                                                       sp.cpu_device)
                            D_hat = D_hat_flat.reshape((*block_shape, num_filters))
                            A_hat = A_hat_flat.reshape((num_filters, *app.img_blocks_shape[:2]))

                            pl.ImagePlot(D_hat, x=-3, y=-2, z=-1, interpolation='nearest', mode='r')

                            img_hat_flat = D_hat_flat.dot(A_hat_flat)
                            img_dict = app.reshape_block_op * img_hat_flat.T
                            img_dict /= app.block_scale_factor
                            rec_DictL = app.img_out

                            # calc NRMSE & SSIM
                            DictL_err = error_metrics(np.abs(rec_gold), np.abs(rec_DictL))
                            DictL_err.calc_NRMSE()
                            DictL_err.calc_SSIM()

                            print('DictL NRMSE = ', DictL_err.NRMSE)

                            DictL_NRMSE_vs_quality[qi, ns - 1] = DictL_err.NRMSE
                            DictL_SSIM_vs_quality[qi, ns - 1] = DictL_err.SSIM
                            
                            if ns <= N_recs_to_store:
                                cmax = np.max(np.abs(rec_gold))

                                fig = plt.figure()
                                plt.subplot(1, 3, 1)
                                plt.imshow(np.rot90(np.abs(rec_gold),2), cmap="gray")
                                plt.colorbar(shrink=0.5)
                                plt.axis('off')
                                plt.title('img_jpeg (ref)')
                                plt.clim(0,cmax)

                                plt.subplot(1, 3, 2)
                                plt.imshow(np.rot90(np.abs(rec_CS),2), cmap="gray")
                                plt.colorbar(shrink=0.5)
                                plt.axis('off')
                                plt.title(f'CS rec - NRMSE {CS_Err.NRMSE:.4f}')
                                plt.clim(0, cmax)

                                plt.subplot(1, 3, 3)
                                plt.imshow(np.rot90(np.abs(rec_DictL),2), cmap="gray")
                                plt.colorbar(shrink=0.5)
                                plt.axis('off')
                                plt.title(f'DictL - NRMSE {DictL_err.NRMSE :.4f}')
                                plt.clim(0, cmax)

                                plt.suptitle(f'R={R} VD {var_dens_flag} q={q} slice {ns}')
                                plt.show()
                                fname = run_foldername + f'/rec_slice_{ns}'
                                fig.savefig(fname,dpi=1000)

                    if ns == N_examples:
                        break

                # save current results (during the run)
                if n % 3 == 0:


                    res_filename = run_foldername + f'/CS_DictLearn_results_scans_n{n_init}_to_{n}'
                    print(f'saving currest results n={n} filename: {res_filename}')

                    np.savez(res_filename,
                             CS_NRMSE_vs_quality=CS_NRMSE_vs_quality[:,0:ns-1,:,:].squeeze(),
                             CS_SSIM_vs_quality=CS_SSIM_vs_quality[:,0:ns-1,:,:].squeeze(),
                             DictL_NRMSE_vs_quality=DictL_NRMSE_vs_quality[:,0:ns-1].squeeze(),
                             DictL_SSIM_vs_quality=DictL_SSIM_vs_quality[:,0:ns-1].squeeze(),
                             q_vec=q_vec,
                             R=R,
                             N_examples=N_examples,
                             DictL_flag=DictL_flag)



print('run finished successfully')

# save final results
res_filename = run_foldername + f'/CS_DictLearn_results'

np.savez(res_filename,
         CS_NRMSE_vs_quality=CS_NRMSE_vs_quality,
         CS_SSIM_vs_quality=CS_SSIM_vs_quality,
         DictL_NRMSE_vs_quality=DictL_NRMSE_vs_quality,
         DictL_SSIM_vs_quality=DictL_SSIM_vs_quality,
         q_vec=q_vec,
         R = R,
         gold_recs_array=gold_recs_array,
         CS_recs_array=CS_recs_array,
         DictL_recs_array=DictL_recs_array,
         N_examples=N_examples,
         DictL_flag=DictL_flag)

print('data saved successfully')

# ----------- preparation for plots -----------------
x_ticks_labels = []
for i in range(q_vec.shape[0]):
   x_ticks_labels.append('{}'.format(q_vec[(-i - 1)]))

print(x_ticks_labels)

# ------------ NRMSE graph - CS results -------------
fig = plt.figure()

if N_examples > 1:
    NRMSE_av_per_quality_val = CS_NRMSE_vs_quality.mean(axis=1)
    NRMSE_std_per_quality_val = CS_NRMSE_vs_quality.std(axis=1)

    plt.errorbar(q_vec[::-1], NRMSE_av_per_quality_val[::-1], yerr=NRMSE_std_per_quality_val[::-1], fmt='-o',
                 label='R={}'.format(R))
else:
    plt.plot(q_vec[::-1], CS_NRMSE_vs_quality[::-1], '-o', label='R={}'.format(R))

ax = plt.gca()
ax.set_xticks(q_vec[::-1])
ax.set_xticklabels(x_ticks_labels, fontsize=14)
ax.invert_xaxis()
plt.ylabel('NRMSE', fontsize=20)
y_ticks_vec = np.array([0, 0.02, 0.04])
ax.set_yticks(y_ticks_vec)
ax.set_yticklabels(['0', '0.02', '0.04'], fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax = plt.gca()
ax.legend(fontsize=14, loc="upper right")
plt.title('CS results - {} examples'.format(N_examples))
plt.show()
fname = run_foldername + 'RES_CS_NRMSE_vs_q.png'
fig.savefig(fname=fname)

# ------------ NRMSE graph - DictL results -------------
if DictL_flag == 1:
    fig = plt.figure()

    if N_examples > 1:
        NRMSE_av_per_quality_val = DictL_NRMSE_vs_quality.mean(axis=1)
        NRMSE_std_per_quality_val = DictL_NRMSE_vs_quality.std(axis=1)

        plt.errorbar(q_vec[::-1], NRMSE_av_per_quality_val[::-1], yerr=NRMSE_std_per_quality_val[::-1], fmt='-o',
                     label='R={}'.format(R))
    else:
        plt.plot(q_vec[::-1], DictL_NRMSE_vs_quality[::-1], '-o', label='R={}'.format(R))

    ax = plt.gca()
    ax.set_xticks(q_vec[::-1])
    ax.set_xticklabels(x_ticks_labels, fontsize=14)
    ax.invert_xaxis()
    plt.ylabel('NRMSE', fontsize=20)
    y_ticks_vec = np.array([0, 0.02, 0.04])
    ax.set_yticks(y_ticks_vec)
    ax.set_yticklabels(['0', '0.02', '0.04'], fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()
    ax.legend(fontsize=14, loc="upper right")
    plt.title('Dictionary Learning results - {} examples'.format(N_examples))
    plt.show()
    fname = run_foldername +'RES_DictL_NRMSE_vs_q.png'
    fig.savefig(fname=fname)

