# This code demonstrates subtle crime I with Compressed sensing
# Run the *fast* experiment to see the images & NRMSE valus on top of them.
# Run the *long* experiment (10 slices x 3 samlpling mask realizations each) to get statistics.
# Then run the script CS_DL_knee_prep_NRMSE_figure.py to produce the statistics graphs

#############################################################
# calibration run - this code (1_knee_calib_CS_lada.py)


##########################################################################################
import os
import numpy as np
import h5py
import sys
# add path to functions library - when running on mikQNAP
sys.path.append("/mikQNAP/efrat/1_inverse_crimes/1_mirror_PyCharm_CS_MoDL_merged/SubtleCrimesRepo/")

import matplotlib.pyplot as plt
from functions.utils import merge_multicoil_data
import sigpy as sp
from sigpy import mri as mr
#from matplotlib import interactive
#from functions.demos_funcs import demo1_zero_pad_MAG_run_exps
from functions.utils import pad_multicoil_ksp,save_as_png
from functions.error_funcs import error_metrics
#from functions.sampling_funcs import genPDF, genSampling
from functions.sampling_funcs import gen_2D_var_dens_mask



sys.path.append("/home/efrat/anaconda3/")
sys.path.append("/home/efrat/anaconda3/lib/python3.7/site-packages/")  # path to sigpy

#################################################################################
## Experiment set-up
#################################################################################

R = 4
pad_ratio_vec = np.array([1,2])


sampling_type_vec = np.array([1,2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens
sampling_flag = '2D'

num_slices = 1

im_type_str = 'full_im'  # Options: 'full_im' / 'blocks' (blocks are used for training Deep Learning models, not for CS & DictL).

data_type = 'pathology_1'
#data_type = 'pathology_2'

if data_type=='pathology_1':
    pathology_slice = 22

lamda = 1e-3

# figs_folder = 'pathology_1_NEW_FIGS'
# if not os.path.exists(figs_folder):
#     os.makedirs(figs_folder)

gold_dict = {}  # a python dictionary that will contain the gold standard recons
CS_recs_dict = {}  # a python dictionary that will contain the reconstructions obtained with Compressed Sensing

# #################################################################################
# ##                               Experiments
# #################################################################################

for pad_i, pad_ratio in enumerate(pad_ratio_vec):
    print(f'##################### pad ratio {pad_ratio} ################################')

    t = 0 # counts loaded scans. each scan contains multiple slices.
    ns = 0 # counts loaded slices

    if (pad_ratio==1) | (pad_ratio==2):
        pad_ratio_str = int(pad_ratio)

    # # update the next field and make sure that it's the same one as defined in Fig4_pathology_example/data_prep.py
    FatSat_processed_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_FatSat_data/"

    data_path = FatSat_processed_data_folder + data_type + "/pad_" + str(
        int(100 * pad_ratio)) + "/" + im_type_str + "/"

    files_list = os.listdir(data_path)


    while ns<num_slices:

        print(' === loading h5 file {} === '.format(t))
        # Load k-space data
        filename_h5 = data_path + files_list[t]

        #print('t=', t)
        #print('filename_h5=', filename_h5)
        f = h5py.File(filename_h5, 'r')

        t += 1  # update the number of LOADED scans. Each scan contains multiple slices

        kspace_preprocessed_multislice = f["kspace"]
        im_RSS_multislice = f["reconstruction"]  # these are the RSS images produced from the zero-padded k-space - see fig. 1 in the paper

        n_slices_in_scan = kspace_preprocessed_multislice.shape[0]


        print(f'pad_ratio {pad_ratio}  t={t}')

        for s_i in range(n_slices_in_scan):

            if s_i==pathology_slice:
                print(f'slice {s_i}')

                kspace_slice = kspace_preprocessed_multislice[s_i,:,:].squeeze()
                im_RSS = im_RSS_multislice[s_i,:,:].squeeze()

                ns += 1  # number of slices
                print(f'ns={ns}')

                imSize = im_RSS.shape


                kspace_slice = np.expand_dims(kspace_slice, axis=0) # restore coil dimension (for Sigpy data format)
                _ , NX_padded, NY_padded = kspace_slice.shape  # get size. Notice: the first one is the coils dimension

                virtual_sens_maps = np.ones_like(kspace_slice)  # sens maps are all ones because we have a "single-coil" magnitude image.

                # ------- gold standard rec -----------------

                rec_gold = sp.ifft(kspace_slice)
                rec_gold = rec_gold[0,:,:].squeeze() # remove artificial coil dim
                rec_gold_rotated = np.abs(np.rot90(rec_gold, 2))


                # fig = plt.figure()
                # plt.imshow(np.rot90(np.abs(rec_gold),2), cmap="gray")
                # plt.title('rec_gold')
                # plt.colorbar()
                # plt.show()


                # check NaN values
                assert np.isnan(rec_gold).any() == False, 'there are NaN values in rec_gold! scan {} slice {}'.format(n,s_i)

                img_shape = np.array([NX_padded,NY_padded])

                # ----- Compressed Sensing recon ----------

                for j in range(sampling_type_vec.shape[0]):

                    if sampling_type_vec[j] == 0:  # random uniform
                        samp_type = 'random'
                    elif sampling_type_vec[j] == 1:  # weak variable-density
                        samp_type = 'weak'
                    elif sampling_type_vec[j] == 2: # strong variable-density
                        samp_type = 'strong'

                    data_filename = f'{data_type}_R{R}_{samp_type}_VD'

                    # calib is assumed to be 12 for NX=640
                    calib_x = int(12 * im_RSS.shape[0] / 640)
                    calib_y = int(12 * im_RSS.shape[1] / 640)
                    calib = np.array([calib_x, calib_y])


                    mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, samp_type, calib=calib)

                    mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils dimension, for compatibility with Sigpy's dimension convention
                    kspace_sampled = np.multiply(kspace_slice, mask_expanded)

                    rec = mr.app.L1WaveletRecon(kspace_sampled, virtual_sens_maps, lamda=lamda, show_pbar=False).run()
                    rec_CS_rotated = np.abs(np.rot90(rec, 2))

                    gold_dict[pad_ratio, samp_type] = rec_gold_rotated
                    CS_recs_dict[pad_ratio, samp_type] = rec_CS_rotated


                    # # --------- TODO: move this to the display code -----------
                    # A = error_metrics(rec_gold, rec)
                    # A.calc_NRMSE()
                    # A.calc_SSIM()
                    #
                    # #print(f'CS rec; NRMSE={A.NRMSE:.4f}')
                    #
                    # cmax = np.max([np.abs(rec_gold),np.abs(rec)])
                    #
                    #
                    # fig = plt.figure()
                    # plt.subplot(1,2,1)
                    # plt.imshow(np.abs(np.rot90(rec_gold,2)), cmap="gray")
                    # plt.title('rec_gold')
                    # plt.clim(0,cmax)
                    # plt.colorbar(shrink=0.25)
                    #
                    # plt.subplot(1,2,2)
                    # plt.imshow(np.abs(np.rot90(rec,2)),cmap="gray")
                    # plt.title(f'CS NRMSE {A.NRMSE:.3f}')
                    # plt.clim(0, cmax)
                    # plt.colorbar(shrink=0.25)
                    # plt.suptitle(f'{data_type} data; R={R}; pad_ratio={pad_ratio}; {samp_type} VD samp; scan {t}; slice {ns}')
                    # plt.show()
                    # figname = figs_folder + f'/slice{ns}_pad_{pad_ratio}_{samp_type}.png'
                    # fig.savefig(figname)
                    #
                    # fig = plt.figure()
                    #
                    #
                    #
                    #
                    # # figures for the paper
                    # # zoom-in coordinates for pathology 1
                    # x1 = 335
                    # x2 = 380
                    # y1 = 210
                    # y2 = 300
                    # # scale the zoom-in coordinates to fit changing image size
                    # x1s = int(335 * pad_ratio)
                    # x2s = int(380 * pad_ratio)
                    # y1s = int(210 * pad_ratio)
                    # y2s = int(300 * pad_ratio)
                    #
                    # cmax = np.max(np.abs(rec))

                    # # gold standard zoomed - png figure
                    # fig = plt.figure()
                    # plt.imshow(rec_gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    # plt.axis('off')
                    # plt.clim(0, cmax)
                    # plt.show()
                    # figname = figs_folder + f'/rec_gold_pad_x{pad_ratio_str}_zoomed.png'
                    # fig.savefig(figname, dpi=1000)
                    #
                    # # gold standard zoomed - eps figure
                    # fig = plt.figure()
                    # plt.imshow(rec_gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    # plt.axis('off')
                    # plt.clim(0, cmax)
                    # plt.show()
                    # figname = figs_folder + f'/rec_gold_pad_x{pad_ratio_str}_zoomed.eps'
                    # fig.savefig(figname, format='eps', dpi=1000)

                    # # rec CS zoomed - png figure
                    # fig = plt.figure()
                    # plt.imshow(rec_CS_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    # plt.axis('off')
                    # plt.clim(0, cmax)
                    # plt.show()
                    # figname = figs_folder + f'/CS_rec_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed'
                    # fig.savefig(figname, dpi=1000)
                    #
                    # # rec CS zoomed - eps figure
                    # fig = plt.figure()
                    # plt.imshow(rec_CS_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    # plt.axis('off')
                    # plt.clim(0, cmax)
                    # plt.show()
                    # figname = figs_folder + f'/CS_rec_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed.eps'
                    # fig.savefig(figname, format='eps', dpi=1000)
                    #
                    #
                    # if pad_ratio==2:
                    #     # gold standard full-size .eps figure
                    #     fig = plt.figure()
                    #     plt.imshow(rec_gold_rotated, cmap="gray")
                    #     plt.axis('off')
                    #     plt.clim(0, cmax)
                    #     plt.show()
                    #     figname = figs_folder + f'/rec_gold_full_size.eps'
                    #     fig.savefig(figname, format='eps', dpi=1000)
                    #
                    #     # gold standard full-size .png figure
                    #     fig = plt.figure()
                    #     plt.imshow(rec_gold_rotated, cmap="gray")
                    #     plt.axis('off')
                    #     plt.clim(0, cmax)
                    #     plt.show()
                    #     figname = figs_folder + f'/rec_gold_full_size.eps'
                    #     fig.savefig(figname, dpi=1000)



# --------------------- save ----------------------
# save results
#np.savez(data_filename,NRMSE_arr=NRMSE_arr,masks_dict=masks_dict,R_vec=R_vec,pad_ratio_vec=pad_ratio_vec,
#         sampling_type_vec=sampling_type_vec,sampling_flag=sampling_flag,lamda_vec=lamda_vec,num_slices=num_slices)

# save the recons
results_dir =  data_type + f'_results_R{R}/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

gold_filename = results_dir + '/gold_dict.npy'
np.save(gold_filename , gold_dict)
CS_rec_filename = results_dir + '/CS_dict.npy'
np.save(CS_rec_filename, CS_recs_dict)