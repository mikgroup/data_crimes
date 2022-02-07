'''
This script creates the example shown in Figure 3 in the Subtle Data Crimes paper.

To run this code, please download the file named "file_brain_AXT2_207_2070504.h5" from the FastMRI database
 and place it in this folder:
'../brain_data/'

(c) Efrat Shimron, UC Berkeley, 2021.
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from sigpy import mri as mr

from subtle_data_crimes.functions.error_funcs import error_metrics
from subtle_data_crimes.functions.sampling_funcs import gen_2D_var_dens_mask
from subtle_data_crimes.functions.utils import zpad_merge_scale

###################################### settings ######################################3

R = np.array([4])

sampling_type_vec = np.array([1, 2])  # 0 = random, 1 = weak var-dens, 2 = strong var-dens
sampling_flag = '2D'

lamda = 0.005

### Experiment set-up

num_slices = 1
R_vec = np.array([6])
pad_ratio_vec = np.array([1, 2, 3])  # Define the desired padding ratios
num_realizations = 1  # number of sampling masks that will be generated for each case

#### load data
example_filename = '../brain_data/file_brain_AXT2_207_2070504.h5'
f = h5py.File(example_filename, "r")
kspace_orig_multicoil = f["kspace"]

slice_i = 0
kspace_orig_multicoil = kspace_orig_multicoil[slice_i, :, ::2, :]  # reduce FOV in image domain

#### experiments
CS_recs_dict = {}
masks_dict = {}
CS_NRMSE_arr = np.zeros((pad_ratio_vec.shape[0], sampling_type_vec.shape[0]))

for pad_i in range(pad_ratio_vec.shape[0]):
    pad_ratio = pad_ratio_vec[pad_i]
    print('======================= pad_ratio={} ====================='.format(pad_ratio))

    # zero-pad & merge, and then go to image domain:
    # new code (following data_prep_v11)
    im_mag_scaled = zpad_merge_scale(kspace_orig_multicoil, pad_ratio)
    im_mag_scaled = np.rot90(im_mag_scaled, 2)

    ## go back to k-space
    kspace_slice = sp.fft(im_mag_scaled)

    NX = kspace_slice.shape[0]
    NY = kspace_slice.shape[1]

    # ------- run recon experiment -----------------
    rec_gold = sp.ifft(kspace_slice)

    if pad_i == 0:
        rec_gold_original_size = rec_gold  # used for display only

    cmax = np.max(np.abs(rec_gold))

    imSize = im_mag_scaled.shape

    # # display
    # fig = plt.figure()
    # plt.imshow(np.abs(rec_gold), cmap="gray")
    # #plt.title('rec_gold')
    # plt.clim(0, cmax)
    # plt.axis('off')
    # #plt.colorbar()
    # #plt.title('slice i {}'.format(slice_i))
    # plt.show()
    # fig.savefig('gold_full_FOV.png')

    # zoom-in
    s1 = int(rec_gold.shape[0] / 5)
    s2 = int(rec_gold.shape[1] / 6)
    s3 = int(rec_gold.shape[0] * 1.1 / 2)
    s4 = int(rec_gold.shape[1] * 1.1 / 2)
    rec_gold_zoomed = rec_gold[s1:s3, s2:s4]

    # fig = plt.figure()
    # plt.imshow(np.abs(rec_gold_zoomed), cmap="gray")
    # #plt.title('rec_gold')
    # plt.clim(0, cmax)
    # plt.axis('off')
    # #plt.colorbar()
    # #plt.title('slice i {}'.format(slice_i))
    # plt.show()
    # fig.savefig('gold_zoomed.png')

    # check NaN values
    assert np.isnan(
        rec_gold).any() == False, 'there are NaN values in rec_gold! scan {} slice {}'.format(n, s_i)

    for j in range(sampling_type_vec.shape[0]):

        if sampling_type_vec[j] == 0:
            var_dens_flag = 'random'
        elif sampling_type_vec[j] == 1:
            var_dens_flag = 'weak'
        elif sampling_type_vec[j] == 2:
            var_dens_flag = 'strong'

        print('------------- {} sampling -----------'.format(var_dens_flag))

        # ========== sampling mask =========
        calib_x = int(12 * (NX / 640) * pad_ratio)
        calib_y = int(12 * (NY / 640) * pad_ratio)

        calib = np.array([calib_x, calib_y])

        mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, var_dens_flag, calib=calib)

        # # # ------- plot of pdf profiles - without calib ------------
        # mask4plot, pdf4plot, poly_degree = gen_2D_var_dens_mask(R, imSize, var_dens_flag, calib=np.array([1,1]))
        # y_mid = int(np.floor(mask.shape[0]/2))
        #
        # # fig = plt.figure()
        # # plt.imshow(pdf4plot,cmap="gray")
        # # plt.axis('off')
        # # plt.show()
        #
        # pdf_profile = pdf4plot[:,y_mid]
        #
        # fig = plt.figure()
        # plt.plot(pdf_profile)
        # #plt.axis('off')
        # plt.show()
        # figname = 'pdf_profile_{}'.format(var_dens_flag)
        # fig.savefig(figname)

        # ----------------------------display mask ------------------------------
        # if pad_i==0:
        #     # # dispay
        #     fig = plt.figure()
        #     plt.imshow(mask,cmap="gray")
        #     plt.axis('off')
        #     plt.show()
        #     mask_figname = 'mask_R{}_{}'.format(R,var_dens_flag)
        #     fig.savefig(mask_figname)

        # mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils
        # ksp_padded_sampled = np.multiply(ksp2, mask_expanded)

        # -------------------------------------------------------------------------
        ksp_padded_sampled = np.multiply(kspace_slice, mask)

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
        # plt.imshow(np.log(np.abs(kspace_slice)),cmap="gray")
        # plt.subplot(1,3,2)
        # plt.imshow(np.log(np.abs(ksp_padded_sampled)),cmap="gray")
        # plt.subplot(1, 3, 3)
        # plt.imshow(mask, cmap="gray")
        # plt.show()

        # # ###################################### CS rec  ################################################

        # add the coil dimension for compatibility with Sigpy's requirements
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

        CS_recs_dict[pad_i, j] = rec_CS
        masks_dict[pad_i, j] = mask
        CS_NRMSE_arr[pad_i, j] = CS_Err.NRMSE

        # # display
        # fig = plt.figure()
        # plt.imshow(np.abs(rec_CS), cmap="gray")
        # plt.title('rec_CS - pad x{} {} VD - NRMSE {:0.3}'.format(pad_ratio,var_dens_flag,CS_Err.NRMSE))
        # plt.clim(0, cmax)
        # plt.colorbar()
        # plt.show()

        # s1 = int(rec_CS.shape[0] / 5)
        # s2 = int(rec_CS.shape[1] / 6)
        # s3 = int(rec_CS.shape[0] * 1.1 / 2)
        # s4 = int(rec_CS.shape[1] * 1.1 / 2)
        # rec_CS_zoomed = rec_CS[s1:s3, s2:s4]
        #
        # fig = plt.figure()
        # plt.imshow(np.abs(rec_CS), cmap="gray")
        # #plt.title('rec_CS - pad x{} {} VD - NRMSE {:0.3}'.format(pad_ratio,var_dens_flag,CS_Err.NRMSE))
        # plt.clim(0, cmax)
        # #plt.colorbar()
        # plt.axis('off')
        # plt.show()
        # figname = 'rec_CS_zoomed_pad_x{}_{}_NRMSE_{:0.2}.png'.format(pad_ratio,var_dens_flag,CS_Err.NRMSE)
        # fig.savefig(figname)

# save results
# data_filename = 'fast_exp_results_R{}'.format(R_vec[0])  # filename for saving
# np.savez(data_filename,rec_gold = rec_gold,CS_NRMSE_arr=CS_NRMSE_arr,R_vec=R_vec,sampling_type_vec=sampling_type_vec,pad_ratio_vec = pad_ratio_vec)


### display full-FOV image + white zoom-in square

# display full-FOV gold image in subplot [1,0]
im_full_FOV = np.abs(rec_gold_original_size)
# display zoom-in gold image
s1 = int(0.12 * im_full_FOV.shape[0])
s2 = int(0.3 * im_full_FOV.shape[1])
s3 = int(0.44 * im_full_FOV.shape[0])
s4 = int(0.62 * im_full_FOV.shape[1])

fig = plt.figure()
plt.imshow(im_full_FOV, cmap="gray")
plt.axis('off')
linewidth = 2
plt.plot([s2, s4], [s1, s1], color="white", linewidth=linewidth)
plt.plot([s2, s4], [s3, s3], color="white", linewidth=linewidth)
plt.plot([s2, s2], [s1, s3], color="white", linewidth=linewidth)
plt.plot([s4, s4], [s1, s3], color="white", linewidth=linewidth)
plt.show()
fig.savefig('brain_gold_full_FOV')

### display recons + NRMSEs in subplots
fig, ax = plt.subplots(nrows=(sampling_type_vec.shape[0]), ncols=(pad_ratio_vec.shape[0] + 2),
                       figsize=(20, 10), subplot_kw={'aspect': 1})
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

# rec_gold = np.rot90(rec_gold,2)
# s1 = int(rec_gold.shape[0] / 5)
# s2 = int(rec_gold.shape[1] / 6)
# s3 = int(rec_gold.shape[0] * 1.1 / 2)
# s4 = int(rec_gold.shape[1] * 1.1 / 2)


im_full_FOV = np.abs(rec_gold_original_size)

# display zoom-in gold image
s1 = int(0.12 * im_full_FOV.shape[0])
s2 = int(0.3 * im_full_FOV.shape[1])
s3 = int(0.44 * im_full_FOV.shape[0])
s4 = int(0.62 * im_full_FOV.shape[1])

rec_gold_zoomed = np.abs(rec_gold_original_size[s1:s3, s2:s4])
# im = np.rot90(im,2)
ax[0][0].imshow(rec_gold_zoomed, cmap="gray")
ax[0][0].set_axis_off()
row_cnt = 0

# display recons (zoomed-in)
for j in range(sampling_type_vec.shape[0]):

    for pad_i in range(pad_ratio_vec.shape[0]):
        # mask = masks_dict[pad_i, j]

        im = np.abs(CS_recs_dict[pad_i, j])
        # im = np.rot90(im,2)
        # s1 = int(im.shape[0] / 5)
        # s2 = int(im.shape[1] / 6)
        # s3 = int(im.shape[0] * 1.1 / 2)
        # s4 = int(im.shape[1] * 1.1 / 2)

        s1 = int(0.12 * im.shape[0])
        s2 = int(0.3 * im.shape[1])
        s3 = int(0.44 * im.shape[0])
        s4 = int(0.62 * im.shape[1])

        im4plot = im[s1:s3, s2:s4]
        # im4plot = np.rot90(im4plot,2)
        ax[j][pad_i + 1].imshow(im4plot, cmap="gray")
        ax[j][pad_i + 1].text(  # position text relative to Axes
            0, 0.95 * (s3 - s1), 'NRMSE {:.3f}'.format(CS_NRMSE_arr[pad_i, j]),
            ha='left', va='center',
            transform=ax[j][pad_i + 1].transData, color="yellow", fontsize=24)
        ax[j][pad_i + 1].set_axis_off()

        # sz1 = int(im.shape[0] * 1.25 / 3)
        # sz2 = int(im.shape[1] * 1.1 / 3)
        # sz3 = int(im.shape[0] * 1.1 / 2)
        # sz4 = int(im.shape[1] * 1.1 / 2)
        # im4plot_small = im[sz1:sz3, sz2:sz4]
        # ins = ax[row_cnt][pad_i + 1].inset_axes([0.40, 0.45, 0.53, 0.53])
        # ins.imshow(im4plot_small, cmap="gray")
        #
        # ins.spines['bottom'].set_color('yellow')
        # ins.spines['top'].set_color('yellow')
        # ins.spines['left'].set_color('yellow')
        # ins.spines['right'].set_color('yellow')
        # # ins.set_axis_off()
        # ins.set_xticks([])
        # ins.set_yticks([])
        # ins.set_axis('off')

        if pad_i == 0:
            mask = masks_dict[pad_i, j]
            # display mask
            ax[j][4].imshow(abs(mask), cmap="gray")
            ax[j][4].set_axis_off()

plt.subplots_adjust(wspace=0.05, hspace=0)
plt.axis('off')
fig.delaxes(ax[1, 0])  # remove empty axes
plt.show()
# fig.savefig(fname='brain_zpad_exp_R{}.png'.format(R))
