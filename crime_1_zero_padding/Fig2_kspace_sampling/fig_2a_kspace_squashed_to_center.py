# This code demonstrates subtle crime I with Compressed sensing.
# It reproduces the results in Figure 2a of the Subtle Data Crime paper.
# (c) Efrat Shimron, UC Berkeley, 2021.

##########################################################################################

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../")  # add folder above, for access to the functions library

from functions.utils import merge_multicoil_data, calc_pad_half
from functions.sampling_funcs import genPDF, genSampling
from functions.utils import calc_R_actual,calc_pad_half

#################################################################################
## Experiment set-up
#################################################################################

num_slices = 1
R_vec = np.array([6])
pad_ratio_vec = np.array([1,2,3])  # Define the desired padding ratios


##################################################################################
##                 Load data - a brain image from FastMRI
##################################################################################
example_filename = '../brain_data/file_brain_AXT2_207_2070504.h5'
f = h5py.File(example_filename, "r")
kspace_orig_multicoil = f["kspace"]
kspace_orig_multicoil = kspace_orig_multicoil[:,:,::2,:] # reduce image-domain FOV


#################################################################################
##                 Display kspace example
#################################################################################

fig = plt.figure()

for n in range(num_slices):
    ksp_full_multicoil = kspace_orig_multicoil[n, :, :, :].squeeze()

    print('============ slice %d ==========' % n)

    # for i, pad_half in enumerate(pad_half_vec):
    for i, pad_ratio in enumerate(pad_ratio_vec):
        print('--------- padding ratio %d from %d' % (i + 1, len(pad_ratio_vec)), ' --------- ')

        N_original_dim1 = ksp_full_multicoil.shape[1]
        N_original_dim2 = ksp_full_multicoil.shape[2]

        pad_half_dim1, N_tot_dim1 = calc_pad_half(N_original_dim1, pad_ratio)
        pad_half_dim2, N_tot_dim2 = calc_pad_half(N_original_dim2, pad_ratio)

        # zero-pad k-space - for every coil separately
        padding_lengths = ((0, 0), (pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))
        padding_lengths_yz = (
            (pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))  # padding lengths for the yz plane only
        ksp_full_multicoil_padded = np.pad(ksp_full_multicoil, padding_lengths, mode='constant',
                                           constant_values=(0, 0))

        # compute a single *magnitude* image from the data
        mag_im = merge_multicoil_data(ksp_full_multicoil_padded)

        # go back to k-space
        # ksp2 = np.fft.fftshift(np.fft.fft2(mag_im))
        ksp2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mag_im)))  # correction for the fftshift problem. 7-Jul-2020

        plt.subplot(1,3,i+1)
        plt.imshow(np.log(np.abs(ksp2)),cmap="gray")
        plt.axis('equal')
        plt.axis('off')

plt.show()
## save as png
# fig.savefig(fname='kspace_squashed.png')
## save as eps
# fig.savefig(fname='kspace_squashed.eps',forrmat = 'eps', dpi = 1000)


#######################################################################################################################
#                          Display sampling masks + yellow sqaures showing the original kspace area
#######################################################################################################################

######################## pdf profile figures #########################
R = np.array([6]) # acceleration factor

imSize = (320,320)
poly_degree_vec = np.array([1000,10,4])

for j, poly_degree in enumerate(poly_degree_vec):

    pdf = genPDF(imSize, poly_degree, 1 / R)

    # fig = plt.figure()
    # plt.imshow(pdf,cmap="jet")
    # plt.show()

    if poly_degree==1000:  # random-uniform
        pdf_profile = (1/R)*np.ones((imSize[0]))
        figname = 'pdf_profile_random_uniform.png'
    else:
        pdf_profile = pdf[int(imSize[0] / 2)-1, :]
        if poly_degree==10:
            figname = 'pdf_profile_weak_VD.png'
        else:
            figname = 'pdf_profile_strong_VD.png'

    fig = plt.figure()
    plt.plot(pdf_profile)
    plt.ylim(0,1.1)
    plt.xlim(0,(imSize[1]-1))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig(figname)



# ##################### brain data - load example from fastMRI ##############################

pad_ratio_vec = np.array([1, 2, 3])
poly_degree_vec = np.array([1000,10,4])

fig, ax = plt.subplots(nrows=(poly_degree_vec.shape[0]), ncols=(pad_ratio_vec.shape[0]),figsize=(7,6))

for i, pad_ratio in enumerate(pad_ratio_vec):
    print('========== padding ratio %d from %d' % (i + 1, len(pad_ratio_vec)), ' ============== ')

    for j, poly_degree in enumerate(poly_degree_vec):
        print('---------poly_degree  = %f' % (poly_degree), ' --------- ')

        im1 = np.ones((N_original_dim1, N_original_dim2))

        pad_half_dim1, N_tot_dim1 = calc_pad_half(N_original_dim1, pad_ratio)
        pad_half_dim2, N_tot_dim2 = calc_pad_half(N_original_dim2, pad_ratio)

        # zero-pad k-space - for every coil separately
        padding_lengths_yz = ((pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))

        im_padded = np.pad(im1, padding_lengths_yz, mode='constant', constant_values=(0, 0))
        inds_inner_square = np.nonzero(im_padded)

        imSize = im_padded.shape

        pdf = genPDF(imSize, poly_degree, 1 / R)
        mask = genSampling(pdf, iter=10, tol=60)
        R_full_mask_actual = calc_R_actual(mask)

        mask_effective = np.multiply(im_padded, mask)

        a = im_padded
        b = mask
        inds_inner_square = np.nonzero(a)

        mask_effective_inner_square = b[inds_inner_square]
        mask_effective_vec = np.reshape(mask_effective_inner_square, (1, -1))
        R_eff = mask_effective_vec.shape[1] / np.count_nonzero(mask_effective_vec)
        print('R_eff={:.1f}'.format(R_eff))

        # rectangle
        x1 = pad_half_dim1
        x2 = pad_half_dim1 + N_original_dim1-1
        y1 = pad_half_dim2
        y2 = pad_half_dim2 + N_original_dim2-1

        # masks + yellow lines
        ax[j][i].imshow(mask, cmap="gray")
        ax[j][i].set_xticks([0,mask.shape[0]-1])
        ax[j][i].set_yticks([0,mask.shape[1]-1])
        x_ticks = ['0', str(mask.shape[0])]
        ax[j][i].set_xticklabels(x_ticks, fontsize=12)
        y_ticks = ['0',str(mask.shape[1])]
        ax[j][i].set_yticklabels(y_ticks,fontsize=12)


        ax[j][i].axis('equal')
        ax[j][i].axis('tight')

        gca= ax[j][i]
        if i==0:
            linewidth = 4
        else:
            linewidth = 3
        gca.plot([x1,x1], [y1,y2], color="yellow",linewidth=linewidth)
        gca.plot([x1,x2], [y1,y1], color="yellow",linewidth=linewidth)
        gca.plot([x2,x2], [y1,y2], color="yellow",linewidth=linewidth)
        gca.plot([x1,x2], [y2,y2], color="yellow", linewidth=linewidth)

fig.tight_layout()
plt.show()
# # save as png
# fig.savefig(fname='full_masks_w_yellow_squares_R{}.png'.format(R, poly_degree_vec[0], poly_degree_vec[-1]))
# # save as eps
# fig.savefig(fname='full_masks_w_yellow_sqaures_R{}.eps'.format(R, poly_degree_vec[0], poly_degree_vec[-1]),format = 'eps', dpi = 1000)
#


