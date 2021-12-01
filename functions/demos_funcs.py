import numpy as np
import matplotlib.pyplot as plt

#from SSIM_PIL import compare_ssim
import sigpy as sp  # sigpy can be installed with "pip install sigpy". For more info see https://pypi.org/project/sigpy/
import sigpy
from sigpy import mri as mr
from sigpy import *
from sigpy.mri import *
from PIL import Image  # for JPEG compression
#from SSIM_PIL import compare_ssim

#from functions.sampling_funcs import gen_2D_rand_samp_mask, gen_1D_rand_samp_mask
from functions.sampling_funcs import genPDF, genSampling
from functions.error_funcs import error_metrics
from functions.utils import merge_multicoil_data, calc_pad_half
from functions.new_sigpy_func import new_poisson
from functions.error_funcs import error_metrics

# this function was moved to functions.utils
# def calc_pad_half(N_original,pad_ratio):
#
#     N_tot = N_original*pad_ratio  # this will be the total k-space size
#     diff = np.ceil(N_tot - N_original)  # this is the total padding length
#     pad_size_vec = diff.astype(int)  # convert from float to int
#     pad_half_vec = (pad_size_vec/2).astype(int) # kspace will be padded with "pad_half" from each size, so the total padding length is padd_size
#
#     return pad_half_vec, N_tot



# =============== now with poly_degree_vec instead of alpha_vec ========================
def demo1_zero_pad_MAG_run_exps(ksp_all_data, pad_ratio_vec, num_slices, R_vec, num_realizations, poly_degree_vec,
                                show_flag=0,lamda=0.005):
    # calib = [24,24]
    #calib = [4, 4]
    gold_dict = {}
    recs_dict = {}
    masks_dict = {}
    R_eff_arr = {}
    masks_effective_dict = {}

    NRMSE_arr = np.empty([pad_ratio_vec.shape[0], num_realizations, num_slices, R_vec.shape[0], poly_degree_vec.shape[0]])
    SSIM_arr = np.empty([pad_ratio_vec.shape[0], num_realizations, num_slices, R_vec.shape[0], poly_degree_vec.shape[0]])

    # grid_search = {(x,y) for x in range(num_realizations) for y in range(R_vec.shape[0]) }
    grid_search = {(x, y, z) for x in range(num_realizations) for y in range(R_vec.shape[0]) for z in
                   range(poly_degree_vec.shape[0])}

    for n in range(num_slices):
        ksp_full_multicoil = ksp_all_data[:, :, :, n]
        initial_mag_im = merge_multicoil_data(ksp_full_multicoil)  # this is only for display, not for computations

        print('============ slice %d ==========' % n)

        # for i, pad_half in enumerate(pad_half_vec):
        for i, pad_ratio in enumerate(pad_ratio_vec):
            print('--------- padding ratio %d from %d' % (i + 1, len(pad_ratio_vec)), ' --------- ')

            N_original_dim1 = ksp_all_data.shape[1]
            N_original_dim2 = ksp_all_data.shape[2]

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
            ksp2 = np.fft.fftshift(
                np.fft.fft2(np.fft.fftshift(mag_im)))  # correction for the fftshift problem. 7-Jul-2020

            ksp_unscaled = ksp2 # for debugging only

            ksp2 = ksp2 / np.max(np.abs(ksp2))

            # # deugging plots
            # fig = plt.figure()
            # plt.imshow(mag_im, cmap="gray")
            # plt.colorbar()
            # plt.title('mag_im (merged image)')
            # plt.show()
            #
            # fig = plt.figure()
            # plt.imshow(np.log(np.abs(ksp2)))
            # plt.title('ksp of merged image')
            # plt.colorbar()
            # plt.show()

            if (show_flag == 1):
                fig, a = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
                a[0][0].imshow(np.log(np.abs(ksp_full_multicoil[0, :, :].squeeze())), vmin=-10, vmax=2, cmap="gray")
                a[0][0].set_title("coil #1 raw ksp")
                a[0][1].imshow(initial_mag_im, cmap="gray")
                a[0][1].set_title("multi-coil merged mag im:  sqrt(SOS)")
                a[1][0].imshow(np.log(np.abs(ksp_full_multicoil_padded[0, :, :].squeeze())), vmin=-10, vmax=2,
                               cmap="gray")
                a[1][0].set_title("coil #1 padded ksp")
                a[1][1].imshow(mag_im, cmap="gray")
                a[1][1].set_title("multi-coil merged mag im:  sqrt(SOS)")
                a[2][0].imshow(np.log(np.abs(ksp2)), vmin=-10, vmax=2, cmap="gray")
                a[2][0].set_title("ksp2 =F[sqrt(SOS(padded_ksp))]")
                fig.delaxes(a[2, 1])
                plt.show()

            # add the coil dimension (although there's only 1 coil - this is for compatibility with Sigpy's inputs requirements)
            ksp_full_padded = np.expand_dims(ksp2, axis=0)
            Scoils, Sx, Sy = ksp_full_padded.shape  # get size. Notice: the first one is the coils dimension

            virtual_sens_maps = np.ones_like(ksp_full_padded)  # sens maps are all ones because we have a "single-coil" magnitude image.

            # ------- run recon experiment -----------------
            # gold standard recon (fully-sampled data, with the current zero padding length)
            print('Gold standard rec from fully sampled data...')

            # old code, it was replaced by sp.ifft
            #rec_gold_CS = mr.app.L1WaveletRecon(ksp_full_padded, virtual_sens_maps, lamda=0.005, show_pbar=False).run()

            rec_gold = sp.ifft(ksp2)

            # deugging plots
            #
            # fig = plt.figure()
            # plt.imshow(np.abs(rec_gold), cmap="gray")
            # plt.title('rec_gold')
            # plt.colorbar()
            # rec_max = np.max(np.abs(rec_gold))
            # plt.clim(0,rec_max)
            # plt.show()

            # sanity check
            # fig = plt.figure
            # plt.subplot(2,1,1)
            # plt.imshow(np.abs(rec_gold_CS),cmap="gray")
            # plt.colorbar()
            # plt.title('CS rec gold')
            #
            # plt.subplot(2,1,2)
            # plt.imshow(np.abs(rec_gold), cmap="gray")
            # plt.colorbar()
            # plt.title('rec gold ifft sigpy')
            # plt.show()



            # debugging
            assert np.isnan(rec_gold).any() == False, 'there are NaN values in rec_gold! slice {}'.format(n)

            gold_dict[i, n] = rec_gold  # store the results in a dictionary (note: we use a dictionary instead of a numpy array beause
            # different images have different sizes due to the k-space zero-padding)

            for k, r, aa in grid_search:
                R = R_vec[r]
                #accel = R
                poly_degree = poly_degree_vec[aa]
                print('realization #', k, ' R=', R_vec[r], ' poly_deg=', poly_degree_vec[aa])

                img_shape = np.array([ksp_full_padded.shape[1], ksp_full_padded.shape[2]])

                # mask,_ = gen_2D_rand_samp_mask(Scoils, Sx, Sy,accel,calib,seed_val=k)
                # mask, _ = new_poisson(img_shape,accel,calib,return_density=True,seed=k,alpha=alpha)
                #alpha = 1
                #mask, _ = new_poisson(img_shape, accel, 30, calib, return_density=True, seed=k, alpha=alpha,crop_corner=False)

                if poly_degree>=1000: # generate a uniform-random sampling mask
                    tmp = np.random.randint(1, 1000, img_shape[0] * img_shape[1])
                    inds = np.where(tmp <= (1000 / R))
                    mask_1D = np.zeros(img_shape[0] * img_shape[1])
                    mask_1D[inds] = 1
                    mask = mask_1D
                    #dtype = np.complex
                    #mask = mask.reshape(img_shape).astype(dtype)
                    mask = mask.reshape(img_shape)
                else:
                    pdf = genPDF(img_shape, poly_degree, 1 / R)
                    mask = genSampling(pdf, iter=10, tol=60)

                mask_expanded = np.expand_dims(mask, axis=0)  # add the empty coils dim to the mask
                ksp_padded_sampled = np.multiply(ksp_full_padded, mask_expanded)

                # CS recon from sampled data
                print('CS rec from sub-sampled data...')
                rec = mr.app.L1WaveletRecon(ksp_padded_sampled, virtual_sens_maps, lamda=lamda, show_pbar=False).run()

                fig = plt.figure()
                plt.imshow(np.log(np.abs(ksp_full_padded[0,:,:].squeeze())), cmap="gray")
                plt.axis('off')
                plt.clim(-10,0)
                #plt.colorbar()
                plt.show()
                ksp_figname = 'kspace_full_pad_{}'.format(pad_ratio)
                fig.savefig(ksp_figname)

                # display
                fig = plt.figure()
                plt.imshow(np.log(np.abs(ksp_padded_sampled[0,:,:].squeeze())), cmap="gray")
                plt.axis('off')
                plt.clim(-5, 0)
                #plt.colorbar()
                plt.show()
                ksp_figname = 'kspace_sampled_pad_{}_R{}'.format(pad_ratio,R)
                fig.savefig(ksp_figname)

                # # debugging - code for calibrating lamda
                # lam_vec = np.array([1e-6, 5*1e-6, 1e-5, 5*1e-5, 0.0001])
                # err_vs_lam_vec = np.zeros_like(lam_vec)
                # for lam_i in range(lam_vec.shape[0]):
                #     lam = lam_vec[lam_i]
                #     print('lam=',lam)
                #     rec = mr.app.L1WaveletRecon(ksp_padded_sampled, virtual_sens_maps, lamda=lam,show_pbar=False).run()
                #
                #     fig = plt.figure()
                #     plt.imshow(np.abs(rec),cmap="gray")
                #     plt.title('rec - lambda = {}'.format(lam))
                #     plt.colorbar()
                #     plt.clim(0,rec_max)
                #     plt.show()
                #
                #     ERR = error_metrics(rec_gold, rec)
                #     ERR.calc_NRMSE()
                #     err_vs_lam_vec[lam_i]=ERR.NRMSE
                #
                # fig = plt.figure()
                # plt.plot(lam_vec[0:4],err_vs_lam_vec[0:4])
                # plt.xlabel('lam')
                # plt.ylabel('NRMSE')
                # plt.show()


                # debugging plots
                # fig = plt.figure()
                # plt.imshow(mask,cmap="gray")
                # plt.show()
                #
                # ksp_s = ksp_padded_sampled[0,:,:].squeeze()
                # fig = plt.figure()
                # plt.imshow(np.log(np.abs(ksp_s)))
                # plt.colorbar()
                # plt.title('sampled k-space')
                # plt.show()



                # recs_dict[i, k, n, R, aa] = rec  # store the results in a dictionary
                # masks_dict[i, k, n, R, aa] = mask

                assert np.isnan(rec).any() == False, 'there are NaN values in rec! slice {}'.format(n)

                A = error_metrics(rec_gold, rec)
                A.calc_NRMSE()
                A.calc_SSIM()


                # ------------ calc R_eff & mask_effective -------------------
                ones_square = np.ones([N_original_dim1, N_original_dim2])
                ones_square_padded = np.pad(ones_square, padding_lengths_yz, mode='constant',
                                            constant_values=(0, 0))  # this is a
                inds_inner_square = np.nonzero(ones_square)

                # mask, _ = new_poisson(im_padded.shape,accel,30,calib,return_density=True,seed=0,alpha=alpha,crop_corner=False)
                # mask = np.abs(mask)
                mask_effective = np.multiply(ones_square_padded,
                                             mask)  # for display only, this isn't used for reconstruction

                a = ones_square_padded
                b = mask
                # inds_inner_square = np.nonzero(a)
                # inds_inner_sqare_sampled = np.nonzero(b==1)

                mask_effective_inner_square = b[
                    inds_inner_square]  # extract only the mask's inner part, with the original dims
                mask_effective_vec = np.reshape(mask_effective_inner_square, (1, -1))
                R_eff = mask_effective_vec.shape[1] / np.count_nonzero(mask_effective_vec)

                assert np.isnan(A.NRMSE).any() == False, 'A.NRMSE==Nan for slice {}'.format(n)

                # ------------- save in arrays --------------
                # print("NRMSE= %3f" % A.NRMSE)
                NRMSE_arr[i, k, n, r, aa] = A.NRMSE
                SSIM_arr[i, k, n, r, aa] = A.SSIM


                recs_dict[i, k, n, r, aa] = rec  # store the results in a dictionary
                masks_dict[i, k, n, r, aa] = mask
                masks_effective_dict[i, k, n, r, aa] = mask_effective

                R_eff_arr[i, k, n, r, aa] = R_eff


                # ---------------- plot ----------------

                # if (show_flag == 1) & r == 0 & k == 0 & (i == pad_ratio_vec.shape[0]):
                #     fig, ax = plt.subplots(nrows=1, ncols=3)
                #     ax[0].imshow(np.abs(rec_gold), cmap="gray")
                #     ax[0].set_title("gold standard")
                #     ax[1].imshow(np.abs(mask.squeeze()), cmap="gray")
                #     ax[1].set_title("samp mask")
                #     ax[2].imshow(np.abs(rec), cmap="gray")
                #     ax[2].set_title("CS rec")
                #     # fig.title('padding half={}'.format(pad_half))

    return gold_dict, recs_dict, NRMSE_arr, SSIM_arr, masks_dict, R_eff_arr, masks_effective_dict


# # =============== pad_ratio_vec instead of pad_half_vec ========================
# def demo1_zero_pad_MAG_run_exps(ksp_all_data,pad_ratio_vec,num_slices,R_vec,num_realizations,alpha_vec,show_flag=0):
#     #calib = [24,24]
#     calib = [4,4]
#     gold_dict = {}
#     recs_dict = {}
#     masks_dict = {}
#     R_eff_arr = {}
#     masks_effective_dict = {}
#
#     NRMSE_arr = np.empty([pad_ratio_vec.shape[0],num_realizations,num_slices,R_vec.shape[0],alpha_vec.shape[0]])
#     SSIM_arr = np.empty([pad_ratio_vec.shape[0],num_realizations,num_slices,R_vec.shape[0],alpha_vec.shape[0]])
#
#     #grid_search = {(x,y) for x in range(num_realizations) for y in range(R_vec.shape[0]) }
#     grid_search = {(x,y,z) for x in range(num_realizations) for y in range(R_vec.shape[0]) for z in range(alpha_vec.shape[0])}
#
#
#     for n in range(num_slices):
#         ksp_full_multicoil = ksp_all_data[:,:,:,n]
#         initial_mag_im = merge_multicoil_data(ksp_full_multicoil)  # this is only for display, not for computations
#
#         print('============ slice %d ==========' % n)
#
#
#         #for i, pad_half in enumerate(pad_half_vec):
#         for i, pad_ratio in enumerate(pad_ratio_vec):
#             print('--------- padding ratio %d from %d' % (i + 1, len(pad_ratio_vec)),' --------- ')
#
#             N_original_dim1 = ksp_all_data.shape[1]
#             N_original_dim2 = ksp_all_data.shape[2]
#
#             pad_half_dim1, N_tot_dim1 = calc_pad_half(N_original_dim1,pad_ratio)
#             pad_half_dim2, N_tot_dim2 = calc_pad_half(N_original_dim2,pad_ratio)
#
#             # zero-pad k-space - for every coil separately
#             padding_lengths = ((0,0),(pad_half_dim1,pad_half_dim1),(pad_half_dim2,pad_half_dim2))
#             padding_lengths_yz = ((pad_half_dim1,pad_half_dim1),(pad_half_dim2,pad_half_dim2)) # padding lengths for the yz plane only
#             ksp_full_multicoil_padded = np.pad(ksp_full_multicoil,padding_lengths,mode='constant',constant_values = (0,0))
#
#             # compute a single *magnitude* image from the data
#             mag_im = merge_multicoil_data(ksp_full_multicoil_padded)
#
#             # go back to k-space
#             # ksp2 = np.fft.fftshift(np.fft.fft2(mag_im))
#             ksp2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mag_im)))  # correction for the fftshift problem. 7-Jul-2020
#
#             if (show_flag==1):
#                     fig, a = plt.subplots(nrows=3,ncols=2,figsize=(15,15))
#                     a[0][0].imshow(np.log(np.abs(ksp_full_multicoil[0,:,:].squeeze())),vmin=-10,vmax=2,cmap="gray")
#                     a[0][0].set_title("coil #1 raw ksp")
#                     a[0][1].imshow(initial_mag_im,cmap="gray")
#                     a[0][1].set_title("multi-coil merged mag im:  sqrt(SOS)")
#                     a[1][0].imshow(np.log(np.abs(ksp_full_multicoil_padded[0,:,:].squeeze())),vmin=-10,vmax=2,cmap="gray")
#                     a[1][0].set_title("coil #1 padded ksp")
#                     a[1][1].imshow(mag_im,cmap="gray")
#                     a[1][1].set_title("multi-coil merged mag im:  sqrt(SOS)")
#                     a[2][0].imshow(np.log(np.abs(ksp2)),vmin=-10,vmax=2,cmap="gray")
#                     a[2][0].set_title("ksp2 =F[sqrt(SOS(padded_ksp))]")
#                     fig.delaxes(a[2, 1])
#
#
#
#             # add the coil dimension (although there's only 1 coil - this is for compatibility with Sigpy's inputs requirements)
#             ksp_full_padded = np.expand_dims(ksp2,axis=0)
#             Scoils, Sx, Sy = ksp_full_padded.shape # get size. Notice: the first one is the coils dimension
#
#             virtual_sens_maps = np.ones_like(ksp_full_padded)  # sens maps are all ones because we have a "single-coil" magnitude image.
#
#             # ------- run recon experiment -----------------
#             # gold standard recon (fully-sampled data, with the current zero padding length)
#             print('Gold standard rec from fully sampled data...')
#             rec_gold = mr.app.L1WaveletRecon(ksp_full_padded, virtual_sens_maps, lamda=0.005, show_pbar=False).run()
#
#             gold_dict[i,n] = rec_gold  # store the results in a dictionary (note: we use a dictionary instead of a numpy array beause
#                                        # different images have different sizes due to the k-space zero-padding)
#
#             for k,r,aa in grid_search:
#                         accel = R_vec[r]
#                         alpha = alpha_vec[aa]
#                         print('realization #',k,' accel=',R_vec[r],' alpha=',alpha_vec[aa])
#
#                         img_shape = np.array([ksp_full_padded.shape[1], ksp_full_padded.shape[2]])
#
#                         #mask,_ = gen_2D_rand_samp_mask(Scoils, Sx, Sy,accel,calib,seed_val=k)
#                         #mask, _ = new_poisson(img_shape,accel,calib,return_density=True,seed=k,alpha=alpha)
#                         mask, _ = new_poisson(img_shape,accel,30,calib,return_density=True,seed=k,alpha=alpha,crop_corner=False)
#
#                         ksp_padded_sampled = np.multiply(ksp_full_padded,mask)
#
#                         # CS recon from sampled data
#                         print('CS rec from sub-sampled data...')
#                         rec = mr.app.L1WaveletRecon(ksp_padded_sampled, virtual_sens_maps, lamda=0.005, show_pbar=False).run()
#
#                         recs_dict[i,k,n,accel,aa] = rec   # store the results in a dictionary
#                         masks_dict[i,k,n,accel,aa] = mask
#
#                         A = error_metrics(rec_gold,rec)
#                         A.calc_NRMSE()
#                         A.calc_SSIM()
#                         #print("NRMSE= %3f" % A.NRMSE)
#                         NRMSE_arr[i,k,n,r,aa] = A.NRMSE
#                         SSIM_arr[i,k,n,r,aa] = A.SSIM
#
#                         # ------------ calc R_eff & mask_effective -------------------
#                         ones_square = np.ones([N_original_dim1,N_original_dim2])
#                         ones_square_padded = np.pad(ones_square,padding_lengths_yz,mode='constant',constant_values = (0,0))   # this is a
#                         inds_inner_square = np.nonzero(ones_square)
#
#                         #mask, _ = new_poisson(im_padded.shape,accel,30,calib,return_density=True,seed=0,alpha=alpha,crop_corner=False)
#                         #mask = np.abs(mask)
#                         mask_effective = np.multiply(ones_square_padded,mask) # for display only, this isn't used for reconstruction
#
#                         a = ones_square_padded
#                         b = mask
#                         #inds_inner_square = np.nonzero(a)
#                         #inds_inner_sqare_sampled = np.nonzero(b==1)
#
#                         mask_effective_inner_square = b[inds_inner_square]  # extract only the mask's inner part, with the original dims
#                         mask_effective_vec = np.reshape(mask_effective_inner_square,(1,-1))
#                         R_eff = mask_effective_vec.shape[1]/np.count_nonzero(mask_effective_vec)
#
#                         R_eff_arr[i,k,n,r,aa] = R_eff
#                         masks_effective_dict[i,k,n,accel,aa] = mask_effective
#
#                         # ---------------- plot ----------------
#
#                         if (show_flag==1) & r==0 & k==0 & (i==pad_ratio_vec.shape[0]):
#                                 fig,ax = plt.subplots(nrows=1,ncols=3)
#                                 ax[0].imshow(np.abs(rec_gold),cmap="gray")
#                                 ax[0].set_title("gold standard")
#                                 ax[1].imshow(np.abs(mask.squeeze()),cmap="gray")
#                                 ax[1].set_title("smap mask")
#                                 ax[2].imshow(np.abs(rec),cmap="gray")
#                                 ax[2].set_title("CS rec")
#                                 #fig.title('padding half={}'.format(pad_half))
#
#     return gold_dict, recs_dict, NRMSE_arr, SSIM_arr, masks_dict, R_eff_arr, masks_effective_dict
#

#
#
# def demo3_run_exps(ksp_all_data,compression_val,num_slices,num_realizations,R_vec,show_flag):
#     calib = [24,24]
#
#     grid_search = {(x,y) for x in range(num_realizations) for y in range(R_vec.shape[0]) }
#
#     _, Sx, Sy, _ = ksp_all_data.shape
#
#     NRMSE_vs_quality= np.empty([len(compression_val),num_slices,num_realizations,R_vec.shape[0]])
#     SSIM_vs_quality= np.empty([len(compression_val),num_slices,num_realizations,R_vec.shape[0]])
#     gold_recs_array = np.empty([Sx, Sy, num_slices,len(compression_val)],dtype='complex')
#     recs_array = np.empty([Sx, Sy, num_slices,len(compression_val),num_realizations,R_vec.shape[0]],dtype='complex')
#
#     for n in range(num_slices):
#         ksp_full_multicoil = ksp_all_data[:,:,:,n]
#
#         # create virtual single-coil data by averaging the multi-coils data along the coils dimension
#         ksp_1coil = np.mean(ksp_full_multicoil,axis=0)
#
#         im_complex =  np.fft.ifft2(ksp_1coil)
#         im_mag = np.abs(im_complex)     # convert from complex to real values for saving image in JPEG format
#         im_mag_uint8 = (im_mag * 255 / np.max(im_mag)).astype('uint8')  # convert from float to uint8 for saving in JPEG format
#
#
#         for i,quality_val in enumerate(compression_val):
#                 print('--------- i={}, quality_val={} ---------'.format(i,quality_val))
#                 # -------- apply lossy compression - save in JPEG format -------
#                 # Generally, the quality value can be anything between 0 and 100; the default value is 75
#                 #quality_val = compression_val[i].astype(int)  # let's try 2, 3, 4 or 5 for extremeley low quality.
#                 #print(type(quality_val))
#                 #im_PIL.save('try1.jpg', format='JPEG',quality=quality_val)
#                 im_PIL = Image.fromarray(im_mag_uint8)
#
#                 im_PIL.save('data/OLD_Demo2_JPEG_data/im_compressed.jpg', format='JPEG',quality=quality_val,subsampling=0)
#
#                 # -------- load JPEG image -------
#                 im_compressed = np.asarray(Image.open('data/OLD_Demo2_JPEG_data/im_compressed.jpg'))  # convert from pillow format to numpy ndarray
#                 #print('image after save + load:')
#                 #pl.ImagePlot(im_compressed)
#
#                 kspace_gold = np.fft.fftshift(np.fft.fft2(im_compressed))
#                 kspace_gold_1coil = kspace_gold
#                 #kspace_gold = np.fft.fft2(im_mag_uint8)
#                 kspace_gold = np.expand_dims(kspace_gold,axis=0)
#
#                 # calc gold standard recon - from fully sampled data
#                 print('gold standard calc')
#                 rec_gold = mr.app.L1WaveletRecon(kspace_gold, np.ones_like(kspace_gold), lamda=0.005, show_pbar=False).run()
#                 #rec_gold = np.fft.fftshift(rec_gold)
#                 #pl.ImagePlot(rec_gold)
#
#                 # calc CS recon from subsampled data
#                 #Scoils, Sx, Sy = kspace_gold.shape
#
#                 gold_recs_array[:,:,n,i] = rec_gold
#
#
#                 for s, r in  grid_search:
#                     accel = R_vec[r]
#                     print('------ accel = {} ------'.format(accel))
#                     mask, _ = gen_2D_rand_samp_mask(1, Sx, Sy,accel,calib, seed_val=s)
#                     #print('actual R={:0.2f}'.format(R))
#
#                     kspace_sampled = np.multiply(kspace_gold,mask)
#
#                     print('CS rec')
#                     rec = mr.app.L1WaveletRecon(kspace_sampled, np.ones_like(kspace_sampled), lamda=0.005, show_pbar=False).run()
#                     #rec = np.fft.fftshift(rec)
#                     #pl.ImagePlot(rec)
#                     recs_array[:,:,n,i,s,r] = rec
#
#                     A = error_metrics(rec_gold,rec)
#                     A.calc_NRMSE()
#                     A.calc_SSIM()
#                     print("NRMSE= %3f" % A.NRMSE)
#                     NRMSE_vs_quality[i,n,s,r] = A.NRMSE
#                     SSIM_vs_quality[i,n,s,r] = A.SSIM
#
#                     if (show_flag==1) & i==0 & s==0 & r==0:
#                             fig, ax = plt.subplots(2,3)
#                             ax[0][0].imshow(np.fft.fftshift(np.abs(im_complex)))
#                             ax[0][0].set_title('Full-quality image',fontsize=14)
#
#                             ax[1][0].imshow(np.fft.fftshift(im_compressed))
#                             ax[1][0].set_title('JPEG-compressed image \n q={}'.format(quality_val),fontsize=14)
#
#
#                             ax[0][1].imshow(np.log(np.abs(kspace_gold).squeeze()),cmap="jet")
#                             #fig.colorbar(im, ax=ax[0][0])
#                             ax[0][1].set_title('Fully-sampled k-space ',fontsize=14)
#
#                             ax[1][1].imshow(np.log(np.abs(kspace_sampled).squeeze()),cmap="jet")
#                             ax[1][1].set_title('Sampled k-space')
#
#                             ax[0][2].imshow(np.abs(rec_gold.squeeze()))
#                             ax[0][2].set_title('Gold Standard from \n JPEG-compressed fully-sampled data')
#
#                             ax[1][2].imshow(np.abs(rec_gold.squeeze()))
#                             ax[1][2].set_title('CS rec NRMSE={:0.3f} \n from subsampled data R={}'.format(A.NRMSE,accel))
#                             plt.show
#
#     return NRMSE_vs_quality,SSIM_vs_quality,gold_recs_array,recs_array