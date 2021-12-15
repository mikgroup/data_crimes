"""
This module includes functions for creating variable-density sampling patterns from a Poisson distribution,
for 2D Cartesian data.
The Variable Density functions are based on Miki Lustig's Sparse MRI Matlab toolbox (2007).

Efrat Shimron (UC Berkeley, 2021).
"""

import numpy as np
import matplotlib.pyplot as plt


############### genPDF  ###########################
def genPDF(imSize, p, pctg, distType=2, radius=0, disp=0, pdf_show_flag=0):
    # This function generates a pdf for a 1d or 2d random sampling pattern
    # with polynomial variable density sampling

    # Input:
    # imSize - size of matrix or vector
    # p - power of polynomial
    # pctg - partial sampling factor e.g. 0.5 for half
    # distType - 1 or 2 for L1 or L2 distance measure
    # radius - radius of fully sampled center
    # disp - display output

    # Output:
    # pdf - the pdf
    # val - min sampling density

    # (c) Michael Lustig 2007. Converted from Matlab to Python by Efrat Shimron (2020)

    minval = 0
    maxval = 1
    val = 0.5

    if len(imSize) == 2:  # 2D case

        sx = imSize[0]
        sy = imSize[1]
        PCTG = np.floor(pctg * sx * sy)

        x_co = np.linspace(-1, 1, sy)  # coordinates
        y_co = np.linspace(-1, 1, sx)  # coordinates
        x, y = np.meshgrid(x_co, y_co)

        if distType == 1:
            r = np.max(np.abs(x), np.abs(y))
        else:
            r = np.sqrt(x ** 2 + y ** 2)
            r = r / np.max(np.abs(r.reshape(1, -1)))

    elif len(imSize) == 1:  # 1D case
        sx = imSize[0]
        PCTG = np.floor(pctg * sx)
        r = np.abs(np.linspace(-1, 1, sx))

        # create PDF
    idx = np.where(r < radius)
    pdf = (1 - r) ** p + val
    pdf[pdf > 1] = 1
    pdf[idx] = 1

    if pdf_show_flag == 1:
        # pdf display
        # 2D case
        if len(pdf.shape) == 2:
            plt.imshow(pdf, cmap="gray")
            plt.show()
        # 1D case
        elif len(pdf.shape) == 1:
            plt.plot(pdf)
            plt.show()

    assert np.floor(np.sum(pdf)) > PCTG, "infeasible sampling, fully-sampled DC area is too large, increase poly_degree"

    while (1):
        val = minval / 2 + maxval / 2
        pdf = (1 - r) ** p + val
        pdf[pdf > 1] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf))
        if N > PCTG:  # infeasible
            maxval = val
        if N < PCTG:  # feasible, but not optimal
            minval = val
        if N == PCTG:  # optimal
            break

    return pdf


############### genSampling  ###########################


def genSampling(pdf, iter, tol, calib=[1, 1]):
    #  A monte-carlo algorithm to generate a sampling pattern with
    #  minimum peak interference. The number of samples will be
    #  sum(pdf) +- tol
    #
    # Inputs:
    #   pdf - probability density function to choose samples from
    #   iter - vector of min interferences measured each try
    #   tol  - the deviation from the desired number of samples in samples
    # Outputs:
    #  mask - sampling pattern

    # (c) Michael Lustig 2007.
    # Converted from Matlab to Python by Efrat Shimron (2020)

    # print('calib=', calib)

    pdf[pdf > 1] = 1
    K = np.sum(pdf[::])
    minIntr = np.array([1e99])
    minIntrVec = np.zeros(pdf.shape)

    np.random.seed(seed=None)  # remove this line to generate the same mask each time

    # 2D case
    if len(pdf.shape) == 2:
        for n in range(iter):
            tmp = np.zeros(pdf.shape)
            while np.abs(np.sum(tmp[::]) - K) > tol:
                tmp = np.random.random(pdf.shape) < pdf

            TMP = np.fft.ifft2(tmp / pdf)
            if np.max(np.abs(TMP[1:-1])) < minIntr:
                minIntr = np.max(np.abs(TMP[1:-1]))
                minIntrVec = tmp

        mask = minIntrVec

        # add calibration area
        nx = mask.shape[-1]
        ny = mask.shape[-2]

        mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
        int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    # 1D case
    elif len(pdf.shape) == 1:
        for n in range(iter):
            tmp = np.zeros(pdf.shape)
            while np.abs(np.sum(tmp[::]) - K) > tol:
                tmp = np.random.random(pdf.shape) < pdf

            TMP = np.fft.ifft(tmp / pdf)
            if np.max(np.abs(TMP[1:-1])) < minIntr:
                minIntr = np.max(np.abs(TMP[1:-1]))
                minIntrVec = tmp

        mask = minIntrVec

        # add calibration area
        nx = mask.shape[0]

        mask[int(nx / 2 - calib / 2): int(nx / 2 + calib / 2)] = 1

    return mask


###################################################################################################
#                                             2D var-dens (from 1c_MoDL_OLD_1D_sampling run10)
###################################################################################################
def gen_2D_var_dens_mask(R, imSize, sampling_flag, calib=[24, 24]):
    NX = imSize[0]
    NY = imSize[1]

    if sampling_flag == 'random':
        # 2D random-uniform
        tmp = np.random.randint(1, 1000, (NX, NY))
        inds = np.where(tmp <= (1000 / R))
        mask = np.zeros([NX, NY])
        mask[inds] = 1



        fig = plt.figure()
        plt.imshow(np.abs(mask), cmap="gray")
        plt.title('mask_1D')
        plt.show()

        pdf = (1 / R) * np.ones(imSize)
        poly_degree = 'NaN'

    else:
        if R == 12:
            if sampling_flag == 'weak':
                poly_degree = 15
            elif sampling_flag == 'strong':
                poly_degree = 9
        elif R == 8:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 7
        elif R == 6:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 5
        elif R == 5:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 4.5
        elif R == 4:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 3
        elif R == 3:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 2
        elif R == 2:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 1

    
        # create a 2D pdf
        pdf = genPDF(imSize, poly_degree, 1 / R)
        mask = genSampling(pdf, iter=50, tol=20, calib=calib)

    return mask, pdf, poly_degree



# # ---------------------- 2D var-dens example & display ------------------------------
# NY = 256
# NX = 128
#
# calib = np.array(24*[NX/256,NY/256])
#
# R_vec = np.array([2,4,6])
# imSize = np.array([NX,NY])
#
# cnt = 0
# fig = plt.figure()
# for r in range(R_vec.shape[0]):
#     R = R_vec[r]
#     print('R=',R)
#
#     sampling_flag = 'strong'
#     mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, sampling_flag,calib=calib)
#
#     cnt += 1
#     plt.subplot(R_vec.shape[0],2,cnt)
#     plt.imshow(mask,cmap="gray")
#     str = 'R={}'.format(R) + " " + sampling_flag + ' p={}'.format(poly_degree)
#     plt.title(str)
#     #
#     # cnt += 1
#     # plt.subplot(R_vec.shape[0],2,cnt)
#     # plt.imshow(pdf,cmap="gray")
#     # plt.title('pdf')
#
#     sampling_flag = 'weak'
#     mask, pdf, poly_degree = gen_2D_var_dens_mask(R, imSize, sampling_flag,calib=calib)
#
#     cnt += 1
#     plt.subplot(R_vec.shape[0],2,cnt)
#     plt.imshow(mask,cmap="gray")
#     str = 'R={}'.format(R) + " " + sampling_flag + ' p={}'.format(poly_degree)
#     plt.title(str)
#
# #str = sampling_flag + ' var-dens'
# #plt.suptitle(str)
# plt.show()




###################################################################################################
#                                             1D var-dens (from 1c_MoDL_OLD_1D_sampling run10)
###################################################################################################
def gen_1D_var_dens_mask(R,imSize,sampling_flag,calib=[24, 24]):
    NX = imSize[0]
    NY = imSize[1]

    if sampling_flag == 'random':
        # 1D random - sampling columns
        tmp = np.random.randint(1, 1000, NX)
        inds = np.where(tmp <= (1000 / R))
        mask = np.zeros((NX, NY))
        mask[:, inds] = 1

        # fig = plt.figure()
        # plt.imshow(np.abs(mask), cmap="gray")
        # plt.title('mask_1D')
        # plt.show()


    if R >= 8:
        poly_degree = 10
    elif R == 7:
        poly_degree = 8
    elif R == 6:
        if sampling_flag == 'weak':
            poly_degree = 12
        elif sampling_flag == 'strong':
            poly_degree = 6
    elif R == 5:
        if sampling_flag == 'weak':
            poly_degree = 5
        # elif sampling_flag == 'strong':
        # TODO: implement this
    elif R == 4:
        if sampling_flag == 'weak':
            poly_degree = 10
        elif sampling_flag == 'strong':
            poly_degree = 4
            # print('poly_degree=', poly_degree)
    elif R == 3:
        poly_degree = 4
    elif R == 2:
        poly_degree = 4

    # # create a 1D pdf
    imSize_1D = np.array([imSize[1]])
    pdf_1D = genPDF(imSize_1D, poly_degree, 1 / R)
    mask_1D = genSampling(pdf_1D, iter=10, tol=60, calib=calib)

    pdf_1D = np.expand_dims(pdf_1D,0)
    mask_1D = np.expand_dims(mask_1D,0)

    pdf = np.repeat(pdf_1D,NY,0)
    mask = np.repeat(mask_1D,NY,0)

    return mask, pdf, poly_degree


#
# # ---------------------- 1D var-dens example & display ------------------------------
# NY = 256
# NX = 128
#
# calib = np.array([int(24*NY/256)])
#
# R_vec = np.array([2,4,6])
# imSize = np.array([NX,NY])
#
#
#
# cnt = 0
# fig = plt.figure()
# for r in range(R_vec.shape[0]):
#     R = R_vec[r]
#     print('R=',R)
#
#     sampling_flag = 'strong'
#     mask, pdf, poly_degree = gen_1D_var_dens_mask(R, imSize, sampling_flag,calib=calib)
#
#     cnt += 1
#     plt.subplot(R_vec.shape[0],2,cnt)
#     plt.imshow(mask,cmap="gray")
#     str = 'R={}'.format(R) + " " + sampling_flag + ' p={}'.format(poly_degree)
#     plt.title(str)
#     #
#     # cnt += 1
#     # plt.subplot(R_vec.shape[0],2,cnt)
#     # plt.imshow(pdf,cmap="gray")
#     # plt.title('pdf')
#
#     sampling_flag = 'weak'
#     mask, pdf, poly_degree = gen_1D_var_dens_mask(R, imSize, sampling_flag,calib=calib)
#
#     cnt += 1
#     plt.subplot(R_vec.shape[0],2,cnt)
#     plt.imshow(mask,cmap="gray")
#     str = 'R={}'.format(R) + " " + sampling_flag + ' p={}'.format(poly_degree)
#     plt.title(str)
#
# #str = sampling_flag + ' var-dens'
# #plt.suptitle(str)
# plt.show()
#










# TODO: replace all the next functions with functions from var_dens_NEW_funcs

#
# # =================================== 2D random/weak_var_dens/strong_var_dens sampling ===============================
# def gen_2D_samp_mask(R,imSize,sampling_flag,calib=[24, 24],mask_show_flag=0):
#     """This function takes the following inputs:
#     R = acceleration factor
#     imSize = a 2D numpy array that specifies the image size
#     sampling_flag: 0=random-uniform, 1=weak variable density, 2=strong variable density"""
#
#     if sampling_flag==0: # random uniform
#         #samp_str = 'random'
#         tmp = np.random.randint(1, 1000, NX * NY)
#         inds = np.where(tmp <= (1000 / R))
#         mask_1D = np.zeros(NX * NY)
#         mask_1D[inds] = 1
#         mask = mask_1D
#
#     elif sampling_flag==1:  # weak variable-density
#         #samp_str = 'weak var-dens'
#         poly_degree = 10
#         pdf = genPDF(imSize, poly_degree, 1 / R)
#         mask = genSampling(pdf, iter=10, tol=60)
#
#     elif sampling_flag == 2:  # strong variable-density
#         #samp_str = 'strong var-dens'
#         if R<=3:
#             poly_degree = 6
#         elif R==4:
#             poly_degree = 5
#         elif R==5:
#             poly_degree = 4.5
#         elif R>=6:
#             poly_degree = 4
#
#         pdf = genPDF(imSize, poly_degree, 1 / R)
#         mask = genSampling(pdf, iter=10, tol=60)
#
#     #print(samp_str)
#     return mask
#
#
#
#
#
# # ===================== 2D Variable-density Sampling  ============================
# # TODO: replace this func with the newer func "create_2D_var_dens_samp_mask", because the func here generates only strong variable-density patterns
# def create_samp_mask(R,imSize,calib=[24, 24],mask_show_flag=0):
#     print('gen PDF & sampling mask...')
#
#     # Here we define the variable "poly_degree", which controls the shape of the PDF.
#     # Strong variable density can be obtained with poly_degree =~5
#     # Weak variable density can be obtained with poly_degree = 50
#     # Extremeley weak variable density, which is almost (but not exactly) uniform random, is obtained with poly_dgree = 1000
#
#     print('inside sampling_funcs')
#     print('R=',R)
#
#     if R == 10:
#         poly_degree = 4.5
#     elif R == 8:
#         poly_degree = 4
#     elif R == 6:
#         poly_degree = 3
#     elif R == 5:
#         poly_degree = 2.5
#     elif R == 4:
#         poly_degree = 2
#     elif R == 3:
#         poly_degree = 1.5
#     elif R == 2:
#         poly_degree = 1.5
#     elif R > 10:
#         poly_degree = 10  # works OK for R=6,8,10 without calib, but results are unstable
#
#
#     pdf = genPDF(imSize, poly_degree, 1 / R)
#     mask = genSampling(pdf, iter=10, tol=60, calib=calib)
#     # mask = np.expand_dims(mask,axis=0)  # add coils dim to mask
#
#     if mask_show_flag==1:
#         # display sampling mask
#         fig = plt.figure()
#         plt.imshow(mask, cmap="gray")
#         plt.axis('off')
#         plt.title('R={}'.format(R))
#         plt.show()
#         fname = 'mask_R{}'.format(R)
#         fig.savefig(fname=fname)
#
#     elif mask_show_flag==2:  # display mask & pdf
#         # display sampling mask & PDF
#         fig = plt.figure()
#         plt.imshow(np.concatenate((mask,pdf), axis=1),cmap="gray")
#         plt.axis('off')
#         plt.title('sampling mask & pdf \n R={}'.format(R))
#         plt.show()
#         fname = 'mask_and_PDF_R{}'.format(R)
#         fig.savefig(fname=fname)
#
#
#
#     return mask, pdf
#
#
#
#
# # ===================== 1D Variable-density Sampling (based on Miki Lustig's Sparse MRI toolbox) ============================
#
# def create_samp_mask_1D(R,imSize,calib=[24,24],subsampled_axis=0,mask_show_flag=0):
#     # this function creates a 1D sampling mask for a 2D Cartesian k-space, this means that subsampling will
#     # be performed along one dimension only (i.e. full columns or full rows will be sampled).
#
#     print('gen PDF & sampling mask...')
#
#     # Here we define the variable "poly_degree", which controls the shape of the PDF.
#     # Strong variable density can be obtained with poly_degree =~5
#     # Weak variable density can be obtained with poly_degree = 50
#     # Extremeley weak variable density, which is almost (but not exactly) uniform random, is obtained with poly_dgree = 1000
#
#     if R >= 8:
#        poly_degree = 10
#     elif R == 7:
#        poly_degree = 8
#     elif R == 6:
#        poly_degree = 5.5
#     elif R == 5:
#        poly_degree = 5
#     elif R == 4:
#        poly_degree = 4.1
#     elif R == 3:
#        poly_degree = 4
#     elif R == 2:
#         poly_degree = 4
#
#     # if R == 10:
#     #     poly_degree = 4.5
#     # elif R == 8:
#     #     poly_degree = 4
#     # elif R == 6:
#     #     poly_degree = 3
#     # elif R == 5:
#     #     poly_degree = 2.5
#     # elif R == 4:
#     #     poly_degree = 2
#     # elif R == 3:
#     #     poly_degree = 1.5
#     # elif R == 2:
#     #     poly_degree = 1.5
#     # elif R > 10:
#     #     poly_degree = 10  # works OK for R=6,8,10 without calib, but results are unstable
#
#     if subsampled_axis==1:
#         calib_1D = calib[0]
#         imSize_1D = np.array([imSize[0]])
#     elif subsampled_axis==0:
#         calib_1D = calib[1]
#         imSize_1D = np.array([imSize[1]])
#
#
#     # create a 1D pdf
#     pdf = genPDF(imSize_1D, poly_degree, 1 / R)
#
#     # create a 1D mask
#     mask_1D = genSampling(pdf, iter=10, tol=60, calib=calib_1D)
#
#     # expand the mask dims to 2D, to
#     if subsampled_axis == 1:
#         mask_1D = np.expand_dims(mask_1D, axis=1)
#         mask = np.tile(mask_1D,(1,imSize[1]))
#         pdf = np.expand_dims(pdf, axis=1)
#         pdf4plot = np.tile(pdf,(1,imSize[1]))
#     elif subsampled_axis == 0:
#         mask = np.tile(mask_1D,(imSize[0],1))
#         pdf4plot = np.tile(pdf,(imSize[0], 1))
#
#
#     if mask_show_flag==1:
#         # display sampling mask
#         fig = plt.figure()
#         plt.imshow(mask, cmap="gray")
#         plt.axis('off')
#         plt.title('R={}'.format(R))
#         plt.show()
#         fname = 'mask_1D_R{}'.format(R)
#         fig.savefig(fname=fname)
#
#     elif mask_show_flag==2:  # display mask & pdf
#         # display sampling mask & PDF
#         fig = plt.figure()
#         plt.imshow(np.concatenate((mask,pdf4plot), axis=1),cmap="gray")
#         plt.axis('off')
#         plt.title('sampling mask & pdf \n R={}'.format(R))
#         plt.show()
#         fname = 'mask_and_PDF_1D_R{}'.format(R)
#         fig.savefig(fname=fname)
#
#
#
#     return mask, pdf



# ############### Example for calling genPDF, genSampling  ###########################
# imSize=np.array([128,128])
# #imSize=np.array([128])

# R_wanted = 6
# pctg = 1/R_wanted
# p=3


# pdf,_ = genPDF(imSize,p,pctg)
# mask = genSampling(pdf,iter=10,tol=60)

# fig = plt.figure()
# plt.imshow(abs(pdf))

# fig = plt.figure()
# plt.imshow(mask)





#
#
# # ===================== 2D Poisson Disk Sampling (based on Frank Ong's Sigpy package) ============================
#
# def gen_2D_rand_samp_mask(Scoils, Sx, Sy,accel,calib,seed_val,alpha=1,crop_corner=True):
#     # mask_poisson = sp.mri.poisson([Sx, Sy], accel, calib=calib,seed=seed_val)  # create 2D poisson sampling mask
#     mask_poisson = new_poisson([Sx, Sy], accel, calib=calib,crop_corner=crop_corner,seed=seed_val,alpha=alpha)  # create 2D poisson sampling mask and control the density of the var-dens mask using alpha
#
#     mask = np.expand_dims(mask_poisson,axis=0)   # add the coils dimension
#     mask = np.tile(mask, (Scoils, 1, 1))         # replicagte sampling mask along coils dimension
#
#     # calc R (actual reduction factor)
#     mask_reshaped = np.reshape(mask_poisson,(1,-1))
#     nnz = np.count_nonzero(mask_reshaped)
#     actual_accel = mask_reshaped.shape[1]/nnz
#     return mask,actual_accel
#
#
# def gen_1D_rand_samp_mask(Scoils, Sx, Sy,accel,calib,seed_val):
#     mask_poisson = sp.mri.poisson([Sx, Sy], accel, calib=calib,seed=seed_val)  # create 2D poisson sampling mask
#     idx_mid = math.floor(Sx/2)
#     vec = mask_poisson[idx_mid,]  # take the middle row from the 2D mask
#     mask_1D_1coil = np.matlib.repmat(vec,Sy,1)  # replicate this row
#     mask = np.expand_dims(mask_1D_1coil,axis=0)   # add the coils dimension
#     mask = np.tile(mask, (Scoils, 1, 1))         # replicagte sampling mask along coils dimension
#
#     # calc R (reduction factor)
#     mask_1D_reshaped = np.reshape(mask_1D_1coil,(1,-1))
#     nnz = np.count_nonzero(mask_1D_reshaped)
#     R = mask_1D_reshaped.shape[1]/nnz
#     return mask, R
