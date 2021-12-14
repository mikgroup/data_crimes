import numpy as np
import sigpy as sp
import math
from subtle_data_crimes.functions import new_poisson
import matplotlib.pyplot as plt


# ===================== 2D Variable-density Sampling (based on Miki Lustig's Sparse MRI toolbox) ============================

def create_samp_mask(R, imSize, calib=[24, 24], mask_show_flag=0):
    print('gen PDF & sampling mask...')

    # Here we define the variable "poly_degree", which controls the shape of the PDF.
    # Strong variable density can be obtained with poly_degree =~5
    # Weak variable density can be obtained with poly_degree = 50
    # Extremeley weak variable density, which is almost (but not exactly) uniform random, is obtained with poly_dgree = 1000

    if R == 10:
        poly_degree = 4.5
    elif R == 8:
        poly_degree = 4
    elif R == 6:
        poly_degree = 3
    elif R == 5:
        poly_degree = 2.5
    elif R == 4:
        poly_degree = 2
    elif R == 3:
        poly_degree = 1.5
    elif R == 2:
        poly_degree = 1.5
    elif R > 10:
        poly_degree = 10  # works OK for R=6,8,10 without calib, but results are unstable

    pdf = genPDF(imSize, poly_degree, 1 / R)
    mask = genSampling(pdf, iter=10, tol=60, calib=calib)
    # mask = np.expand_dims(mask,axis=0)  # add coils dim to mask

    if mask_show_flag == 1:
        # display sampling mask
        fig = plt.figure()
        plt.imshow(mask, cmap="gray")
        plt.axis('off')
        plt.title('R={}'.format(R))
        plt.show()
        fname = 'mask_R{}'.format(R)
        fig.savefig(fname=fname)

    elif mask_show_flag == 2:  # display mask & pdf
        # display sampling mask & PDF
        fig = plt.figure()
        plt.imshow(np.concatenate((mask, pdf), axis=1), cmap="gray")
        plt.axis('off')
        plt.title('sampling mask & pdf \n R={}'.format(R))
        plt.show()
        fname = 'mask_and_PDF_R{}'.format(R)
        fig.savefig(fname=fname)

    return mask, pdf


############### genPDF  ###########################
def genPDF(imSize, p, pctg, distType=2, radius=0, disp=0):
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
        # sx = imSize(1);
        # sy = imSize(2);
        # PCTG = floor(pctg*sx*sy);
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
        r = np.abs(np.linspace(-1, 1, sx))

        # create PDF
    idx = np.where(r < radius)
    pdf = (1 - r) ** p + val
    pdf[pdf > 1] = 1
    pdf[idx] = 1

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

    # print('inside genSampling')
    print('calib=', calib)

    pdf[pdf > 1] = 1
    K = np.sum(pdf[::])
    minIntr = np.array([1e99])
    minIntrVec = np.zeros(pdf.shape)

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

    return mask

# ############### Example for calling the above two functions:  ###########################
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



