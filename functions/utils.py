import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ----------------- functions for subtle_inverse_crime_I -----------------

def calc_pad_half(N_original, pad_ratio):
    N_tot = N_original * pad_ratio  # this will be the total k-space size
    diff = np.ceil(N_tot - N_original)  # this is the total padding length
    pad_size_vec = diff.astype(int)  # convert from float to int
    pad_half_vec = (pad_size_vec / 2).astype(
        int)  # kspace will be padded with "pad_half" from each size, so the total padding length is padd_size

    return pad_half_vec, N_tot


def pad_multicoil_ksp(ksp_slice, pad_ratio):
    ''' This function applies zero-padding to multi-coil k-space data of a single slice.
    The zero-padding is applied to each coil separately. The pad_ratio determines the zero padding
    factor, i.e. if the initial ksp size is NxN, after padding it will be of size (N*pad_ratio)x(N*pad_ratio)

    Inputs:
    ksp_slice - dimensions (N_coils, NX, NY),
    pad_ratio - a scalar

    Output:
    ksp_slice_padded - dimensions (Ncoils, NX*pad_ratio, NY*pad_ratio)
    '''
    N_original_dim1 = ksp_slice.shape[1]
    N_original_dim2 = ksp_slice.shape[2]

    pad_half_dim1, N_tot_dim1 = calc_pad_half(N_original_dim1, pad_ratio)
    pad_half_dim2, N_tot_dim2 = calc_pad_half(N_original_dim2, pad_ratio)

    # zero-pad k-space - for every coil separately
    padding_lengths = ((0, 0), (pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))
    # padding_lengths_yz = ((pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))
    ksp_slice_padded = np.pad(ksp_slice, padding_lengths, mode='constant', constant_values=(0, 0))

    return ksp_slice_padded


def merge_multicoil_data(ksp_slice_all_coils):
    # This function receives *complex* multi-coil data and merges it into a single *magtniude* image.
    # The merging method: Square Root Sum of Squares.
    # Expected input dimensions: [Ncoils,Sx,Sy]
    # Notice: the input should contain data of a *single slice*

    ksp_slice_data = ksp_slice_all_coils[:, :, :].squeeze()  # eliminate slice dimension

    Nc, Sx, Sy = ksp_slice_data.shape

    squares_data = np.empty([Nc, Sx, Sy])

    # fig, ax = plt.subplots(nrows=1,ncols=4)

    for n in range(Nc):
        ksp_1coil = ksp_slice_data[n, :, :].squeeze()  # kspace data of 1 coil
        im_1coil_complex = np.fft.fftshift(np.fft.ifft2(ksp_1coil))  # complex image of 1 coil
        im_square = np.abs(im_1coil_complex) ** 2
        squares_data[n, :, :] = im_square
        # ax[n].imshow(np.abs(im_square)) #,cmap="gray"

    SOS = np.sum(squares_data, axis=0)

    # fig = plt.figure()
    # plt.imshow(SOS)

    mag_im = np.sqrt(SOS)

    return mag_im


# --------------------- JPEG compression ----------------------------

def JPEG_compression(im_mag, quality_val=100):
    # inputs:
    # im_mag - a magnitude image
    # quality_val - a paramter that controls the JPEG compression quality.
    #      Examples:
    #      quality_val=100 produces lossless compression
    #      quality_val=75  is JPEG's default
    #      quality_val=10  produces highly lossy compression

    # normalize the range to [0,1]
    im_mag = im_mag / np.max(im_mag)

    # normalize the range to [0,255]
    scale_factor = 255  # 255 / np.max(im_mag)
    im_mag_scaled = im_mag * scale_factor
    im_mag_uint8 = (im_mag_scaled).astype('uint8')  # prepare for JPEG compression
    im_PIL = Image.fromarray(im_mag_uint8)  # apply lossy compression by saving in JPEG format

    # compress the image by saving it in JPEG format
    im_comp_filename = 'im_compressed_q{}.jpg'.format(quality_val)
    im_PIL.save(im_comp_filename, format='JPEG', quality=quality_val, subsampling=0)

    # load the JPEG image
    im_compressed = np.asarray(Image.open(im_comp_filename))  # convert from pillow format to numpy format
    im_compressed = im_compressed / scale_factor
    # print('max(im_compressed) after scaling:', np.max(np.abs(im_compressed)))

    return im_compressed


# -------------------- calc_R_actual -------------------------------

def calc_R_actual(mask):
    mask_1D = np.reshape(mask, (1, -1))
    R_actual = mask_1D.shape[1] / np.count_nonzero(mask_1D)

    return R_actual


# ------------ save as png image -------------------
def save_as_png(im_orig, filename):
    rescaled = (255.0 / im_orig.max() * (im_orig - im_orig.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    filename_str = filename + '.png'
    im.save(filename_str)
