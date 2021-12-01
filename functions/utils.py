import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


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
    # The merging method: square Root Sum of Squares (RSS)
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

    SOS = np.sum(squares_data, axis=0)

    mag_im = np.sqrt(SOS)

    return mag_im


# ---------------------------------------------------------------------


def zpad_merge_scale(ksp_block_multicoil, pad_ratio):
    ''' inputs:
        kspace - numpy array of size [Ncoils, NX, NY]
        pad_ratio - numpy array (scalar) that denotes the desired padding ratio
        '''

    NX = ksp_block_multicoil.shape[1]
    NY = ksp_block_multicoil.shape[2]

    ############## zero-pad, merge & save ###################

    pad_half_dim1, N_tot_dim1 = calc_pad_half(NX, pad_ratio)
    pad_half_dim2, N_tot_dim2 = calc_pad_half(NY, pad_ratio)

    padding_lengths = ((0, 0), (pad_half_dim1, pad_half_dim1), (pad_half_dim2, pad_half_dim2))

    # NX_padded = int(NX * pad_ratio)
    # NY_padded = int(NY * pad_ratio)

    ksp_block_multicoil_padded = np.pad(ksp_block_multicoil, padding_lengths, mode='constant',
                                        constant_values=(0, 0))

    # compute a single *magnitude* image from the data
    im_mag = merge_multicoil_data(ksp_block_multicoil_padded)

    # intensity normalization by the 98% percentile
    magnitude_vals = im_mag.reshape(-1)
    mag_vals_sorted = np.sort(magnitude_vals)
    k = int(round(0.98 * magnitude_vals.shape[0]))
    scale_factor = mag_vals_sorted[k]
    im_mag_scaled = im_mag / scale_factor

    return im_mag_scaled


# --------------------- JPEG compression ----------------------------

def JPEG_compression(im_mag, quality_val=100):
    # inputs:
    # im_mag - a magnitude image
    # quality_val - a paramter that controls the JPEG compression quality:
    #      quality_val=100 yields the mininal compression
    #      quality_val=75 is JPEG's default
    #      quality_val=5 (or close to 0) yields an extreme compression

    # normalize the range to [0,1]
    im_mag = im_mag / np.max(im_mag)

    # normalize the range to [0,255]
    scale_factor = 255  # 255 / np.max(im_mag)
    im_mag_scaled = im_mag * scale_factor
    im_mag_uint8 = (im_mag_scaled).astype('uint8')  # prepare for JPEG compression
    im_PIL = Image.fromarray(im_mag_uint8)  # apply lossy compression by saving in JPEG format

    jpeg_figs_folder = "jpeg_imgs_q" + str(quality_val) + "/"
    if not os.path.exists(jpeg_figs_folder):
        os.makedirs(jpeg_figs_folder)

    # compress the image by saving it in JPEG format
    im_comp_filename = jpeg_figs_folder + 'im_compressed_q{}.jpg'.format(quality_val)
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


# ------------ save as png image
def save_as_png(im_orig, filename):
    rescaled = (255.0 / im_orig.max() * (im_orig - im_orig.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    filename_str = filename + '.png'
    im.save(filename_str)


# ------------------- extract block ---------------------------------
def extract_block(im, block_asp_ratio_x, block_asp_ratio_y, x_margin, y_margin):
    NX = im.shape[0]
    NY = im.shape[1]

    NX_block = int(block_asp_ratio_x * NX)
    NY_block = int(block_asp_ratio_y * NY)

    x_max_offset = NX - NX_block - x_margin - 25
    y_max_offset = NY - NY_block - y_margin - 25

    assert x_max_offset > x_margin, 'x_max_offset<y_margin'
    assert y_max_offset > y_margin, 'y_max_offset<y_margin'

    valid_block_flag = 0

    # Next we extract a block from the image and check that it contains some signal, i.e. that it's not empty.
    # If the block is "empty" (i.e. contains mostly noise) we will try to extract another block. Max 50 trials.
    # If after 50 trials the block is still not good we'll store it anyway.
    trial_cnt = 0
    while (valid_block_flag == 0) & (trial_cnt <= 50):
        trial_cnt += 1

        x_i = np.random.randint(x_margin, x_max_offset, size=1)  # offset in x axis
        y_i = np.random.randint(y_margin, y_max_offset, size=1)  # offset in x axis
        im_block = im[x_i[0]:(x_i[0] + NX_block), y_i[0]:(y_i[0] + NY_block)]

        if np.max(np.abs(im_block)) > 0.5 * np.max(np.abs(im)):
            # print('block is OK')
            valid_block_flag = 1
        else:
            print('block contains mostly noise - not good - extract a different one')

    # print('block size:')
    # print(im_block.shape)

    return im_block