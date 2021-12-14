'''
This code takes raw data (knee Proton Density data from FastMRI), extracts Fat-Sat data (i.e. data with label
'CORPDFS_FBK'), and creates processed datasets for the DL experiments of subtle data crime 1 that are shown
in Figure 4 and in Figure 8a. Notice that only Fat-Sat data is used for these examples, because pathology is
usually observed in such scans.

The code also splits the data into training, validation and test sets. The two images that contain pathology
and were displayed in the paper are saved as separate test sets.

NOTICE: you should update the following variables to YOUR desired path (see first code cell):
FastMRI_train_folder    # input folder
FastMRI_val_folder      # input folder
FatSat_processed_data_folder  # desired output folder

(c) Efrat Shimron, UC Berkeley, 2021.
'''


# add folder above (while running on mikQNAP):
#sys.path.append("/mikQNAP/efrat/1_inverse_crimes/1_mirror_PyCharm_CS_MoDL_merged/SubtleCrimesRepo/")

import numpy as np
import os
import h5py
import sigpy as sp
from subtle_data_crimes.functions.utils import zpad_merge_scale
import matplotlib.pyplot as plt



######################################################################################################################
#                                             Set data paths to YOUR paths
######################################################################################################################

# Download the data from FastMRI and set the following paths to YOUR paths:
FastMRI_train_folder = "/mikQNAP/NYU_knee_data/multicoil_train/"
FastMRI_val_folder = "/mikQNAP/NYU_knee_data/multicoil_val/"

# Set the next path to YOUR desired output path.
FatSat_processed_data_folder = "/mikQNAP/NYU_knee_data/efrat/subtle_inv_crimes_zpad_data_v19_FatSatPD/"

######################################################################################################################
#                                                   Prep data
######################################################################################################################

N_train_datasets = 300
N_val_datasets = 10
N_test_datasets = 7
pad_ratio_vec = np.array([1,1.25,1.5,1.75,2])  # Define the desired kspace zero-padding ratio values


#################################### data split ###############################################
# NOTICE: the original FastMRI database is divided to train/val/test data.
# The train & val datasets are fully-sampled, but the test data is subsampled.
# Since we need fully-sampled data for training the models, we will split the FastMRI train data into train & test data.
# Val data will remain val data and will be used only for hyper parameter calibration.

# create file lists
FastMRI_train_folder_files = os.listdir(FastMRI_train_folder )
FastMRI_val_folder_files = os.listdir(FastMRI_val_folder )

# Split the LISTS of FastMRI training files into two lists: train & test files.
train_files_list = FastMRI_train_folder_files[0:800] # There are 973 scans in total. We reserve the last 173 scans for test data.
test_files_list =  FastMRI_train_folder_files[800::]

val_files_list = FastMRI_val_folder_files

# Here we remove the 2 pathology examples from the lists of training files. The pathologies will be used separately, as test cases.
if 'file1000425.h5' in train_files_list:
    print(f'pathology 1 is in train_files_list')
    train_files_list.remove('file1000425.h5')
    print('removed')

if 'file1002455.h5' in train_files_list:
    print(f'pathology 2 is in train_files_list')
    train_files_list.remove('file1002455.h5')
    print('removed')


N_available_test_scans = len(test_files_list)
N_available_train_scans = len(train_files_list)
N_available_val_scans = len(val_files_list)

print('N_available_train_scans=',N_available_train_scans)
print('N_available_val_scans=',N_available_val_scans)
print('N_available_test_scans=',N_available_test_scans)


# initialize counters
N_PD_scans_train = 0
N_PD_scans_val = 0
N_PD_scans_test = 0

N_imgs_train = 0
N_imgs_val = 0
N_imgs_test = 0

n_imgs = 0

#########################################################################################


for data_i in range(5):  # 0 = train, 1 = val, 2 = test, 3 = pathology case I, 4 = pathology case II
    if data_i == 0:  # prep training data
        original_files_folder = FastMRI_train_folder
        original_files_list = train_files_list
        N_wanted_scans = N_train_datasets  # the first 10 scans will be used for test data. Each scan contains 20-30 slices so it will give us 200-300 test images.
        data_type = 'train'
        im_type_vec = np.array([0, 1])  # 0 = full images, 1 = blocks
    elif data_i == 1:  # prep validation data
        original_files_folder = FastMRI_val_folder
        original_files_list = val_files_list
        N_wanted_scans = N_val_datasets
        data_type = 'val'
        im_type_vec = np.array([0, 1])  # 0 = full images, 1 = blocks
    elif data_i == 2: # prep test data
        original_files_folder = FastMRI_train_folder     #NOTICE: as explained above, we divided the LIST of files in the FastMRI train folder into lists of training & test files
        original_files_list = test_files_list
        N_wanted_scans = N_test_datasets
        data_type = 'test'
        im_type_vec = np.array([0])  # inference of the deep-learning method is done for full-size images, so block are not needed
    elif data_i == 3:  # prep 1st pathology example
        # NOTE: THIS SCAN IS FAT-SAT-PD
        original_files_folder = FastMRI_train_folder
        original_files_list = ['file1000425.h5']
        N_wanted_scans = np.array([1])
        data_type = 'pathology_1'
        im_type_vec = np.array([0])  # inference of the deep-learning method is done for full-size images, so block are not needed
    elif data_i == 4:  # prep 2nd pathology example
        # NOTE: THIS SCAN IS FAT-SAT-PD
        original_files_folder = FastMRI_train_folder
        original_files_list = ['file1002455.h5']
        N_wanted_scans = np.array([1])
        data_type = 'pathology_2'
        im_type_vec = np.array([0])  # inference of the deep-learning method is done for full-size images, so block are not needed



    print('=======================================================')
    print(f'                      {data_type} data      ')
    print('=======================================================')

    N_available_scans = len(original_files_list)

    # counters for loaded/processed scans. Each scan has about 20-30 slices, but we throw away the 7 edge slices on each side
    t = 0  # counts files for checking and processing
    n_PD_scans = 0


    while n_PD_scans < N_wanted_scans:

            print(' === loading h5 file {} === '.format(t))
            # Load k-space data
            filename_h5 = original_files_folder + original_files_list[t]

            print('t=',t)
            print('filename_h5=',filename_h5)
            f = h5py.File(filename_h5, 'r')

            t += 1  # updated the number of LOADED scans

            scan_label = f.attrs['acquisition']
            kspace_orig = np.array(f["kspace"])

            # print some info about the scan
            if (scan_label != 'CORPDFS_FBK'):
                print('scan label is not FatSatPD')

            #if (kspace_orig.shape[2] != 640) | (kspace_orig.shape[3] != 372):
            #    print('scan is not 640x372 pixels')


            #if (kspace_orig.shape[2] == 640) & (kspace_orig.shape[3] == 372) & (scan_label == 'CORPDFS_FBK'):
            if (scan_label == 'CORPDFS_FBK'):
                print('valid slice - FatSatPD scan')
                kspace = kspace_orig
                n_PD_scans += 1

                N_slices = kspace.shape[0]  # number of slices
                NC = kspace.shape[1]
                NX = kspace.shape[2]
                NY = kspace.shape[3]

                # define margins - to avoid choosing blocks that are at the image's edges and contain only noise
                x_margin = 0.15 * NX
                y_margin = 0.15 * NY

                ################ image domain #################
                # throw out the 8 edge slices on each side of the scan, because they contain mostly noise.
                slices_to_store_inds = np.arange(8, (N_slices - 8), 1)

                if data_type == 'val':
                    # for validation data we arbitrarily choose a SINGLE slice from the scan, i.e. we store only one slice per subject.
                    slices_to_store_inds = np.random.randint(8,(N_slices - 8),1)
                elif data_type == 'pathology_1':
                    #slices_to_store_inds = np.arange(21, 24, 1)  # take slices only around slice 22 (which contains the pathology)
                    slices_to_store_inds = np.arange(0,N_slices,1)
                elif data_type == 'pathology_2':
                    #slices_to_store_inds = np.arange(26, 27,
                    #                                1)  # take slices only around slice 26 (which contains the pathology)
                    slices_to_store_inds = np.arange(0,N_slices,1)

                n_slices_to_store = slices_to_store_inds.shape[0]


                n_imgs += n_slices_to_store  # update counter

                print(f'n_PD_scans (valid scans) = {n_PD_scans}')
                print(f'n_imgs (saved slices)={n_imgs}')


                # NOTICE: we must iterate on pad_ratio first (before iterating on slices), because this codes saves
                # multi-slice padded data. All the slices must be padded with the same zero-padding rate.

                for im_type_i in range(im_type_vec.shape[0]):
                    if im_type_vec[im_type_i] == 0:
                        create_blocks_flag = 0  # process full-size images
                        im_type_str = 'full_im'
                        print('----------------------------')
                        print(f'   prep full-size images   ')
                        print('----------------------------')


                    elif im_type_vec[im_type_i] == 1:
                        create_blocks_flag = 1  # blocks are needed for training the deep-learning method, since zero-padded full-size imgs are too large to fit on GPU
                        im_type_str = 'blocks'
                        print('----------------------------')
                        print(f'   prep blocks   ')
                        print('----------------------------')

                    for i, pad_ratio in enumerate(pad_ratio_vec):
                        print(' --- padding ratio {} ---'.format(pad_ratio))

                        for s_i in range(n_slices_to_store):
                            # print(s_i)
                            s = slices_to_store_inds[s_i]
                            print(f' slice {s}')

                            ksp_block_multicoil = kspace[s, :, :, :].squeeze()

                            im_mag_scaled = zpad_merge_scale(ksp_block_multicoil, pad_ratio)

                            if (data_type=='pathology_1') | (data_type=='pathology_2'):
                                fig = plt.figure()
                                plt.imshow(np.rot90(im_mag_scaled,2),cmap="gray")
                                plt.show()

                            # print('shape before padding:')
                            # print(ksp_block_multicoil.shape)
                            # print('shape after padding:')
                            # print(im_mag_scaled.shape)

                            if create_blocks_flag == 0:  # process full-fove images (merge multi-coil, zero-padd, scale, and store the data)

                                img = im_mag_scaled

                                # print('stored image size:')
                                # print(img.shape)


                            else:  # extract blocks & process them (merge multi-coil, zero-padd, scale, and store the data)

                                NX_padded = int(NX * pad_ratio)
                                NY_padded = int(NY * pad_ratio)

                                # block size = 0.2*im_size
                                NX_block = int(0.2 * NX_padded)
                                NY_block = int(0.2 * NY_padded)

                                x_max_offset = NX_padded - NX_block - x_margin - 25
                                y_max_offset = NY_padded - NY_block - y_margin - 25

                                assert x_max_offset > x_margin, 'x_max_offset<y_margin'
                                assert y_max_offset > y_margin, 'y_max_offset<y_margin'

                                x_mid = int(NX / 2)
                                y_mid = int(NY / 2)

                                valid_block_flag = 0

                                # Next we extract a block from the image and check that it contains some signal, i.e. that it's not empty.
                                # If the block is "empty" (i.e. contains mostly noise) we will try to extract another block. Max 50 trials.
                                # If after 50 trials the block is still not good we'll store it anyway.
                                trial_cnt = 0
                                while (valid_block_flag == 0) & (trial_cnt <= 50):
                                    trial_cnt += 1

                                    x_i = np.random.randint(x_margin, x_max_offset, size=1)  # offset in x axis
                                    y_i = np.random.randint(y_margin, y_max_offset, size=1)  # offset in x axis
                                    im_block = im_mag_scaled[x_i[0]:(x_i[0] + NX_block), y_i[0]:(y_i[0] + NY_block)]

                                    if np.max(im_block) > 0.5 * np.max(im_mag_scaled):
                                        # print('block is OK')
                                        valid_block_flag = 1
                                    else:
                                        print('block contains mostly noise - not good - extract a different one')

                                # print('block size:')
                                # print(im_block.shape)

                                img = im_block

                            ################### compute k-space of the zero-padded image/block #####################

                            kspace_slice = sp.fft(img)

                            if (s_i == 0) & (n_PD_scans == 1):
                                fig = plt.figure()
                                plt.subplot(1, 3, 1)
                                plt.imshow(np.abs(im_mag_scaled), cmap="gray")
                                plt.colorbar(shrink=0.4)
                                plt.title('im_mag_scaled')

                                plt.subplot(1, 3, 2)
                                plt.imshow(np.abs(img), cmap="gray")
                                plt.colorbar(shrink=0.4)
                                plt.title('img / block')

                                plt.subplot(1, 3, 3)
                                plt.imshow(np.log(np.abs(kspace_slice)), cmap="gray")
                                plt.title('ksp')
                                plt.colorbar(shrink=0.4)
                                plt.suptitle(
                                    f'{data_type} data; {im_type_str}; pad x{pad_ratio}; scan {n_PD_scans} slice {s}')
                                plt.show()

                            # initialize arrays
                            if s_i == 0:
                                kspace_coils_merged_padded_multi_slice = np.empty(
                                    (n_slices_to_store, kspace_slice.shape[0],
                                     kspace_slice.shape[1]),
                                    dtype='complex64')
                                images_coils_merged_padded_multi_slice = np.empty(
                                    (n_slices_to_store, kspace_slice.shape[0],
                                     kspace_slice.shape[1]),
                                    dtype='complex64')

                            # save slice to arrays
                            images_coils_merged_padded_multi_slice[s_i, :, :] = img
                            kspace_coils_merged_padded_multi_slice[s_i, :, :] = kspace_slice

                        ################## saving -  all slices extracted from a single h5 file are saved together  =================

                        out_folder = FatSat_processed_data_folder + data_type + "/pad_" + str(
                            int(100 * pad_ratio)) + "/" + im_type_str + "/"


                        if not os.path.exists(out_folder):
                            os.makedirs(out_folder)

                        # Save data in h5 file
                        h5f = h5py.File(out_folder + '%d.h5' % (n_PD_scans), 'w')
                        h5f.create_dataset('kspace', data=kspace_coils_merged_padded_multi_slice)
                        h5f.create_dataset('reconstruction', data=images_coils_merged_padded_multi_slice)
                        h5f.close()

            if (t == (N_available_scans - 1)) | (n_PD_scans == N_wanted_scans):
                break


    if data_i == 0:   # train data
        N_PD_scans_train = n_PD_scans
        N_imgs_train = n_imgs
    elif data_i == 1: # val data
        N_PD_scans_val = n_PD_scans
        N_imgs_val = n_imgs
    elif data_i == 2: # test data
        N_PD_scans_test = n_PD_scans
        N_imgs_test = n_imgs
    #elif data_i == 3:  # pathology 1
    #elif data_i == 4:  # pathology 2



########### save meta-data ##################

print('N_PD_scans_train: ', N_PD_scans_train)
print('N_PD_scans_val: ', N_PD_scans_val)
print('N_PD_scans_test: ', N_PD_scans_test)
print('N_imgs_train: ', N_imgs_train)
print('N_imgs_val: ', N_imgs_val)
print('N_imgs_test: ', N_imgs_test)


meta_filename = FatSat_processed_data_folder + 'num_stored_slices'
np.savez(meta_filename, N_PD_scans_train=N_PD_scans_train, N_PD_scans_val=N_PD_scans_val, N_PD_scans_test=N_PD_scans_test,N_imgs_train=N_imgs_train,N_imgs_val=N_imgs_val,N_imgs_test=N_imgs_test)







