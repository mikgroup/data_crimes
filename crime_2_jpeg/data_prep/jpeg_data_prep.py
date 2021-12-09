'''
This code demonstrates prepares processed version of the raw data (knee Proton Density data from FastMRI) for the
CS, DictL and DL experiments of subtle data crime 2.

Once you define the desired data path in the basic_out_folder here, make sure to update it in the scripts of the above experiments.

(c) Efrat Shimron (UC Berkeley, 2021)
'''

# TODO: remove the option for a small dataset

import sys
import numpy as np
import os
import h5py
import sigpy as sp
from functions.utils import merge_multicoil_data, JPEG_compression, extract_block
import matplotlib.pyplot as plt



# Set this to your path - where to save the output files:
basic_out_folder = "/mikQNAP/NYU_knee_data/multicoil_efrat/5_JPEG_compressed_data"


######################################################################################################################
#                                                   Prep data
######################################################################################################################


#q_vec = np.array([999])
# q_vec = np.array([100])
q_vec = np.array([20,50,75,100,999])

print('q_vec: ',q_vec)

create_small_dataset_flag = 0  # 0 = large database, 1 = small database, 2 = large database consisting of images of size 372 only

if create_small_dataset_flag == 1:
    print('creating a SMALL dataset')
    N_train_datasets = 2  # num scans
    N_val_datasets = 3
    N_test_datasets = 3

elif create_small_dataset_flag == 0:

    print('creating a LARGE dataset')
    ############# large dataset #############
    N_train_datasets = 80
    N_val_datasets = 10
    N_test_datasets = 7

#################################### data split ###############################################
# NOTICE: the original FastMRI database is divided to train/val/test data.
# The train & val datasets are fully-sampled, but the test data is subsampled.
# Since we need fully-sampled data for training the models, we will split the FastMRI train data into train & test data.
# Val data will remain val data.

FastMRI_train_folder = "/mikQNAP/NYU_knee_data/multicoil_train/"
FastMRI_val_folder = "/mikQNAP/NYU_knee_data/multicoil_val/"
# home_dir_test = "/mikQNAP/NYU_knee_data/multicoil_test/"

FastMRI_train_folder_files = os.listdir("/mikQNAP/NYU_knee_data/multicoil_train/")
FastMRI_val_folder_files = os.listdir("/mikQNAP/NYU_knee_data/multicoil_val/")
# home_dir_test_files = os.listdir("/mikQNAP/NYU_knee_data/multicoil_test/")

# Split the LISTS of FastMRI training files into two lists: train & test files.
train_files_list = FastMRI_train_folder_files[
                   0:800]  # There are 973 scans in total. We reserve the last 173 scans for test data.
test_files_list = FastMRI_train_folder_files[800::]

val_files_list = FastMRI_val_folder_files

# Remove the 2 pathology examples from the lists of training files. The pathologies will be used separately, as test cases.

if 'file1000425.h5' in train_files_list:
    print(f'pathology 1 is in train_files_list')
    train_files_list.remove('file1000425.h5')
    print('removed')

if 'file1002455.h5' in train_files_list:
    print(f'pathology 2 is in train_files_list')
    train_files_list.remove('file1002455.h5')
    print('removed')

# N_available_test_scans = len(test_files_list)
# N_available_train_scans = len(train_files_list)
# N_available_val_scans = len(val_files_list)

# print('N_available_train_scans=',N_available_train_scans)
# print('N_available_val_scans=',N_available_val_scans)
# print('N_available_test_scans=',N_available_test_scans)


for q_i in range(q_vec.shape[0]):
    q = int(q_vec[q_i])

    N_PD_scans_train = 0
    N_PD_scans_val = 0
    N_PD_scans_test = 0

    N_imgs_train = 0
    N_imgs_val = 0
    N_imgs_test = 0

    n_imgs = 0

    print(f'############################ q ={q} ##########################')

    for data_i in np.array([2]): #range(2):  # range(3):  # 0 = train, 1 = val, 2 = test
        if data_i == 0:  # prep training data
            original_files_folder = FastMRI_train_folder
            original_files_list = train_files_list
            N_wanted_scans = N_train_datasets  # the first 10 scans will be used for test data. Each scan contains 20-30 slices so it will give us 200-300 test images.
            data_type = 'train'
            im_type_vec = np.array([0,1])  # 0 = full images, 1 = blocks
        elif data_i == 1:  # prep validation data
            original_files_folder = FastMRI_val_folder
            original_files_list = val_files_list
            # home_dir = FastMRI_val_folder
            # home_dir_files = FastMRI_val_folder_files
            N_wanted_scans = N_val_datasets
            data_type = 'val'
            im_type_vec = np.array([0, 1])  # 0 = full images, 1 = blocks
        elif data_i == 2:  # prep test data
            original_files_folder = FastMRI_train_folder  # NOTICE: as explained above, we divided the LIST of files in the FastMRI train folder into lists of training & test files
            original_files_list = test_files_list
            # home_dir = home_dir_test
            # home_dir_files = home_dir_test_files
            N_wanted_scans = N_test_datasets
            data_type = 'test'
            im_type_vec = np.array(
                [0])  # inference of the deep-learning method is done for full-size images, so block are not needed
        elif data_i == 3:  # prep 1st pathology example - it's taken from the training set
            # NOTE: THIS SCAN IS NOT Proton-Density, so the network wasn't trained for it.
            original_files_folder = FastMRI_train_folder
            original_files_list = ['file1000425.h5']
            N_wanted_scans = np.array([1])
            data_type = 'pathology_1'
            im_type_vec = np.array(
                [0])  # inference of the deep-learning method is done for full-size images, so block are not needed
        elif data_i == 4:  # prep 1st pathology example - it's taken from the training set
            # NOTE: THIS SCAN IS NOT Proton-Density, so the network wasn't trained for it.
            original_files_folder = FastMRI_train_folder
            original_files_list = ['file1002455.h5']
            N_wanted_scans = np.array([1])
            data_type = 'pathology_2'
            im_type_vec = np.array(
                [0])  # inference of the deep-learning method is done for full-size images, so block are not needed

        print('=======================================================')
        print(f'                      {data_type} data      ')
        print('=======================================================')

        N_available_scans = len(original_files_list)

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


            # counters for loaded/processed scans. Each scan has about 20-30 slices, but we throw away the 7 edge slices on each side
            t = 0  # counts files for checking and processing
            n_PD_scans = 0


            while n_PD_scans < N_wanted_scans:

                print(' === loading h5 file {} === '.format(t))
                # Load k-space data
                filename_h5 = original_files_folder + original_files_list[t]

                print('t=', t)
                print('filename_h5=', filename_h5)
                f = h5py.File(filename_h5, 'r')

                t += 1  # updated the number of LOADED scans

                scan_label = f.attrs['acquisition']
                kspace_orig = np.array(f["kspace"])

                # print some info about the scan
                if (scan_label != 'CORPD_FBK'):
                    print('scan label is not PD')

                if (kspace_orig.shape[2] != 640) | (kspace_orig.shape[3] != 372):
                    print('scan is not 640x372 pixels')

                if (kspace_orig.shape[2] == 640) & (kspace_orig.shape[3] == 372) & (scan_label == 'CORPD_FBK'):
                    print('valid slice - PD scan with size 640x372')
                    kspace = kspace_orig
                    n_PD_scans += 1

                    N_slices = kspace.shape[0]  # number of slices
                    NC = kspace.shape[1]
                    NX = kspace.shape[2]
                    NY = kspace.shape[3]

                    # # define margins - to avoid choosing blocks that are at the image's edges and contain only noise
                    # x_margin = 0.15 * NX
                    # y_margin = 0.15 * NY

                    # define parameters for blocks extraction (margins are used to avoid choosing blocks that are at the image's edges and contain only noise)
                    block_asp_ratio_x = 0.2
                    block_asp_ratio_y = 0.2
                    x_margin = 0.15 * NX
                    y_margin = 0.15 * NY

                    ################ image domain #################
                    # throw out the 8 edge slices on each side of the scan, because they contain mostly noise.
                    slices_to_store_inds = np.arange(8, (N_slices - 8), 1)

                    if data_type == 'val':
                        # for validation data we arbitrarily choose a SINGLE slice from the scan, i.e. we store only one slice per subject.
                        slices_to_store_inds = np.random.randint(8, (N_slices - 8), 1)

                    N_slices_to_store = slices_to_store_inds.shape[0]

                    print(f'slices_to_store_inds={slices_to_store_inds}')
                    print(f'N_slices_to_store={N_slices_to_store}')



                    if (q_i == 0) & (im_type_i==0):
                        n_imgs += N_slices_to_store  # update counter

                        print(f'total number of stored slices: n_imgs={n_imgs}')

                        #print(f'n_PD_scans (valid scans) = {n_PD_scans}')


                    for s_i in range(N_slices_to_store):
                        # print(s_i)
                        s = slices_to_store_inds[s_i]
                        print(f' slice {s}')

                        ksp_block_multicoil = kspace[s, :, :, :].squeeze()

                        im_RSS = merge_multicoil_data(ksp_block_multicoil)

                        # fig = plt.figure()
                        # plt.imshow(im_RSS, cmap="gray")
                        # plt.title('im_RSS full size')
                        # plt.show()



                        if create_blocks_flag==1:
                            # extract a block from im_RSS
                            im_RSS_block = extract_block(im_RSS,block_asp_ratio_x,block_asp_ratio_y,x_margin,y_margin)

                            # fig = plt.figure()
                            # plt.imshow(im_RSS_block,cmap="gray")
                            # plt.title('block')
                            # plt.show()

                            im_RSS = im_RSS_block

                        # scale-down & compress
                        scale_factor = np.max(im_RSS)
                        im_RSS_scaled = im_RSS / scale_factor

                        kspace_NC = sp.fft(im_RSS_scaled, axes=(1, 2)) # NC = No Compression

                        if q==999: # No Compression
                            im_compressed = im_RSS_scaled
                        else:
                            im_compressed = JPEG_compression(im_RSS_scaled, q)

                        kspace_slice = sp.fft(im_compressed, axes=(1, 2))

                        ################### JPEG compression & compute k-space of the JPEG image #####################
                        if s_i==0:
                            # initialize arrays
                            slices_compressed = np.zeros((N_slices_to_store,im_compressed.shape[0],im_compressed.shape[1]), dtype='complex64')
                            kspace_slices_compressed = np.zeros((N_slices_to_store,im_compressed.shape[0],im_compressed.shape[1]), dtype='complex64')

                            # store in arrays
                        slices_compressed[s_i, :, :] = im_compressed
                        kspace_slices_compressed[s_i, :, :] = kspace_slice

                        if (n_PD_scans == 1) & (s_i<=3):
                            fig = plt.figure()
                            plt.subplot(1, 2, 1)
                            plt.imshow(np.rot90(np.abs(im_RSS_scaled),2), cmap="gray")
                            plt.colorbar(shrink=0.4)
                            plt.title('im_RSS_scaled')

                            plt.subplot(1, 2, 2)
                            plt.imshow(np.rot90(np.abs(im_compressed),2), cmap="gray")
                            plt.colorbar(shrink=0.4)
                            plt.title('im_compressed')

                            plt.suptitle(
                                f'{data_type} data; scan {n_PD_scans} slice {s}')
                            plt.show()


                            fig = plt.figure()
                            plt.subplot(1, 2, 1)
                            plt.imshow(np.rot90(np.log(np.abs(kspace_NC)),2),cmap="gray")
                            plt.colorbar(shrink=0.4)
                            plt.clim(-12,3)
                            plt.title('kspace of im_RSS')

                            plt.subplot(1, 2, 2)
                            plt.imshow(np.rot90(np.log(np.abs(kspace_slice)),2),cmap="gray")
                            plt.colorbar(shrink=0.4)
                            plt.clim(-12, 3)
                            plt.title(f'kspace of im_JPEG q={q}')

                            plt.suptitle(
                                f'{data_type} data; scan {n_PD_scans} slice {s}')
                            plt.show()


                    ################## saving -  all slices extracted from a single h5 file are saved together  =================

                    if create_small_dataset_flag == 1:
                        basic_out_folder = basic_out_folder + '_small/'
                    else:
                        basic_out_folder = basic_out_folder + '/'

                    out_folder = basic_out_folder + data_type + "/q" + str(q) + "/" + im_type_str + "/"

                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)

                    # Save data in h5 file
                    h5f = h5py.File(out_folder + '%d.h5' % (n_PD_scans), 'w')
                    h5f.create_dataset('kspace', data=kspace_slices_compressed)
                    h5f.create_dataset('img_jpeg', data=slices_compressed)
                    h5f.close()

                    if (t == (N_available_scans - 1)) | (n_PD_scans == N_wanted_scans):
                        break


            # update counters
            if data_i == 0:  # train data
                N_PD_scans_train = n_PD_scans
                N_imgs_train = n_imgs
            elif data_i == 1:  # val data
                N_PD_scans_val = n_PD_scans
                N_imgs_val = n_imgs
            elif data_i == 2:  # test data
                N_PD_scans_test = n_PD_scans
                N_imgs_test = n_imgs


    ########### save meta-data ##################

    print('N_PD_scans_train: ', N_PD_scans_train)
    print('N_PD_scans_val: ', N_PD_scans_val)
    print('N_PD_scans_test: ', N_PD_scans_test)
    print('N_imgs_train: ', N_imgs_train)
    print('N_imgs_val: ', N_imgs_val)
    print('N_imgs_test: ', N_imgs_test)

meta_filename = basic_out_folder + '/metadata'
np.savez(meta_filename, N_PD_scans_train=N_PD_scans_train, N_PD_scans_val=N_PD_scans_val,
         N_PD_scans_test=N_PD_scans_test, N_imgs_train=N_imgs_train, N_imgs_val=N_imgs_val, N_imgs_test=N_imgs_test)
