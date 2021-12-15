"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pathlib
import random
import torch

import h5py
from torch.utils.data import Dataset, DataLoader
import sigpy as sp
import matplotlib.pyplot as plt

from subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils.subsample_fastmri import MaskFunc
from subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils.subsample_var_dens import MaskFuncVarDens_1D,MaskFuncVarDens_2D
from subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils import complex_utils as cplx



# def calc_scaling_factor(kspace):
#     im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace, (640, 24)), (640, 372))))
#     magnitude_vals = im_lowres.reshape(-1)
#     k = int(round(0.05 * magnitude_vals.shape[0]))
#     scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
#     return scale


class SliceData(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            #self.examples += [(fname, slice+8) for slice in range(num_slices-16)]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data['img_jpeg'][slice]
            return self.transform(kspace,target,slice)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()
        self.sampling_flag = args.sampling_flag

    def __call__(self, kspace, target, slice):

        # # compute a scaling factor for scaling the image intensity to a range of approximately [0,1]
        # #scale = calc_scaling_factor(kspace)
        # fig = plt.figure()
        # plt.imshow(np.abs(target),cmap="gray")
        # plt.colorbar()
        # plt.show()
        # fig.savefig('im_target_before_scaling_debuggin')
        #
        # fig = plt.figure()
        # plt.imshow(np.log(np.abs(kspace)),cmap="gray")
        # plt.colorbar()
        # plt.show()
        # fig.savefig('kspace_before_scaling_debuggin')
        #
        # im_fully_sampled = abs(sp.ifft(kspace, center=True))
        #
        # fig = plt.figure()
        # plt.imshow(np.abs(im_fully_sampled),cmap="gray")
        # plt.colorbar()
        # plt.show()
        # fig.savefig('im_fully_sampled_before_scaling_debuggin')


        # # Compute a scaling factor for scaling the image intensity to a range of approximately [0,1]

        # # scaling method for subsampled data - CHECK that there aren't any fftshift problems here with sp.ifft
        # # Notice that the scaling is performed based on the middle 24 columns
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace, (640, 32)), (640, 372)),center=False))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]

        # # scaling method for fully sampled data
        # #im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace, (640, 24)), (640, 372)),center=True))
        # im_fully_sampled = np.abs(np.fft.ifft2(kspace))
        # magnitude_vals = im_fully_sampled.reshape(-1)
        # k = int(round(0.05 * magnitude_vals.shape[0]))  # here we take a value of ~95% of the max intensity value (95% of sorted distribution)
        # scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        #
        #
        # #return scale
        #
        # # #debugging
        # im_target_before_scaling = target
        # # target_max_before_scaling = np.max(np.abs(target))
        # # print('target_max_before_scaling =',target_max_before_scaling )
        #
        kspace = kspace / scale

        # NOTICE - THE TARGET IS ASSUMED TO BE NORMALIZED ALREADY TO [0,1]
        target = target / scale

        # # target_max_after_scaling = np.max(np.abs(target))
        # # print('target_max_after_scaling =',target_max_after_scaling )
        #
        # # display - for debugging
        # fig = plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(np.abs(im_fully_sampled), cmap="gray")
        # plt.colorbar()
        # plt.title('im_fully_sampled')
        #
        # plt.subplot(1,3,2)
        # plt.imshow(np.abs(im_target_before_scaling ), cmap="gray")
        # plt.title('target_before_scaling')
        # plt.colorbar()
        #
        # plt.subplot(1,3,3)
        # plt.imshow(np.abs(target),cmap="gray")
        # plt.colorbar()
        # plt.title('target')
        # plt.show()
        # fig.savefig('scaling_monitoring_fig')
        #
        # print('scaling monitoring fig saved')
        # #print('stop for debug')

        # Convert everything from numpy arrays to tensors
        kspace_torch = cplx.to_tensor(kspace).float()
        target_torch = cplx.to_tensor(target).float()
        #mask_slice = np.ones((640, 372))
        mask_slice = np.ones((kspace.shape[0], kspace.shape[1]))

        # call mask_func
        if self.sampling_flag == 'random_uniform_1D':
            mk1 = self.mask_func((1, 1, kspace.shape[1], 2))[0, 0, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch

        elif self.sampling_flag == 'var_dens_1D':
            mk1 = self.mask_func((1, 1, kspace.shape[1], 2))[0, 0, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch

        elif self.sampling_flag == 'var_dens_2D':
            mk1 = self.mask_func((1, kspace.shape[0], kspace.shape[1], 2))[0, :, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch

        # # debugging checks
        # #target = np.ifft
        # kspace_sampled = knee_masks * kspace
        # target_new = np.abs(sp.ifft(kspace,center=True))
        # im_zf = np.abs(sp.ifft(kspace_sampled,center=True))

        # fig = plt.figure()
        # plt.subplot(2,3,1)
        # plt.imshow(np.log(np.abs(kspace)))
        # plt.title('kspace scaled')
        #
        # plt.subplot(2,3,2)
        # plt.imshow(np.abs(target))
        # plt.colorbar(shrink=0.5)
        # plt.title('target scaled')
        #
        # plt.subplot(2,3,3)
        # plt.imshow(np.abs(target_new))
        # plt.colorbar(shrink=0.5)
        # plt.title('target_new')
        #
        # plt.subplot(2,3,4)
        # plt.imshow(np.log(np.abs(kspace_sampled)))
        # plt.title('kspace_sampled')
        #
        # plt.subplot(2,3,5)
        # plt.imshow(np.abs(im_zf))
        # plt.colorbar(shrink=0.5)
        # plt.title('im_zf')
        # plt.show()
        # fig.savefig('monitoring_kspace_scaling.png')
        # print('saved monitoring_kspace_scaling.png')
        #

        # print('stop')


        return kspace_torch, target_torch, mask_torch


def create_datasets(args):
    # Generate undersampling mask

    calib = args.calib
    R = args.R

    pad_ratio = np.array([1]) # there is no zero-padding in the JPEG crime experiments
    var_dens_flag = args.var_dens_flag

    if args.sampling_flag == 'random_uniform_1D':
        train_mask = MaskFunc([calib / 372], [R])  # random-uniform mask

    elif args.sampling_flag == 'var_dens_1D':
        train_mask = MaskFuncVarDens_1D(calib, R, pad_ratio, var_dens_flag)  # variable-density mask

    elif args.sampling_flag == 'var_dens_2D':
        train_mask = MaskFuncVarDens_2D(calib, R, pad_ratio, var_dens_flag)  # variable-density mask

    # print('train_mask shape:',train_mask.shape)

    # Explanation from utils.datasets.SliceData:
    # SliceData is a generic PyTorch Dataset class that provides access to 2D MR image slices.
    # Its inputs args are:
    #    root (pathlib.Path): Path to the dataset.
    #    transform (callable): A callable object that pre-processes the raw data into
    #           appropriate form. The transform function should take 'kspace', 'target',
    #           attributes', 'filename', and 'slice' as inputs. 'target' may be null for test data.
    #    sample_rate (float, optional): A float between 0 and 1. This controls what fraction
    #                 of the volumes should be loaded.

    train_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(train_mask, args),
        sample_rate=1
    )
    return train_data


def create_data_loaders(args):
    train_data = create_datasets(args)
    # print('train data shape:', train_data.shape)
    print('type(train data):', type(train_data))
    #     print(train_data[0])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        #shuffle=True,
        shuffle=args.shuffle_flag,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader



